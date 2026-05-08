import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from src.constants import Constants as C
from src.ctc_parsers import CTCDataset, ctc_collate_fn
from src.augment import SpecAugment
from src.CTCModel import CTC_CRNN
from src.ctc_trainers import train_ctc_model
from src.ctc_evaluator import greedy_decode, decode_to_phonemes, compute_per


def log(message):
    print(f"[check] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end CTC pipeline check with no required arguments."
    )
    parser.add_argument("--data-dir", default=None, help="Path to PSD data split")
    parser.add_argument("--max-files", type=int, default=4, help="Limit files used")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def find_data_dir(base_dir):
    candidates = [
        os.path.join(base_dir, "data", "1-500"),
        os.path.join(base_dir, "data", "501-1000"),
        os.path.join(base_dir, "data", "1001-1500"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def main():
    args = parse_args()
    torch.manual_seed(0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = args.data_dir or find_data_dir(project_root)
    if data_dir is None:
        raise SystemExit("No PSD data directory found under ./data")

    log("starting CTC pipeline check")
    log(f"data_dir: {data_dir}")
    log(f"device: {args.device}")

    # 1. Dataset Generation
    dataset = CTCDataset(
        data_dir=data_dir,
        cache_mode=True,
        apply_augmentations=True,
        max_files=args.max_files,
        noiseprob=0.5,
        gainprob=0.5,
        tempo_prob=0.2,
        noise_level=(10, 30),
        gain_range=(-5, 5),
        tempo_range=(0.95, 1.05),
    )

    n_total = len(dataset)
    if n_total == 0:
        raise SystemExit("No files found in the dataset")
    log(f"total files loaded: {n_total}")

    # 2. Train/Val Split
    if n_total < 2:
        log("dataset too small for split, using the same set for train/val")
        train_set = dataset
        val_set = dataset
    else:
        n_val = max(1, int(0.2 * n_total))
        n_train = n_total - n_val
        train_set, val_set = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
        )
        log(f"train files: {n_train}, val files: {n_val}")

    # 3. Data Loaders using the specific CTC collation logic
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    # 4. Model & Augmentation Initialization
    model = CTC_CRNN().to(args.device)
    mel_augmenter = SpecAugment(freq_mask_percent=0.2, time_mask_percent=0.125, p=0.5)

    # 5. Training Process
    log("training model with AMP and CTCLoss")
    tmp_ckpt = os.path.join(
        project_root, "trained_models", "check_pipeline_ctc_tmp.pth"
    )

    try:
        model = train_ctc_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=args.epochs,
            device=args.device,
            spec_augment=mel_augmenter,
            save_best_to=tmp_ckpt,
        )
        log("CTC training completed")
    finally:
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
            log(f"removed temp checkpoint: {tmp_ckpt}")

    # 6. Verify Shapes Output and Decoding
    mels, targets, in_lens, t_lens = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        mels = mels.to(args.device)
        logits = model(mels)
        log(f"Input batch shape    (B, Mels, Time): {tuple(mels.shape)}")
        log(f"Output logits shape  (B, Time/4, Cls): {tuple(logits.shape)}")
        log(f"Length tensor shape  (B): {tuple(in_lens.shape)}")

        # Decode a sample
        cpu_logits = logits.cpu()
        decoded_batch = greedy_decode(cpu_logits)

        # We need to unpad the targets for PER computation
        unpadded_targets: list[list[int]] = []
        for b_idx in range(targets.shape[0]):
            length = t_lens[b_idx].item()
            unpadded_targets.append(targets[b_idx, :length].tolist())

        per = compute_per(decoded_batch, unpadded_targets)
        log(f"Batch native PER estimate: {per:.4f}")

        log("Sample decoding comparison (1st item):")
        log(f"  Target: {decode_to_phonemes(unpadded_targets[0])}")
        log(f"  Prediction: {decode_to_phonemes(decoded_batch[0])}")

    log("CTC PIPELINE CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
