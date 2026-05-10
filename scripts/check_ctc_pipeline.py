import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

from src.ctc.config import CTCConfig as C
from src.ctc.dataset import CTCDataset, ctc_collate_fn
from src.ctc.augmentation import SpecAugment
from src.ctc.model import CTCModel
from src.ctc.training import train_ctc
from src.ctc.metrics import greedy_decode, decode_to_phonemes, compute_per


def log(message: str) -> None:
    print(f"[check] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end CTC pipeline check (data -> train -> save/load -> decode)."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to PSD data root (will be searched if not provided).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=4,
        help="Limit number of files used for the check.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--cache-mode",
        action="store_true",
        help="Precompute and cache spectrograms in RAM",
    )
    parser.add_argument(
        "--apply-aug",
        action="store_true",
        help="Precompute waveform augmentations into cache",
    )
    parser.add_argument("--n-mels", type=int, default=C.N_MELS)
    parser.add_argument("--n-fft", type=int, default=C.N_FFT)
    parser.add_argument("--hop-length", type=int, default=C.HOP_LENGTH)

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)

    return parser.parse_args()


def find_data_dir(base_dir: str) -> str | None:
    """
    Try to find a PSD-like data directory under ./data.
    Adjust candidates to your actual layout if needed.
    """
    candidates = [
        os.path.join(base_dir, "data", "1-500"),
        os.path.join(base_dir, "data", "501-1000"),
        os.path.join(base_dir, "data", "1001-1500"),
        os.path.join(base_dir, "data"),  # fallback
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_dir = args.data_dir or find_data_dir(project_root)
    if data_dir is None:
        raise SystemExit("No data directory found under ./data")

    device = torch.device(args.device)

    log("starting CTC pipeline check")
    log(f"data_dir: {data_dir}")
    log(f"device: {device}")

    # 1. Dataset construction (no dataset-level augmentation here)
    dataset = CTCDataset(
        data_root=data_dir,
        cache_mode=args.cache_mode,
        apply_augmentations=args.apply_aug,
        max_files=args.max_files,
        sample_rate=C.SAMPLE_RATE,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        standardize=True,
        noise_prob=0.5,
        gain_prob=0.5,
        tempo_prob=0.2,
        noise_level=(10.0, 30.0),
        gain_range=(-5.0, 5.0),
        tempo_range=(0.95, 1.05),
    )

    n_total = len(dataset)
    if n_total == 0:
        raise SystemExit("No files found in the dataset")
    log(f"total items loaded (mel, target pairs): {n_total}")

    # 2. Train/Val split
    if n_total < 2:
        log("dataset too small for split, using the same set for train/val")
        train_set = dataset
        val_set = dataset
    else:
        n_val = max(1, int(0.2 * n_total))
        n_train = n_total - n_val
        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(0),
        )
        log(f"train items: {len(train_set)}, val items: {len(val_set)}")

    # 3. DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # 4. Model & training components
    model = CTCModel(n_mels=args.n_mels)

    criterion = torch.nn.CTCLoss(blank=C.N_CLASSES, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.2,
    )

    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    mel_augmenter = SpecAugment(freq_mask_percent=0.2, time_mask_percent=0.125, p=0.5)

    # 5. Training with temporary checkpoint.
    # The best checkpoint is now selected by lowest validation PER, not CTC loss.
    log("training model with AMP and CTCLoss (best model selected by val PER)")
    tmp_ckpt = os.path.join(project_root, "trained_models", "check_pipeline_ctc_tmp.pt")

    try:
        model = train_ctc(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_epochs=args.epochs,
            spec_augment=mel_augmenter,
            scheduler=scheduler,
            scaler=scaler,
            save_best_to=tmp_ckpt,
            use_amp=(device.type == "cuda"),
            grad_clip_norm=2.0,
            checkpoint_config={
                "source": "check_pipeline",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        )
        log("CTC training completed")

        # 6. Reload model from file and run inference/decoding
        if not os.path.exists(tmp_ckpt):
            raise SystemExit(f"Expected checkpoint {tmp_ckpt} not found")

        log(f"loading model from checkpoint: {tmp_ckpt}")
        reloaded_model = CTCModel().to(device)
        checkpoint = torch.load(tmp_ckpt, map_location=device)
        reloaded_model.load_state_dict(checkpoint["model_state_dict"])
        reloaded_model.eval()

        # Take one batch from train_loader
        mels, targets, in_lens, t_lens = next(iter(train_loader))

        with torch.no_grad():
            mels_dev = mels.to(device)
            logits = reloaded_model(mels_dev)

        log(f"Input batch shape   (B, Mels, Time): {tuple(mels.shape)}")
        log(f"Output logits shape (B, Time', Cls): {tuple(logits.shape)}")
        log(f"Length tensor shape  (B): {tuple(in_lens.shape)}")

        # Decode batch with greedy CTC
        cpu_logits = logits.cpu()
        decoded_batch = greedy_decode(cpu_logits)

        # Unpad targets for PER computation
        unpadded_targets: list[list[int]] = []
        for b_idx in range(targets.shape[0]):
            length = t_lens[b_idx].item()
            unpadded_targets.append(targets[b_idx, :length].tolist())

        per = compute_per(decoded_batch, unpadded_targets)
        log(f"Batch PER estimate: {per:.4f}")

        # Show target vs prediction for first item
        log("Sample decoding comparison (1st item):")
        log(f"  Target:      {decode_to_phonemes(unpadded_targets[0])}")
        log(f"  Prediction:  {decode_to_phonemes(decoded_batch[0])}")

        # Verify the saved checkpoint records the val_loss at the best-PER epoch
        saved_val_loss = checkpoint.get("val_loss")
        if saved_val_loss is not None:
            log(f"Checkpoint val CTC loss at best-PER epoch: {saved_val_loss:.4f}")

        log("CTC PIPELINE CHECK OK")

    finally:
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
            log(f"removed temp checkpoint: {tmp_ckpt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
