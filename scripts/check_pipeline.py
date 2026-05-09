import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.constants import Constants as C
from src.parsers import build_augmented_cache_soft, SoftLabelCacheDataset
from src.augment import SpecAugment
from src.NeuralModel import CRNN
from src.trainers import (
    train_model,
    evaluate_tm,
    save_checkpoint,
    load_checkpoint,
    SoftCrossEntropyLoss,
    SoftAccuracy,
)
from src.evaluator import evaluate_audio


def log(message):
    print(f"[check] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline check with no required arguments."
    )
    parser.add_argument("--data-dir", default=None, help="Path to PSD data split")
    parser.add_argument("--max-files", type=int, default=1, help="Limit files used")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
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


def find_first_pair(data_dir):
    for tg_path in sorted(Path(data_dir).rglob("*.TextGrid")):
        wav_path = tg_path.with_suffix(".wav")
        if wav_path.exists():
            return str(wav_path), str(tg_path)
    return None, None


def main():
    args = parse_args()
    torch.manual_seed(0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = args.data_dir or find_data_dir(project_root)
    if data_dir is None:
        raise SystemExit("No PSD data directory found under ./data")

    log("starting pipeline check")
    log(f"data_dir: {data_dir}")
    log(f"device: {args.device}")

    dataset = None  # Replaced by soft cache
    log(f"generating soft-label cache in RAM")
    data = build_augmented_cache_soft(
        data_dir=data_dir,
        output_path=None,
        n_augmentations=4,
        standardize=True,
        silences_same=True,
        max_files=args.max_files,
        noiseprob=0.5,
        gainprob=0.5,
        tempo_prob=0.0,
        noise_level=(10, 30),
        gain_range=(-5, 5),
        tempo_range=(0.95, 1.05),
    )
    X_data = data["X"]
    y_data = data["y"]
    n_total = len(y_data)

    if n_total == 0:
        raise SystemExit("No windows produced from the dataset")
    log(f"total windows: {n_total}")

    if n_total < 2:
        log("dataset too small for split, using the same set for train/val")
        train_idx = list(range(n_total))
        val_idx = list(range(n_total))
    else:
        n_val = max(1, int(0.2 * n_total))
        n_train = n_total - n_val
        indices = torch.randperm(n_total).tolist()
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        log(f"train windows: {n_train}, val windows: {n_val}")

    train_set = SoftLabelCacheDataset(
        X_data[train_idx], y_data[train_idx], train=True, device="cpu"
    )
    val_set = SoftLabelCacheDataset(
        X_data[val_idx], y_data[val_idx], train=False, device="cpu"
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = CRNN().to(args.device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)
    loss_fn = SoftCrossEntropyLoss(label_smoothing=0.05)
    metric = SoftAccuracy(num_classes=C.N_CLASSES).to(args.device)
    mel_augmenter = SpecAugment(freq_mask_percent=0.2, time_mask_percent=0.125, p=0.5)

    log("training model")
    history = train_model(
        model,
        optimizer,
        loss_fn,
        metric,
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        patience=1,
        factor=0.5,
        device=args.device,
        early_stop_patience=1,
        verbose=args.verbose,
        spec_augment=mel_augmenter,
    )
    log("training completed")

    val_metric = evaluate_tm(model, val_loader, metric, args.device).item()
    log(f"val metric: {val_metric:.4f}")

    X_batch, y_batch = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        logits = model(X_batch.to(args.device))
        preds = logits.argmax(dim=1).cpu()
        y_batch_labels = y_batch.argmax(dim=1).cpu()
        batch_acc = (preds == y_batch_labels).float().mean().item()
    log(f"batch shapes: X={tuple(X_batch.shape)}, y={tuple(y_batch.shape)}")
    log(f"logits shape: {tuple(logits.shape)}")
    log(f"batch accuracy: {batch_acc:.4f}")

    tmp_ckpt = os.path.join(project_root, "trained_models", "check_pipeline_tmp.pth")
    config = {
        "sample_rate": C.SAMPLE_RATE,
        "n_fft": C.N_FFT,
        "hop_length": C.HOP_LENGTH,
        "n_mels": C.N_MELS,
        "win_frames": C.WIN_FRAMES,
        "shift_frames": C.SHIFT_FRAMES,
        "hidden": 64,
    }

    try:
        log(f"saving checkpoint: {tmp_ckpt}")
        save_checkpoint(
            tmp_ckpt,
            model,
            labels=C.LABELS,
            config=config,
            history=history,
        )

        log("loading checkpoint into a fresh model")
        reloaded = CRNN()
        meta = load_checkpoint(tmp_ckpt, reloaded, device=args.device)
        log(f"checkpoint meta keys: {list(meta.keys())}")

        reloaded.eval()
        with torch.no_grad():
            logits2 = reloaded(X_batch.to(args.device))
        log(f"reloaded logits shape: {tuple(logits2.shape)}")

        wav_path, tg_path = find_first_pair(data_dir)
        if wav_path is not None:
            log(f"running evaluate_audio on: {wav_path}")
            evaluate_audio(
                wav_path,
                textgrid_path=tg_path,
                model=reloaded,
                device=args.device,
                show_per_window=False,
                top_k=3,
            )
        else:
            log("no wav/TextGrid pair found for evaluate_audio")
    finally:
        if os.path.exists(tmp_ckpt):
            os.remove(tmp_ckpt)
            log(f"removed temp checkpoint: {tmp_ckpt}")

    log("PIPELINE CHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
