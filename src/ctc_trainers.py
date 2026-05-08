import torch
import torch.nn.functional as F
import copy
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader

from .constants import Constants as C


def save_ctc_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: dict[str, Any] | None = None,
    epoch: int | None = None,
    val_loss: float | None = None,
) -> None:
    payload = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        payload["config"] = config
    if epoch is not None:
        payload["epoch"] = epoch
    if val_loss is not None:
        payload["val_loss"] = val_loss

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def evaluate_ctc(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: str | torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    with torch.no_grad():
        for mels, targets, input_lengths, target_lengths in val_loader:
            mels, targets = mels.to(device), targets.to(device)
            input_lengths, target_lengths = (
                input_lengths.to(device),
                target_lengths.to(device),
            )

            with torch.amp.autocast(
                device_type=device_type, enabled=(device_type == "cuda")
            ):
                logits = model(mels)

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_transposed = log_probs.transpose(0, 1)

                adj_input_lengths = input_lengths // 4

                loss = criterion(
                    log_probs_transposed, targets, adj_input_lengths, target_lengths
                )

            bsz = targets.size(0)
            total_loss += loss.item() * bsz
            total_samples += bsz

    return total_loss / total_samples


def train_ctc_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    device: str | torch.device,
    spec_augment: Any | None = None,
    save_best_to: str | Path | None = None,
) -> torch.nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.2,
    )

    criterion = torch.nn.CTCLoss(blank=C.N_CLASSES, zero_infinity=True)
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    scaler = torch.amp.GradScaler(device=device_type, enabled=(device_type == "cuda"))

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0

        for batch_idx, (mels, targets, input_lengths, target_lengths) in enumerate(
            train_loader
        ):
            mels, targets = mels.to(device), targets.to(device)
            input_lengths, target_lengths = (
                input_lengths.to(device),
                target_lengths.to(device),
            )

            if spec_augment is not None:
                mels = spec_augment(mels)

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=device_type, enabled=(device_type == "cuda")
            ):
                logits = model(mels)

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.transpose(0, 1)

                adj_input_lengths = input_lengths // 4

                loss = criterion(log_probs, targets, adj_input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bsz = targets.size(0)
            running_loss += loss.item() * bsz
            running_n += bsz

            print(
                f"\rEpoch {epoch:3d}/{n_epochs} | Batch {batch_idx + 1}/{len(train_loader)} | CTC Loss: {running_loss / running_n:.4f}",
                end="",
            )

        train_loss = running_loss / running_n

        val_loss = evaluate_ctc(model, val_loader, criterion, device)

        improved = val_loss < best_val_loss
        marker = "  [Best]" if improved else ""
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            if save_best_to:
                save_ctc_checkpoint(
                    save_best_to,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    val_loss=val_loss,
                )

        print(
            f"\rEpoch {epoch:3d}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}{marker}"
            + " " * 10
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights with Val Loss: {best_val_loss:.4f}")

    return model
