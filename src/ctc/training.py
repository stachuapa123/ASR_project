from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CTCConfig as C


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: dict[str, Any] | None = None,
    epoch: int | None = None,
    val_loss: float | None = None,
) -> None:
    """
    Save a training checkpoint.

    The caller decides what config / metadata to attach.
    """
    payload: dict[str, Any] = {"model_state_dict": model.state_dict()}

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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def evaluate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    time_reduction_factor: int = C.TIME_REDUCTION_FACTOR,
) -> float:
    """
    Run one validation epoch and return mean CTC loss.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for mels, targets, input_lengths, target_lengths in val_loader:
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device_type,
                enabled=(device_type == "cuda"),
            ):
                logits = model(mels)  # (B, T', C+1)

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.transpose(0, 1)  # (T', B, C+1)

                adj_input_lengths = input_lengths // time_reduction_factor

                loss = criterion(
                    log_probs,
                    targets,
                    adj_input_lengths,
                    target_lengths,
                )

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def train_ctc(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    n_epochs: int,
    spec_augment: Any | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    save_best_to: str | Path | None = None,
    time_reduction_factor: int = C.TIME_REDUCTION_FACTOR,
    checkpoint_config: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """
    Generic CTC training loop.

    The caller provides:
        - optimizer (e.g. AdamW, SGD, ...)
        - criterion (typically CTCLoss)
        - optional scheduler (e.g. OneCycleLR, CosineAnnealingLR, ...)
        - optional GradScaler (for mixed precision on CUDA)
    """
    model.to(device)

    device_type = "cuda" if device.type == "cuda" else "cpu"

    # If caller did not pass a scaler, create a "no-op" one for CUDA, or disable on CPU.
    if scaler is None:
        scaler = torch.amp.GradScaler(
            device=device_type,
            enabled=(device_type == "cuda"),
        )

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for batch_idx, (mels, targets, input_lengths, target_lengths) in enumerate(
            train_loader
        ):
            mels = mels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            if spec_augment is not None:
                mels = spec_augment(mels)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device_type,
                enabled=(device_type == "cuda"),
            ):
                logits = model(mels)  # (B, T', C+1)

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.transpose(0, 1)  # (T', B, C+1)

                adj_input_lengths = input_lengths // time_reduction_factor

                loss = criterion(
                    log_probs,
                    targets,
                    adj_input_lengths,
                    target_lengths,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_count += batch_size

            print(
                f"\rEpoch {epoch:3d}/{n_epochs} | "
                f"Batch {batch_idx + 1:4d}/{len(train_loader)} | "
                f"CTC Loss: {running_loss / running_count:.4f}",
                end="",
            )

        train_loss = running_loss / max(running_count, 1)

        val_loss = evaluate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            time_reduction_factor=time_reduction_factor,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = model.state_dict()
            if save_best_to is not None:
                save_checkpoint(
                    save_best_to,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=checkpoint_config,
                    epoch=epoch,
                    val_loss=val_loss,
                )

        lr = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else optimizer.param_groups[0]["lr"]
        )
        marker = " [BEST]" if improved else ""
        print(
            f"\rEpoch {epoch:3d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.1e}{marker}" + " " * 10
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights with Val Loss = {best_val_loss:.4f}")

    return model
