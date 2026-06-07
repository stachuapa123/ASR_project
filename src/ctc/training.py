from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CTCConfig as C
from .metrics import greedy_decode, compute_per


class EarlyStopping:
    """Stop training when `val_loss` doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Call after each epoch. Returns True if training should stop."""

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


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
    use_amp: bool | None = None,
    feature_extractor: torch.nn.Module | None = None,
) -> tuple[float, float]:
    """Run one validation epoch -> (mean CTC loss, PER).

    With ``feature_extractor``, batches are waveforms (features built on-device, no
    aug); otherwise precomputed mels.
    """
    model.eval()
    if feature_extractor is not None:
        feature_extractor.eval()
    total_loss = 0.0
    total_samples = 0

    # collect preds + targets for PER
    all_preds: list[list[int]] = []
    all_targets: list[list[int]] = []

    if use_amp is None:
        use_amp = device.type == "cuda"

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                if feature_extractor is not None:
                    # inputs are raw waveforms; build mels + adjusted lengths
                    mels, mel_lengths = feature_extractor(
                        inputs, input_lengths, augment=False
                    )
                    adj_input_lengths = mel_lengths // time_reduction_factor
                else:
                    mels = inputs
                    adj_input_lengths = input_lengths // time_reduction_factor

                logits = model(mels)  # (B, T', C+1)

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.transpose(0, 1)  # (T', B, C+1)

                loss = criterion(
                    log_probs,
                    targets,
                    adj_input_lengths,
                    target_lengths,
                )

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # greedy-decode valid frames; collect targets for PER
            decoded = greedy_decode(logits, lengths=adj_input_lengths)
            all_preds.extend(decoded)
            all_targets.extend(
                [t[:l].tolist() for t, l in zip(targets.cpu(), target_lengths.cpu())]
            )

    mean_ctc_loss = total_loss / max(total_samples, 1)
    per = compute_per(all_preds, all_targets)
    return mean_ctc_loss, per


def train_ctc(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    objective: torch.nn.Module,
    device: torch.device,
    n_epochs: int,
    spec_augment: Any | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
    save_best_to: str | Path | None = None,
    time_reduction_factor: int = C.TIME_REDUCTION_FACTOR,
    use_amp: bool | None = None,
    grad_clip_norm: float | None = 2.0,
    step_scheduler_per_batch: bool = True,
    checkpoint_config: dict[str, Any] | None = None,
    feature_extractor: torch.nn.Module | None = None,
) -> torch.nn.Module:
    """CTC training loop; model selection (early stop + best checkpoint) by val PER.

    With ``feature_extractor``, loaders yield raw waveforms and features+augmentation
    run on-device (the standalone ``spec_augment`` arg is then ignored).
    """
    model.to(device)
    if feature_extractor is not None:
        feature_extractor.to(device)

    if use_amp is None:
        use_amp = device.type == "cuda"

    # default scaler: enabled on CUDA, off on CPU
    if scaler is None:
        scaler = torch.amp.GradScaler(
            device=device.type,
            enabled=use_amp,
        )

    best_val_per = float("inf")
    best_state: dict[str, Any] | None = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(
            train_loader
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                if feature_extractor is not None:
                    # raw waveforms -> mels (+augmentation) on device
                    mels, mel_lengths = feature_extractor(
                        inputs, input_lengths, augment=True
                    )
                    adj_input_lengths = mel_lengths // time_reduction_factor
                else:
                    mels = inputs
                    if spec_augment is not None:
                        mels = spec_augment(mels)
                    adj_input_lengths = input_lengths // time_reduction_factor

                logits = model(mels)  # (B, T', C+1)

                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.transpose(0, 1)  # (T', B, C+1)

                loss = objective(
                    log_probs,
                    targets,
                    adj_input_lengths,
                    target_lengths,
                )

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )

            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # don't step the scheduler when AMP skipped the optimizer step
            optimizer_stepped = scaler.get_scale() >= prev_scale
            if scheduler is not None and step_scheduler_per_batch and optimizer_stepped:
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
        val_loss, val_per = evaluate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=objective,
            device=device,
            time_reduction_factor=time_reduction_factor,
            use_amp=use_amp,
            feature_extractor=feature_extractor,
        )

        # select + early-stop on PER
        improved = val_per < best_val_per
        if improved:
            best_val_per = val_per
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
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
            f"Val PER: {val_per:.4f} | "
            f"LR: {lr:.1e}{marker}" + " " * 10
        )

        if early_stopping is not None and early_stopping.step(val_per):
            print(f"\nEarly stopping triggered after epoch {epoch}.")
            break

        if scheduler is not None and not step_scheduler_per_batch:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights with Val PER = {best_val_per:.4f}")

    return model
