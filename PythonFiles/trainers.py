import torch
import copy
from pathlib import Path
import torchmetrics
def train_model(model, optimizer, loss_fn, metric, train_loader, valid_loader,
                n_epochs, patience=2, factor=0.5, device=None,
                early_stop_patience=5, save_best_to=None,
                grad_clip=None, epoch_callback=None, verbose=True):
    """
    Train with LR scheduling, early stopping, best-model checkpointing.

    Args:
        model: nn.Module
        optimizer: torch optimizer
        loss_fn: loss function (e.g. nn.CrossEntropyLoss())
        metric: torchmetric (must have .update, .compute, .reset)
        train_loader, valid_loader: DataLoaders
        n_epochs: max epochs
        patience: epochs without improvement before LR reduction
        factor: LR reduction factor (0.5 = halve LR)
        device: 'cuda' or 'cpu'
        early_stop_patience: stop after this many epochs without val improvement
                             (set to None to disable)
        save_best_to: path to save best checkpoint (set to None to skip)
        grad_clip: max gradient norm (set to None to skip clipping)
        epoch_callback: optional fn(model, epoch, history) called after each epoch
        verbose: whether to print per-batch progress
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=patience, factor=factor)

    history = {"train_losses": [], "train_metrics": [], "valid_metrics": [],
               "lrs": [], "best_epoch": 0, "best_val": 0.0}
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        # -------- TRAIN --------
        model.train()
        metric.reset()                              # FIX: reset per epoch
        running_loss, running_n = 0.0, 0

        for index, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # weighted running loss (handles last small batch correctly)
            bsz = y_batch.size(0)
            running_loss += loss.item() * bsz
            running_n    += bsz

            metric.update(y_pred.detach(), y_batch)  # detach: avoid graph leak

            if verbose:
                print(f"\rEpoch {epoch}/{n_epochs}  "
                      f"batch {index+1}/{len(train_loader)}  "
                      f"loss={running_loss/running_n:.4f}",
                      end="")

        train_loss   = running_loss / running_n
        train_metric = metric.compute().item()      # compute ONCE at end

        # -------- VALIDATE --------
        val_metric = evaluate_tm(model, valid_loader, metric, device).item()

        # -------- BOOKKEEPING --------
        current_lr = optimizer.param_groups[0]['lr']
        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_metric)
        history["valid_metrics"].append(val_metric)
        history["lrs"].append(current_lr)

        scheduler.step(val_metric)

        # best model tracking
        improved = val_metric > history["best_val"]
        if improved:
            history["best_val"]   = val_metric
            history["best_epoch"] = epoch
            best_state            = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if save_best_to is not None:
                Path(save_best_to).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_metric': val_metric,
                    'history': history,
                }, save_best_to)
        else:
            epochs_without_improvement += 1

        marker = "  ← best" if improved else ""
        print(f"\rEpoch {epoch:3d}/{n_epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"train={train_metric:.2%}  "
              f"val={val_metric:.2%}  "
              f"lr={current_lr:.1e}{marker}" + " " * 20)

        if epoch_callback is not None:
            epoch_callback(model, epoch, history)

        # -------- EARLY STOPPING --------
        if (early_stop_patience is not None
                and epochs_without_improvement >= early_stop_patience):
            print(f"\nEarly stopping: no improvement for "
                  f"{early_stop_patience} epochs.")
            break

    # restore best weights at end
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model from epoch {history['best_epoch']} "
              f"(val={history['best_val']:.2%})")

    return history


def evaluate_tm(model, data_loader, metric, device):
    """Evaluate metric on a loader. Returns the metric tensor."""
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()