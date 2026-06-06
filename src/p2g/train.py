"""
Fine-tune the P2G seq2seq model on (phone-string, text) pairs.

A plain PyTorch loop (no HF Trainer) keeps control simple and avoids
TrainingArguments API churn across transformers versions. Runs in fp32 — T5
fine-tuning is unstable under fp16/AMP.
"""

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import P2GConfig as P
from .metrics import corpus_cer, corpus_wer
from .model import P2GModel
from src.utils.device import get_device


def make_collate(tokenizer, task_prefix: str, max_source: int, max_target: int):
    """Collate raw rows into a tokenized seq2seq batch (labels padded with -100)."""

    def collate(batch: list[dict]) -> dict:
        sources = [task_prefix + b["phones"] for b in batch]
        targets = [b["text"] for b in batch]
        enc = tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=max_source,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=max_target,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc

    return collate


def evaluate(
    p2g: P2GModel,
    rows: list[dict],
    batch_size: int = 16,
    num_beams: int = 4,
) -> dict:
    """Generate over ``rows`` (batched) and return preds/refs plus corpus WER & CER."""
    preds: list[str] = []
    refs = [r["text"] for r in rows]
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        preds.extend(p2g.generate([r["phones"] for r in chunk], num_beams=num_beams))
    return {
        "preds": preds,
        "refs": refs,
        "wer": corpus_wer(preds, refs),
        "cer": corpus_cer(preds, refs),
    }


def train_p2g(
    train_rows: list[dict],
    val_rows: list[dict] | None = None,
    model_name: str = P.MODEL_NAME,
    output_dir: str | None = None,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    num_beams: int = 4,
    device: torch.device | None = None,
    eval_batch_size: int = 16,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    early_stopping_patience: int | None = None,
) -> tuple[P2GModel, dict]:
    """
    Fine-tune a fresh P2G model and (optionally) save it.

    Robustness:
      * Linear-warmup → linear-decay LR schedule (``warmup_ratio`` of total steps),
        stepped per batch — more stable than a flat LR for T5 fine-tuning.
      * Model selection by **validation WER**: the best epoch's weights are kept and
        restored at the end (and saved to ``output_dir``), so an overfit final epoch
        never gets shipped. Requires ``val_rows``; without it the final epoch is used.
      * Optional early stopping after ``early_stopping_patience`` epochs without a
        val-WER improvement (``None`` disables it).

    Returns the model and the metrics dict of the selected (best) epoch.
    """
    device = device or get_device()
    p2g = P2GModel(model_name=model_name, device=device)
    collate = make_collate(
        p2g.tokenizer, p2g.task_prefix, p2g.max_source_len, p2g.max_target_len
    )

    loader = DataLoader(
        train_rows, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(
        p2g.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    metrics: dict = {}
    best_wer = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        p2g.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = p2g.model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p2g.model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        lr_now = scheduler.get_last_lr()[0]
        msg = f"[p2g] epoch {epoch}/{epochs}  train_loss {avg_loss:.4f}  lr {lr_now:.2e}"

        if val_rows:
            epoch_metrics = evaluate(
                p2g, val_rows, batch_size=eval_batch_size, num_beams=num_beams
            )
            improved = epoch_metrics["wer"] < best_wer
            if improved:
                best_wer = epoch_metrics["wer"]
                best_epoch = epoch
                metrics = epoch_metrics
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in p2g.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            marker = "  [BEST]" if improved else ""
            msg += (
                f"  val_WER {epoch_metrics['wer']:.3f}"
                f"  val_CER {epoch_metrics['cer']:.3f}{marker}"
            )
            print(msg)
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(f"[p2g] early stopping: no val-WER gain for {early_stopping_patience} epochs")
                break
        else:
            print(msg)

    if best_state is not None:
        p2g.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"[p2g] restored best epoch {best_epoch} (val_WER {best_wer:.3f})")

    if output_dir:
        p2g.save(output_dir)
        print(f"[p2g] saved fine-tuned model to {output_dir}")

    return p2g, metrics
