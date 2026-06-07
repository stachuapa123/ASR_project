"""Fine-tune the P2G seq2seq on (phone-string, text) pairs.

bf16 autocast on CUDA (fp16 underflows T5). Tokenized once up front, except under
phone-noise augmentation (re-corrupted + re-tokenized each epoch).
"""

import random

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import P2GConfig as P
from .metrics import corpus_cer, corpus_wer
from .model import P2GModel
from .phone_noise import PhoneNoiseProfile
from src.utils.device import get_device


def tokenize_rows(
    tokenizer, rows: list[dict], task_prefix: str, max_source: int, max_target: int
) -> list[dict]:
    """Tokenize each row once (no padding) -> encoder/decoder id lists."""
    examples: list[dict] = []
    for r in rows:
        enc = tokenizer(task_prefix + r["phones"], truncation=True, max_length=max_source)
        labels = tokenizer(
            text_target=r["text"], truncation=True, max_length=max_target
        )["input_ids"]
        examples.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }
        )
    return examples


def pad_batch(tokenizer):
    """Collate pre-tokenized examples: pad inputs via the tokenizer, labels with -100."""

    def collate(batch: list[dict]) -> dict:
        enc = tokenizer.pad(
            [
                {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                for b in batch
            ],
            padding=True,
            return_tensors="pt",
        )
        max_len = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, : len(b["labels"])] = torch.tensor(b["labels"], dtype=torch.long)
        enc["labels"] = labels
        return enc

    return collate


def corrupt_rows(
    rows: list[dict], profile: PhoneNoiseProfile, rng: random.Random, sep: str = P.PHONE_SEP
) -> list[dict]:
    """Copies of ``rows`` with phone strings corrupted via the CTC error profile."""
    out: list[dict] = []
    for r in rows:
        labels = r["phones"].split(sep) if r["phones"] else []
        noisy = profile.corrupt(labels, rng)
        out.append({**r, "phones": sep.join(noisy)})
    return out


def _make_loader(p2g: P2GModel, rows: list[dict], batch_size: int) -> DataLoader:
    examples = tokenize_rows(
        p2g.tokenizer, rows, p2g.task_prefix, p2g.max_source_len, p2g.max_target_len
    )
    return DataLoader(
        examples, batch_size=batch_size, shuffle=True, collate_fn=pad_batch(p2g.tokenizer)
    )


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
    val_num_beams: int = 1,
    use_bf16: bool | None = None,
    noise_profile: PhoneNoiseProfile | None = None,
    noise_seed: int = P.SEED,
) -> tuple[P2GModel, dict]:
    """Fine-tune a fresh P2G model, restore the best-by-val-WER epoch, optionally save.

    Returns (model, best metrics).
    """
    device = device or get_device()
    if use_bf16 is None:
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    p2g = P2GModel(model_name=model_name, device=device)

    # with augmentation the loader is rebuilt each epoch (fresh corruption); else once
    loader = None if noise_profile is not None else _make_loader(p2g, train_rows, batch_size)
    steps_per_epoch = (
        len(loader) if loader is not None else max(1, -(-len(train_rows) // batch_size))
    )
    optimizer = torch.optim.AdamW(
        p2g.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = max(1, steps_per_epoch * epochs)
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
        if noise_profile is not None:
            rng = random.Random(noise_seed + epoch)
            epoch_loader = _make_loader(
                p2g, corrupt_rows(train_rows, noise_profile, rng), batch_size
            )
        else:
            epoch_loader = loader
        total_loss = 0.0
        n_batches = len(epoch_loader)
        for i, batch in enumerate(epoch_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            # bf16 autocast, no GradScaler: bf16 doesn't underflow; fp16 NaNs on T5
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss = p2g.model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p2g.model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # Live per-batch heartbeat (overwritten by the epoch summary below)
            print(
                f"\r[p2g] epoch {epoch}/{epochs} | batch {i + 1:4d}/{n_batches} | "
                f"loss {total_loss / (i + 1):.4f} | lr {scheduler.get_last_lr()[0]:.2e}",
                end="",
            )

        avg_loss = total_loss / max(1, n_batches)
        lr_now = scheduler.get_last_lr()[0]
        msg = f"\r[p2g] epoch {epoch}/{epochs}  train_loss {avg_loss:.4f}  lr {lr_now:.2e}"

        if val_rows:
            epoch_metrics = evaluate(
                p2g, val_rows, batch_size=eval_batch_size, num_beams=val_num_beams
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
            print(msg + " " * 12)  # pad to clear leftover per-batch progress chars
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(f"[p2g] early stopping: no val-WER gain for {early_stopping_patience} epochs")
                break
        else:
            print(msg + " " * 20)  # pad to clear leftover per-batch progress chars

    if best_state is not None:
        p2g.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"[p2g] restored best epoch {best_epoch} (val_WER {best_wer:.3f})")

    if output_dir:
        p2g.save(output_dir)
        print(f"[p2g] saved fine-tuned model to {output_dir}")

    return p2g, metrics
