import torch
from collections.abc import Mapping

from .config import CTCConfig as C
from src.utils.edit_distance import corpus_error_rate


def greedy_decode(
    logits: torch.Tensor,
    blank: int = C.BLANK_IDX,
    lengths: torch.Tensor | None = None,
) -> list[list[int]]:
    """Greedy CTC decode: argmax, collapse repeats, drop blanks.

    ``lengths`` (valid frame counts) limits decoding to non-padded frames.
    """

    pred_indices = torch.argmax(logits, dim=-1)  # (B, T)
    total_t = pred_indices.size(1)

    decoded_batch: list[list[int]] = []

    for i in range(pred_indices.size(0)):
        t_end = total_t if lengths is None else min(int(lengths[i]), total_t)
        seq: list[int] = []
        prev = -1
        for t in range(t_end):
            idx = int(pred_indices[i, t].item())
            if idx != blank and idx != prev:
                seq.append(idx)
            prev = idx
        decoded_batch.append(seq)

    return decoded_batch


def decode_to_phones(
    indices: list[int],
    idx2label: Mapping[int, str] | None = None,
    sep: str = " ",
) -> str:
    """Indices -> phone string."""

    mapping = C.IDX2LABEL if idx2label is None else idx2label
    return sep.join(mapping.get(idx, "<UNK>") for idx in indices)


def compute_per(preds: list[list[int]], targets: list[list[int]]) -> float:
    """Phone Error Rate = total edit distance / total target phones."""

    return corpus_error_rate(preds, targets)
