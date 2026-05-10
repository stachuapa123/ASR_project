import torch
from collections.abc import Mapping

from .config import CTCConfig as C


def greedy_decode(logits: torch.Tensor, blank: int = C.BLANK_IDX) -> list[list[int]]:
    """
    Greedy CTC decoding: argmax per time step, collapse repeats, drop blanks.

    Args:
        logits: (B, T, num_classes) tensor.
        blank: index of the CTC blank symbol.

    Returns:
        List of length B with decoded index sequences.
    """

    pred_indices = torch.argmax(logits, dim=-1)  # (B, T)

    decoded_batch: list[list[int]] = []

    for i in range(pred_indices.size(0)):
        seq: list[int] = []
        prev = -1
        for t in range(pred_indices.size(1)):
            idx = int(pred_indices[i, t].item())
            if idx != blank and idx != prev:
                seq.append(idx)
            prev = idx
        decoded_batch.append(seq)

    return decoded_batch


def decode_to_phonemes(
    indices: list[int],
    idx2label: Mapping[int, str] | None = None,
    sep: str = " ",
) -> str:
    """
    Convert a sequence of indices to a phoneme string.
    """

    mapping = C.IDX2LABEL if idx2label is None else idx2label
    return sep.join(mapping.get(idx, "<UNK>") for idx in indices)


def compute_per(preds: list[list[int]], targets: list[list[int]]) -> float:
    """
    Phoneme Error Rate (PER) = total edit distance / total target phonemes.
    """

    total_distance = 0
    total_length = 0

    for pred, target in zip(preds, targets):
        total_distance += _levenshtein_distance(pred, target)
        total_length += len(target)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def _levenshtein_distance(seq1: list[int], seq2: list[int]) -> int:
    """
    Standard dynamic-programming edit distance for integer sequences.
    """

    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n]
