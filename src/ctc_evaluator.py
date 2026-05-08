import torch
from pathlib import Path

from .constants import Constants as C


def greedy_decode(logits: torch.Tensor, blank: int = C.N_CLASSES) -> list[list[int]]:
    """
    Decodes a batch of logits using greedy decoding (argmax).

    Args:
        logits (torch.Tensor): Tensor of shape (B, T, num_classes)
        blank (int): The index of the CTC blank token.

    Returns:
        list[list[int]]: A list of decoded sequences (one list of integers per batch item).
    """
    # Get the most likely class at each timestep
    # pred_indices shape: (B, T)
    pred_indices = torch.argmax(logits, dim=-1)

    decoded_batch: list[list[int]] = []

    for i in range(pred_indices.shape[0]):
        decoded_seq: list[int] = []
        prev_idx = -1
        for step in range(pred_indices.shape[1]):
            idx: int = int(pred_indices[i, step].item())
            # Collapse repeating tokens and ignore the blank token
            if idx != blank and idx != prev_idx:
                decoded_seq.append(idx)
            prev_idx = idx

        decoded_batch.append(decoded_seq)

    return decoded_batch


def decode_to_phonemes(indices: list[int], sep: str = " ") -> str:
    """
    Converts a sequence of phoneme indices into a combined string.
    """
    return sep.join(C.IDX2LABEL.get(idx, "<UNK>") for idx in indices)


def compute_per(preds: list[list[int]], targets: list[list[int]]) -> float:
    """
    Computes Phoneme Error Rate (PER) using Levenshtein distance natively.
    """
    total_distance = 0
    total_len = 0

    for p, t in zip(preds, targets):
        distance = _levenshtein_distance(p, t)
        total_distance += distance
        total_len += len(t)

    if total_len == 0:
        return 0.0

    return total_distance / total_len


def _levenshtein_distance(seq1: list[int], seq2: list[int]) -> int:
    """
    Computes the edit distance between two sequences of integers.
    """
    m, n = len(seq1), len(seq2)

    # Initialize DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[m][n]
