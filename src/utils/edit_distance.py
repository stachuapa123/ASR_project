"""
Edit distance and corpus-level error rate over token sequences.

Pure-Python (no torch) so torch-free metric modules can depend on it. The same
``corpus_error_rate`` backs PER (phone-id lists), WER (word-token lists) and CER
(character lists) — error rate is just total edits / total reference length.
"""

from collections.abc import Iterable, Sequence


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (rolling-row DP)."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def corpus_error_rate(preds: Iterable[Sequence], refs: Iterable[Sequence]) -> float:
    """Aggregate error rate = total edit distance / total reference length.

    ``preds``/``refs`` are iterables of token sequences (e.g. phone-id lists,
    word lists, character lists). Returns 0.0 when the references are empty.
    """
    dist = 0
    total = 0
    for pred, ref in zip(preds, refs):
        dist += edit_distance(pred, ref)
        total += len(ref)
    return dist / total if total else 0.0
