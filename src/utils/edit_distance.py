"""Edit distance + corpus error rate over token sequences.

``corpus_error_rate`` backs PER, WER and CER (total edits / total reference length).
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
    """Total edit distance / total reference length over iterables of token
    sequences (phone-id/word/char lists). Returns 0.0 when references are empty.
    """
    dist = 0
    total = 0
    for pred, ref in zip(preds, refs):
        dist += edit_distance(pred, ref)
        total += len(ref)
    return dist / total if total else 0.0
