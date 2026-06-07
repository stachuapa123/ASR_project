"""WER/CER wrappers over ``src.utils.edit_distance.corpus_error_rate`` (torch-free)."""

from src.utils.edit_distance import corpus_error_rate


def corpus_wer(preds: list[str], refs: list[str]) -> float:
    """Aggregate WER = total word edits / total reference words."""
    return corpus_error_rate((p.split() for p in preds), (r.split() for r in refs))


def corpus_cer(preds: list[str], refs: list[str]) -> float:
    """Aggregate CER = total char edits / total reference characters."""
    return corpus_error_rate((list(p) for p in preds), (list(r) for r in refs))
