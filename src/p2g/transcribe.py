"""End-to-end wav -> text for P2G: CTC phones (``src.ctc.inference``) + P2GModel.

Long recordings are chunked on silence into source-sized pieces (else they overflow
``MAX_SOURCE_LEN``), each transcribed and joined.
"""

import torch

from src.ctc.inference import wav_to_phone_labels
from src.ctc.model import CTCModel
from .config import P2GConfig as P
from .data import _SILENCE


def chunk_phone_labels(
    labels: list[str], max_len: int, silences: set[str] = _SILENCE
) -> list[list[str]]:
    """Split labels into <=``max_len`` chunks, breaking at silences where possible
    (a silence-free run longer than ``max_len`` is hard-split)."""
    if len(labels) <= max_len:
        return [labels]

    # segment at silences (silence token ends its segment)
    segments: list[list[str]] = []
    cur: list[str] = []
    for lbl in labels:
        cur.append(lbl)
        if lbl in silences:
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)

    chunks: list[list[str]] = []
    cur = []
    for seg in segments:
        if cur and len(cur) + len(seg) > max_len:
            chunks.append(cur)
            cur = []
        if len(seg) > max_len:  # silence-free run longer than the window
            for i in range(0, len(seg), max_len):
                chunks.append(seg[i : i + max_len])
            cur = []
            continue
        cur.extend(seg)
    if cur:
        chunks.append(cur)
    return chunks


def transcribe_wav(
    wav_path: str,
    ctc_model: CTCModel,
    p2g,
    device: torch.device,
    num_beams: int = 4,
    max_source_margin: int = 16,
) -> str:
    """Full pipeline: wav -> CTC phones -> chunked seq2seq text (``p2g`` = P2GModel).

    ``max_source_margin`` reserves tokens for the prefix/EOS vs ``p2g.max_source_len``.
    """
    labels = wav_to_phone_labels(wav_path, ctc_model, device)
    max_len = max(1, getattr(p2g, "max_source_len", P.MAX_SOURCE_LEN) - max_source_margin)
    chunks = chunk_phone_labels(labels, max_len)
    texts = [p2g.transcribe(chunk, num_beams=num_beams) for chunk in chunks]
    return " ".join(t for t in texts if t.strip())
