"""
End-to-end wav -> text pipeline for the P2G stage.

CTC inference (model loading + wav -> phone labels) lives in ``src.ctc.inference``;
here we only orchestrate it together with a P2GModel.
"""

import torch

from src.ctc.inference import wav_to_phone_labels
from src.ctc.model import CTCModel


def transcribe_wav(
    wav_path: str, ctc_model: CTCModel, p2g, device: torch.device, num_beams: int = 4
) -> str:
    """Full pipeline: wav -> CTC phones -> seq2seq text. ``p2g`` is a P2GModel."""
    labels = wav_to_phone_labels(wav_path, ctc_model, device)
    return p2g.transcribe(labels, num_beams=num_beams)
