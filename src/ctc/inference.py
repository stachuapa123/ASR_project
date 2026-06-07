"""CTC inference: checkpoint loading and wav -> phone labels (used by P2G too)."""

import torch

from src.utils.checkpoint import load_weights

from .config import CTCConfig as C
from .features import wav_path_to_logmel
from .metrics import greedy_decode
from .model import CTCModel


def load_ctc_model(checkpoint: str, device: torch.device) -> CTCModel:
    """Instantiate the CTC acoustic model and load a checkpoint's weights."""
    model = CTCModel().to(device)
    load_weights(checkpoint, model, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def wav_to_phone_labels(
    wav_path: str, ctc_model: CTCModel, device: torch.device
) -> list[str]:
    """Run the CTC model over a wav and greedy-decode it to phone labels (sil kept)."""
    mel = wav_path_to_logmel(wav_path)  # (F, T) fp32 on CPU
    logits = ctc_model(mel.unsqueeze(0).to(device))  # (1, T', C)
    lengths = torch.tensor([logits.shape[1]])
    ids = greedy_decode(logits.cpu(), lengths=lengths)[0]
    return [C.IDX2LABEL.get(i, "") for i in ids]
