from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from scipy.io import wavfile

from .config import CTCConfig as C


def audio_to_logmel(
    audio: np.ndarray,
    sample_rate: int,
    standardize: bool = True,
) -> torch.Tensor:
    """
    Convert raw mono audio to log-mel spectrogram.

    Args:
        audio: numpy array of shape (T,) or (T, channels).
        sample_rate: original sample rate.
        standardize: per-frequency standardization if True.

    Returns:
        mel: (n_mels, T') tensor on CPU.
    """
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio).float()

    if sample_rate != C.SAMPLE_RATE:
        wav = AF.resample(wav, sample_rate, C.SAMPLE_RATE)
        sample_rate = C.SAMPLE_RATE

    wav = wav.unsqueeze(0)  # (1, T)
    mel = C.db_transform(C.mel_transform(wav)).squeeze(0)  # (F, T')

    if standardize:
        mel = (mel - mel.mean(dim=1, keepdim=True)) / (
            mel.std(dim=1, keepdim=True) + 1e-8
        )

    return mel


def wav_path_to_logmel(path: str | Path, standardize: bool = True) -> torch.Tensor:
    """
    Convenience: read a wav file and convert it to log-mel.
    """
    path = str(path)
    sample_rate, audio = wavfile.read(path)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    return audio_to_logmel(audio, sample_rate, standardize=standardize)
