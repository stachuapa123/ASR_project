from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T
from scipy.io import wavfile

from .config import CTCConfig as C


def audio_to_logmel(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int = C.SAMPLE_RATE,
    n_fft: int = C.N_FFT,
    hop_length: int = C.HOP_LENGTH,
    n_mels: int = C.N_MELS,
    standardize: bool = True,
    mel_transform: T.MelSpectrogram | None = None,
    db_transform: T.AmplitudeToDB | None = None,
) -> torch.Tensor:
    """
    Convert raw mono audio to log-mel spectrogram.
    """

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    wav = torch.from_numpy(audio).float()

    if sample_rate != target_sample_rate:
        wav = AF.resample(wav, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    wav = wav.unsqueeze(0)  # (1, T)
    if mel_transform is None:
        mel_transform = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
    if db_transform is None:
        db_transform = T.AmplitudeToDB(stype="power")

    mel = db_transform(mel_transform(wav)).squeeze(0)  # (F, T')

    if standardize:
        mel = (mel - mel.mean(dim=1, keepdim=True)) / (
            mel.std(dim=1, keepdim=True) + 1e-8
        )

    return mel


def wav_path_to_logmel(
    path: str | Path,
    target_sample_rate: int = C.SAMPLE_RATE,
    n_fft: int = C.N_FFT,
    hop_length: int = C.HOP_LENGTH,
    n_mels: int = C.N_MELS,
    standardize: bool = True,
    mel_transform: T.MelSpectrogram | None = None,
    db_transform: T.AmplitudeToDB | None = None,
) -> torch.Tensor:
    """
    Read a wav file and convert it to log-mel.
    """

    path = str(path)
    sample_rate, audio = wavfile.read(path)

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    return audio_to_logmel(
        audio,
        sample_rate,
        target_sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        standardize=standardize,
        mel_transform=mel_transform,
        db_transform=db_transform,
    )
