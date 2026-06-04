from pathlib import Path

import numpy as np
import torch
from torch import nn
import torchaudio.functional as AF
import torchaudio.transforms as T
from scipy.io import wavfile

from .config import CTCConfig as C
from .augmentation import apply_noise_batch, apply_gain_batch


def audio_to_logmel(
    audio: np.ndarray | torch.Tensor,
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

    Accepts either a NumPy array or a torch.Tensor (e.g. an already-augmented
    waveform). The result stays on the input tensor's device.
    """

    if isinstance(audio, np.ndarray):
        wav = torch.from_numpy(audio).float()
    else:
        wav = audio.float()

    if wav.ndim == 2:
        wav = wav.mean(dim=1)

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


class CTCFeatureExtractor(nn.Module):
    """
    On-device, batched log-mel feature extraction for CTC training.

    Takes a padded batch of raw waveforms and produces standardized log-mel spectrograms entirely on the module's device.
    Optional, length-preserving augmentation (additive noise + gain on the waveform, then SpecAugment on the mel)

    Length-changing augmentation (speed perturbation) is intentionally NOT done here; it must happen per-file on CPU before batching (see CTCDataset).
    """

    def __init__(
        self,
        sample_rate: int = C.SAMPLE_RATE,
        n_fft: int = C.N_FFT,
        hop_length: int = C.HOP_LENGTH,
        n_mels: int = C.N_MELS,
        standardize: bool = True,
        spec_augment: nn.Module | None = None,
        noise_prob: float = 0.5,
        gain_prob: float = 0.5,
        noise_level: tuple[float, float] = (15.0, 30.0),
        gain_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()

        self.hop_length = hop_length
        self.standardize = standardize
        self.spec_augment = spec_augment
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.noise_level = noise_level
        self.gain_range = gain_range

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.db_transform = T.AmplitudeToDB(stype="power")

    def mel_lengths(self, wav_lengths: torch.Tensor) -> torch.Tensor:
        """Valid mel-frame count per sample (center=True): 1 + n_samples // hop."""
        return 1 + wav_lengths // self.hop_length

    def _masked_standardize(
        self, mel: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-sample standardization over valid frames only.

        mel: (B, F, T). Mean/std are computed across the time axis using each
        sample's valid length so zero-padded frames don't skew the statistics.
        """

        b, f, t = mel.shape
        frame_idx = torch.arange(t, device=mel.device).view(1, 1, t)
        mask = (frame_idx < lengths.view(b, 1, 1)).expand(b, f, t)  # (B, F, T)

        counts = mask[:, :1, :].sum(dim=2, keepdim=True).clamp_min(1.0)  # (B, 1, 1)
        zero = torch.zeros((), device=mel.device, dtype=mel.dtype)
        masked = torch.where(mask, mel, zero)
        mean = masked.sum(dim=2, keepdim=True) / counts
        diff = torch.where(mask, mel - mean, zero)
        var = (diff**2).sum(dim=2, keepdim=True) / counts
        std = var.sqrt()
        return (mel - mean) / (std + 1e-8)

    def forward(
        self,
        waveforms: torch.Tensor,
        wav_lengths: torch.Tensor,
        augment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        waveforms: (B, T_samples) padded, on this module's device.
        wav_lengths: (B,) valid sample counts.

        Returns (mel (B, F, T'), mel_lengths (B,)).
        """

        # Feature extraction must run in fp32. Under AMP autocast the power
        # spectrogram and AmplitudeToDB's amin clamp (1e-10) underflow to 0 in
        # fp16, giving log10(0) = -inf and then NaN. Forcing autocast off here
        # keeps the STFT/log path numerically safe regardless of the caller.
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            waveforms = waveforms.float()

            if augment:
                waveforms = apply_noise_batch(
                    waveforms, noise_level=self.noise_level, prob=self.noise_prob
                )
                waveforms = apply_gain_batch(
                    waveforms, gain_range=self.gain_range, prob=self.gain_prob
                )

            mel = self.db_transform(self.mel_transform(waveforms))  # (B, F, T')
            lengths = self.mel_lengths(wav_lengths).clamp_max(mel.shape[-1])

            if self.standardize:
                mel = self._masked_standardize(mel, lengths)

            if augment and self.spec_augment is not None:
                mel = self.spec_augment(mel)

        return mel, lengths
