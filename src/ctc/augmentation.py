import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from .config import CTCConfig as C


def augment_waveform(
    audio: np.ndarray | torch.Tensor,
    sample_rate: int = C.SAMPLE_RATE,
    noise_prob: float = 0.5,
    gain_prob: float = 0.5,
    speed_perturbation_prob: float = 0.3,
    noise_level: tuple[float, float] = (15.0, 30.0),
    gain_range: tuple[float, float] = (-3.0, 3.0),
    speed_perturbation_range: tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    """
    Waveform-level augmentation: noise, gain, speed perturbation.

    Accepts a NumPy array or a torch.Tensor and always returns a torch.Tensor.
    """

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    audio = audio.float()
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)

    # Noise
    if torch.rand(1).item() < noise_prob:
        audio = _apply_noise(audio, noise_level)

    # Gain
    if torch.rand(1).item() < gain_prob:
        audio = _apply_gain(audio, gain_range)

    # Speed perturbation
    if torch.rand(1).item() < speed_perturbation_prob:
        audio = _apply_speed_perturbation(audio, sample_rate, speed_perturbation_range)

    return audio


def _apply_noise(audio: torch.Tensor, noise_level: tuple[float, float]) -> torch.Tensor:
    snr_db = noise_level[0] + (noise_level[1] - noise_level[0]) * torch.rand(1).item()
    signal_power = audio.pow(2).mean()
    if signal_power > 1e-10:
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(audio) * noise_power.sqrt()
        return audio + noise
    return audio


def _apply_gain(audio: torch.Tensor, gain_range: tuple[float, float]) -> torch.Tensor:
    gain_db = gain_range[0] + (gain_range[1] - gain_range[0]) * torch.rand(1).item()
    return audio * (10.0 ** (gain_db / 20.0))


def _apply_speed_perturbation(
    audio: torch.Tensor, sr: int, perturbation_range: tuple[float, float]
) -> torch.Tensor:
    rate = (
        perturbation_range[0]
        + (perturbation_range[1] - perturbation_range[0]) * torch.rand(1).item()
    )
    perturbed, _ = torchaudio.functional.speed(audio.unsqueeze(0), sr, rate)
    return perturbed.squeeze(0)


def apply_noise_batch(
    waveforms: torch.Tensor,
    noise_level: tuple[float, float] = (15.0, 30.0),
    prob: float = 0.5,
) -> torch.Tensor:
    """
    Batched, on-device additive Gaussian noise at a per-sample random SNR.

    waveforms: (B, T). Each sample is independently noised with probability
    `prob`; the SNR is drawn per sample from `noise_level` (dB). Vectorized so
    it runs entirely on the input tensor's device.
    """

    b = waveforms.shape[0]
    device = waveforms.device

    snr_db = noise_level[0] + (noise_level[1] - noise_level[0]) * torch.rand(
        b, device=device
    )
    signal_power = waveforms.pow(2).mean(dim=1)  # (B,)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(waveforms) * noise_power.sqrt().unsqueeze(1)

    apply = (torch.rand(b, device=device) < prob).float().unsqueeze(1)
    # Skip samples with negligible energy to avoid amplifying silence
    apply = apply * (signal_power > 1e-10).float().unsqueeze(1)
    return waveforms + apply * noise


def apply_gain_batch(
    waveforms: torch.Tensor,
    gain_range: tuple[float, float] = (-3.0, 3.0),
    prob: float = 0.5,
) -> torch.Tensor:
    """
    Batched, on-device gain at a per-sample random level (dB).

    waveforms: (B, T). Each sample is independently scaled with probability
    `prob` by a gain drawn from `gain_range`.
    """

    b = waveforms.shape[0]
    device = waveforms.device

    gain_db = gain_range[0] + (gain_range[1] - gain_range[0]) * torch.rand(
        b, device=device
    )
    gain = 10.0 ** (gain_db / 20.0)
    apply = (torch.rand(b, device=device) < prob).float()
    # Where not applied, gain factor is 1.0
    factor = torch.where(apply.bool(), gain, torch.ones_like(gain))
    return waveforms * factor.unsqueeze(1)


class SpecAugment:
    def __init__(
        self,
        n_mels: int = C.N_MELS,
    ) -> None:
        self.freq_mask = T.FrequencyMasking(freq_mask_param=int(0.2 * n_mels))
        self.time_mask = T.TimeMasking(time_mask_param=25, p=0.2)

    def _augment_single(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a single (F, T) tensor.
        """

        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        return mel

    def __call__(self, mels: torch.Tensor) -> torch.Tensor:
        if mels.ndim == 2:
            return self._augment_single(mels)

        if mels.ndim == 3:
            out = []
            for mel in mels:
                out.append(self._augment_single(mel))
            return torch.stack(out, dim=0)

        raise ValueError(f"Expected (F, T) or (B, F, T), got shape {tuple(mels.shape)}")
