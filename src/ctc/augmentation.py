import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from .config import CTCConfig as C


def augment_waveform(
    audio: np.ndarray | torch.Tensor,
    sr: int = C.SAMPLE_RATE,
    noiseprob: float = 0.3,
    gainprob: float = 0.3,
    tempo_prob: float = 0.0,
    noise_level: tuple[float, float] = (15.0, 30.0),
    gain_range: tuple[float, float] = (-3.0, 3.0),
    tempo_range: tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """
    Waveform-level augmentation: noise, gain, optional tempo.

    Operates on mono audio and returns numpy float32 in [-1, 1].
    """

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    else:
        audio = audio.float()

    if audio.ndim > 1:
        audio = audio.mean(dim=-1)

    # Noise
    if torch.rand(1).item() < noiseprob:
        snr_db = (
            noise_level[0] + (noise_level[1] - noise_level[0]) * torch.rand(1).item()
        )
        signal_power = audio.pow(2).mean()
        if signal_power > 1e-10:
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(audio) * noise_power.sqrt()
            audio = audio + noise

    # Gain
    if torch.rand(1).item() < gainprob:
        gain_db = gain_range[0] + (gain_range[1] - gain_range[0]) * torch.rand(1).item()
        audio = audio * (10.0 ** (gain_db / 20.0))

    # Tempo (off by default – this can misalign labels)
    if torch.rand(1).item() < tempo_prob:
        rate = tempo_range[0] + (tempo_range[1] - tempo_range[0]) * torch.rand(1).item()
        audio = torchaudio.functional.speed(audio.unsqueeze(0), sr, rate)[0]

    audio = audio.clamp(-1.0, 1.0)
    return audio.numpy()


class SpecAugment:
    """
    Basic SpecAugment for mel spectrograms.

    Supports both (F, T) and (B, F, T) inputs.
    """

    def __init__(
        self,
        freq_mask_percent: float = 0.1,
        time_mask_percent: float = 0.125,
        p: float = 0.3,
        n_mels: int = C.N_MELS,
        time_frames: int = C.WIN_FRAMES,
    ) -> None:
        self.freq_mask = T.FrequencyMasking(int(freq_mask_percent * n_mels))
        self.time_mask = T.TimeMasking(int(time_mask_percent * time_frames))
        self.p = p

    def _augment_single(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a single (F, T) tensor.
        """
        if torch.rand(1, device=mel.device).item() < self.p:
            mel = self.freq_mask(mel)
        if torch.rand(1, device=mel.device).item() < self.p:
            mel = self.time_mask(mel)
        return mel

    def __call__(self, mels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mels: (F, T) or (B, F, T)

        Returns:
            Augmented tensor with the same shape.
        """
        if mels.ndim == 2:
            return self._augment_single(mels)

        if mels.ndim == 3:
            out = []
            for mel in mels:
                out.append(self._augment_single(mel))
            return torch.stack(out, dim=0)

        raise ValueError(f"Expected (F, T) or (B, F, T), got shape {tuple(mels.shape)}")
