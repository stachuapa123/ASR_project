import os
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T
from scipy.io import wavfile
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import tqdm

from .config import CTCConfig as C
from .features import audio_to_logmel
from .augmentation import augment_waveform
from .textgrid import textgrid_to_phone_ids


class CTCDataset(Dataset):
    """
    Dataset for CTC training.

    Two output modes:
      * mel mode (default): each item is (mel (F, T), target (U,)). Log-mel
        features are computed on CPU in the DataLoader workers.
      * waveform mode (``return_waveform=True``): each item is
        (waveform (T_samples,), target (U,)). Feature extraction is deferred to
        a CTCFeatureExtractor running batched on the GPU; only length-changing
        augmentation (speed perturbation) is applied here.
    """

    def __init__(
        self,
        data_root: str | Path,
        cache_mode: bool = False,
        apply_augmentations: bool = False,
        max_files: int | None = None,
        sample_rate: int = C.SAMPLE_RATE,
        n_fft: int = C.N_FFT,
        hop_length: int = C.HOP_LENGTH,
        n_mels: int = C.N_MELS,
        standardize: bool = True,
        return_waveform: bool = False,
        label2idx: Mapping[str, int] = C.LABEL2IDX,
        map_sp_to_sil: bool = True,
        noise_prob: float = 0.5,
        gain_prob: float = 0.5,
        speed_perturbation_prob: float = 0.3,
        noise_level: tuple[float, float] = (15.0, 30.0),
        gain_range: tuple[float, float] = (-3.0, 3.0),
        speed_perturbation_range: tuple[float, float] = (0.9, 1.1),
    ) -> None:
        super().__init__()

        self.return_waveform = return_waveform
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.standardize = standardize
        self.apply_augmentations = apply_augmentations
        self.cache_mode = cache_mode
        self.label2idx = label2idx
        self.map_sp_to_sil = map_sp_to_sil
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.speed_perturbation_prob = speed_perturbation_prob
        self.noise_level = noise_level
        self.gain_range = gain_range
        self.speed_perturbation_range = speed_perturbation_range

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=True,
            power=2.0,
        )
        self.db_transform = T.AmplitudeToDB(stype="power")

        # Discover all TextGrid files and pair each with its .wav
        data_root = Path(data_root)
        tg_paths = sorted(str(p) for p in data_root.rglob("*.TextGrid"))
        if max_files is not None:
            tg_paths = tg_paths[:max_files]

        self.samples: list[
            tuple[str, str]
        ] = []  # List of (wav_path, textgrid_path) pairs
        for tg_path in tg_paths:
            wav_path = tg_path[: -len(".TextGrid")] + ".wav"
            if os.path.exists(wav_path):
                self.samples.append((wav_path, tg_path))

        self.cached_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.cache_mode:
            # Pre-process every file into RAM once at startup
            print(f"Pre-computing and caching {len(self.samples)} files into RAM...")
            for idx in tqdm.tqdm(range(len(self.samples)), desc="CTC cache"):
                self.cached_data.extend(
                    self._process_file(idx, return_multiple=self.apply_augmentations)
                )
            total_items = len(self.cached_data)
            total_pairs = total_items // (2 if self.apply_augmentations else 1)
            aug_str = (
                f" (with {total_items - total_pairs} augmented variants)"
                if self.apply_augmentations
                else ""
            )
            print(f"Cache complete. Total items in RAM: {total_items}{aug_str}")

    def __len__(self) -> int:
        if self.cache_mode:
            return len(self.cached_data)
        return len(self.samples)

    def _load_wav(self, wav_path: str) -> tuple[int, np.ndarray]:
        sample_rate, audio = wavfile.read(wav_path)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
        return sample_rate, audio

    def _load_waveform(self, wav_path: str) -> torch.Tensor:
        """Load a wav as a mono float tensor resampled to ``self.sample_rate``."""
        sample_rate, audio = self._load_wav(wav_path)
        wav = torch.from_numpy(audio).float()
        if wav.ndim > 1:
            wav = wav.mean(dim=1)
        if sample_rate != self.sample_rate:
            wav = AF.resample(wav, sample_rate, self.sample_rate)
        return wav

    def _process_file(
        self,
        idx: int,
        return_multiple: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        wav_path, tg_path = self.samples[idx]

        # List of phone indices, length U
        target_ids = textgrid_to_phone_ids(
            tg_path, label2idx=self.label2idx, map_sp_to_sil=self.map_sp_to_sil
        )
        target_tensor = torch.tensor(target_ids, dtype=torch.long)  # (U,)

        if self.return_waveform:
            return self._process_file_waveform(wav_path, target_tensor, return_multiple)

        sample_rate, audio = self._load_wav(wav_path)

        items: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Clean
        mel_clean = audio_to_logmel(
            audio,
            sample_rate,
            target_sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            standardize=self.standardize,
            mel_transform=self.mel_transform,
            db_transform=self.db_transform,
        )
        items.append((mel_clean, target_tensor))

        # Augmented
        if return_multiple:
            # Augment at the file's native rate; audio_to_logmel resamples after
            audio_aug = augment_waveform(
                audio=audio,
                sample_rate=sample_rate,
                noise_prob=self.noise_prob,
                gain_prob=self.gain_prob,
                speed_perturbation_prob=self.speed_perturbation_prob,
                noise_level=self.noise_level,
                gain_range=self.gain_range,
                speed_perturbation_range=self.speed_perturbation_range,
            )
            mel_aug = audio_to_logmel(
                audio=audio_aug,
                sample_rate=sample_rate,
                target_sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                standardize=self.standardize,
                mel_transform=self.mel_transform,
                db_transform=self.db_transform,
            )
            items.append((mel_aug, target_tensor))

        return items

    def _process_file_waveform(
        self,
        wav_path: str,
        target_tensor: torch.Tensor,
        return_multiple: bool,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Waveform-mode processing: return raw (resampled) waveforms.

        Only speed perturbation (length-changing) is applied here; noise, gain
        and SpecAugment are deferred to the GPU feature extractor at train time.
        """
        wav = self._load_waveform(wav_path)

        items: list[tuple[torch.Tensor, torch.Tensor]] = [(wav, target_tensor)]

        if return_multiple:
            # Speed perturbation only; disable noise/gain (done on GPU)
            wav_aug = augment_waveform(
                audio=wav,
                sample_rate=self.sample_rate,
                noise_prob=0.0,
                gain_prob=0.0,
                speed_perturbation_prob=self.speed_perturbation_prob,
                speed_perturbation_range=self.speed_perturbation_range,
            )
            items.append((wav_aug, target_tensor))

        return items

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_mode:
            return self.cached_data[idx]

        items = self._process_file(idx, return_multiple=self.apply_augmentations)
        # With augmentation enabled, return the augmented variant (items[-1]); otherwise the single clean item
        return items[-1] if self.apply_augmentations else items[0]


def ctc_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_value: int = C.BLANK_IDX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for CTC training.

    Returns:
        mels: (B, F, T_max)
        targets: (B, U_max) padded with CTC blank index
        input_lengths: (B,) input lengths before conv pooling
        target_lengths: (B,) target sequence lengths
    """

    mels, targets = zip(*batch)

    input_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    mels_t = [mel.transpose(0, 1) for mel in mels]  # (T, F)
    mels_t_padded = pad_sequence(mels_t, batch_first=True, padding_value=0.0)
    mels_padded = mels_t_padded.transpose(1, 2)  # (B, F, T_max)

    targets_padded = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_value,
    )

    return mels_padded, targets_padded, input_lengths, target_lengths


def waveform_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_value: int = C.BLANK_IDX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for waveform-mode CTC training (GPU feature extraction).

    Returns:
        waveforms: (B, T_samples_max) zero-padded
        targets: (B, U_max) padded with CTC blank index
        wav_lengths: (B,) valid sample counts (before padding)
        target_lengths: (B,) target sequence lengths
    """

    waveforms, targets = zip(*batch)

    wav_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    waveforms_padded = pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )  # (B, T_samples_max)

    targets_padded = pad_sequence(
        targets,
        batch_first=True,
        padding_value=pad_value,
    )

    return waveforms_padded, targets_padded, wav_lengths, target_lengths
