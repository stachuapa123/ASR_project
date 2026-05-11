import os
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch
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

    Each item:
        mel: (F, T) log-mel spectrogram,
        target: (U,) tensor of phone indices.
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

        self.samples: list[tuple[str, str]] = []    # List of (wav_path, textgrid_path) pairs
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
            audio_aug = augment_waveform(
                audio,
                sr=self.sample_rate,
                noise_prob=self.noise_prob,
                gain_prob=self.gain_prob,
                speed_perturbation_prob=self.speed_perturbation_prob,
                noise_level=self.noise_level,
                gain_range=self.gain_range,
                speed_perturbation_range=self.speed_perturbation_range,
            )
            mel_aug = audio_to_logmel(
                audio_aug,
                self.sample_rate,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_mode:
            return self.cached_data[idx]

        items = self._process_file(idx, return_multiple=False)  # TODO: support on-the-fly augmentation when not caching
        return items[0]


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
