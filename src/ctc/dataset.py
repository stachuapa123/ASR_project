import os
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import tqdm

from .config import CTCConfig as C
from .textgrid import parse_phoneme_intervals
from .features import audio_to_logmel
from .augmentation import augment_waveform


def textgrid_to_phoneme_ids(
    textgrid_path: str | Path,
    map_sp_to_sil: bool = True,
) -> list[int]:
    """
    Parse a TextGrid file and map phones to integer indices.
    """
    textgrid_path = Path(textgrid_path)
    with textgrid_path.open("r", encoding="utf-8") as f:
        intervals = parse_phoneme_intervals(f.read(), map_sp_to_sil=map_sp_to_sil)

    indices: list[int] = []
    for _, _, label in intervals:
        if not label or label not in C.LABEL2IDX:
            continue
        indices.append(C.LABEL2IDX[label])

    return indices


class CTCDataset(Dataset):
    """
    Dataset for CTC training.

    Each item:
        mel: (F, T) log-mel spectrogram
        target: (U,) tensor of phoneme indices.
    """

    def __init__(
        self,
        data_root: str | Path,
        cache_mode: bool = True,
        apply_augmentations: bool = False,
        max_files: int | None = None,
        noiseprob: float = 0.5,
        gainprob: float = 0.5,
        tempo_prob: float = 0.2,
        noise_level: tuple[float, float] = (10.0, 30.0),
        gain_range: tuple[float, float] = (-5.0, 5.0),
        tempo_range: tuple[float, float] = (0.95, 1.05),
    ) -> None:
        super().__init__()

        self.apply_augmentations = apply_augmentations
        self.cache_mode = cache_mode

        self.noiseprob = noiseprob
        self.gainprob = gainprob
        self.tempo_prob = tempo_prob
        self.noise_level = noise_level
        self.gain_range = gain_range
        self.tempo_range = tempo_range

        data_root = Path(data_root)
        tg_paths = sorted(str(p) for p in data_root.rglob("*.TextGrid"))
        if max_files is not None:
            tg_paths = tg_paths[:max_files]

        self.samples: list[tuple[str, str]] = []
        for tg_path in tg_paths:
            wav_path = tg_path[: -len(".TextGrid")] + ".wav"
            if os.path.exists(wav_path):
                self.samples.append((wav_path, tg_path))

        self.cached_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.cache_mode:
            print(f"Pre-computing and caching {len(self.samples)} files...")
            for idx in tqdm.tqdm(range(len(self.samples)), desc="CTC cache"):
                self.cached_data.extend(
                    self._process_file(idx, return_multiple=self.apply_augmentations)
                )
            print(f"Cache complete. Total items in RAM: {len(self.cached_data)}")

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

        target_ids = textgrid_to_phoneme_ids(tg_path, map_sp_to_sil=True)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        sample_rate, audio = self._load_wav(wav_path)

        items: list[tuple[torch.Tensor, torch.Tensor]] = []

        # clean
        mel_clean = audio_to_logmel(audio, sample_rate, standardize=True)
        items.append((mel_clean, target_tensor))

        # augmented
        if return_multiple:
            aug_audio = augment_waveform(
                audio,
                sr=C.SAMPLE_RATE,
                noiseprob=self.noiseprob,
                gainprob=self.gainprob,
                tempo_prob=self.tempo_prob,
                noise_level=self.noise_level,
                gain_range=self.gain_range,
                tempo_range=self.tempo_range,
            )
            mel_aug = audio_to_logmel(aug_audio, C.SAMPLE_RATE, standardize=True)
            items.append((mel_aug, target_tensor))

        return items

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_mode:
            return self.cached_data[idx]

        items = self._process_file(idx, return_multiple=False)
        return items[0]


def ctc_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
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
        padding_value=C.N_CLASSES,  # CTC blank index
    )

    return mels_padded, targets_padded, input_lengths, target_lengths
