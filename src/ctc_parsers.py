import os
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import tqdm

from .constants import Constants as C
from .parsers import parse_phonemes
from .augment import augment_audio


def get_phoneme_sequence(
    text_grid_path: str | Path, silences_same: bool = True
) -> list[int]:
    with open(text_grid_path, "r", encoding="utf-8") as f:
        intervals = parse_phonemes(f.read(), silences_same=silences_same)

    sequence = []
    for _, _, ptext in intervals:
        if not ptext or ptext not in C.LABEL2IDX:
            continue
        sequence.append(C.LABEL2IDX[ptext])
    return sequence


class CTCDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        cache_mode: bool = True,
        apply_augmentations: bool = False,
        max_files: int | None = None,
        noiseprob: float = 0.5,
        gainprob: float = 0.5,
        tempo_prob: float = 0.2,
        noise_level: tuple[float, float] = (10, 30),
        gain_range: tuple[float, float] = (-5, 5),
        tempo_range: tuple[float, float] = (0.95, 1.05),
    ) -> None:
        self.apply_augmentations = apply_augmentations
        self.cache_mode = cache_mode
        self.noiseprob = noiseprob
        self.gainprob = gainprob
        self.tempo_prob = tempo_prob
        self.noise_level = noise_level
        self.gain_range = gain_range
        self.tempo_range = tempo_range

        tg_paths = sorted(str(p) for p in Path(data_dir).rglob("*.TextGrid"))
        if max_files is not None:
            tg_paths = tg_paths[:max_files]

        self.samples: list[tuple[str, str]] = []
        for tg in tg_paths:
            wav_path = tg[: -len(".TextGrid")] + ".wav"
            if os.path.exists(wav_path):
                self.samples.append((wav_path, tg))

        self.cached_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.cache_mode:
            print(f"Pre-computing and caching {len(self.samples)} files into RAM...")
            for idx in tqdm.tqdm(range(len(self.samples)), desc="Building Cache"):
                self.cached_data.extend(
                    self._process_file(idx, return_multiple=self.apply_augmentations)
                )
            print(f"Cache complete. Total items in RAM: {len(self.cached_data)}")

    def __len__(self) -> int:
        if self.cache_mode:
            return len(self.cached_data)
        return len(self.samples)

    def _process_file(
        self, idx: int, return_multiple: bool = False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        wav_path, tg_path = self.samples[idx]
        target = get_phoneme_sequence(tg_path, silences_same=True)
        target_tensor = torch.tensor(target, dtype=torch.long)

        samplerate, audio = wavfile.read(wav_path)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)

        items = []

        # 1. Base clean audio
        items.append((self._audio_to_mel(audio), target_tensor))

        # 2. Add augmented copies if requested
        if return_multiple:
            aug_audio = augment_audio(
                audio,
                sr=C.SAMPLE_RATE,
                noiseprob=self.noiseprob,
                gainprob=self.gainprob,
                tempo_prob=self.tempo_prob,
                noise_level=self.noise_level,
                gain_range=self.gain_range,
                tempo_range=self.tempo_range,
            )
            items.append((self._audio_to_mel(aug_audio), target_tensor))

        return items

    def _audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        wav_t = torch.from_numpy(audio).unsqueeze(0)
        mel = C.decibel_transformer(C.mel_transformer(wav_t)).squeeze(0)
        mel = (mel - mel.mean(dim=1, keepdim=True)) / (
            mel.std(dim=1, keepdim=True) + 1e-8
        )
        return mel

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_mode:
            return self.cached_data[idx]

        # Fallback for dynamic/no-cache mode
        items = self._process_file(idx, return_multiple=False)
        return items[0]


def ctc_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mels, targets = zip(*batch)

    input_lengths = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    mels_transposed = [mel.transpose(0, 1) for mel in mels]
    padded_mels_transposed = pad_sequence(
        mels_transposed, batch_first=True, padding_value=0.0
    )
    padded_mels = padded_mels_transposed.transpose(1, 2)

    padded_targets = pad_sequence(targets, batch_first=True, padding_value=C.N_CLASSES)

    return padded_mels, padded_targets, input_lengths, target_lengths
