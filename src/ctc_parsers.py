import os
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

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
        augment: bool = False,
        max_files: int | None = None,
        noiseprob: float = 0.4,
        gainprob: float = 0.4,
        tempo_prob: float = 0.2,
        noise_level: tuple[float, float] = (10, 30),
        gain_range: tuple[float, float] = (-5, 5),
        tempo_range: tuple[float, float] = (0.95, 1.05),
    ) -> None:
        self.augment = augment
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav_path, tg_path = self.samples[idx]

        samplerate, audio = wavfile.read(wav_path)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)

        if self.augment:
            audio = augment_audio(
                audio,
                sr=C.SAMPLE_RATE,
                noiseprob=self.noiseprob,
                gainprob=self.gainprob,
                tempo_prob=self.tempo_prob,
                noise_level=self.noise_level,
                gain_range=self.gain_range,
                tempo_range=self.tempo_range,
            )

        wav_t = torch.from_numpy(audio).unsqueeze(0)
        mel = C.decibel_transformer(C.mel_transformer(wav_t)).squeeze(0)

        mel = (mel - mel.mean(dim=1, keepdim=True)) / (
            mel.std(dim=1, keepdim=True) + 1e-8
        )

        target = get_phoneme_sequence(tg_path, silences_same=True)

        return mel, torch.tensor(target, dtype=torch.long)


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
