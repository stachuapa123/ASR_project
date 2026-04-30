from .constants import Constants as C
import numpy as np
from scipy.io import wavfile
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
def parse_phonemes(text_grid, silences_same=False):
    
    phonemes = []
    in_phones = False
    xmin = xmax = None
    for line in text_grid.split('\n'):
        line = line.strip()
        if 'name = "phones"' in line:
            in_phones = True; continue
        if in_phones and line.startswith('name =') and 'phones' not in line:
            break
        if not in_phones:
            continue
        if line.startswith('xmin =') and 'intervals' not in line:
            xmin = float(line.split('=')[1].strip())
        elif line.startswith('xmax =') and 'intervals' not in line:
            xmax = float(line.split('=')[1].strip())
        elif line.startswith('text ='):
            text = line.split('=', 1)[1].strip().strip('"')
            if text == 'sp' and silences_same:
                text = 'sil'
            if xmin is not None and xmax is not None:
                phonemes.append((xmin, xmax, text))
            xmin = xmax = None
    return phonemes

def wav_to_logmel(wav_path, standardize=True):
    samplerate, audio = wavfile.read(wav_path)
    assert samplerate == C.SAMPLE_RATE, f'expected {C.SAMPLE_RATE} Hz, got {samplerate}'
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max        #jeśli int, to normalizujemy do [-1, 1], 
        #np.iinfo(audio.dtype).max zwraca maksymalną wartość dla danego typu danych, np. 32767 dla int16. Dzieląc przez to, skalujemy wartości do zakresu [-1, 1].
    else:
        audio = audio.astype(np.float32)
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, n_samples)
    mel = C.decibel_transformer(C.mel_transformer(wav)).squeeze(0)         # (n_mels, T)

    if standardize:
        mel = (mel - mel.mean(dim = 1, keepdim=True)) / (mel.std(dim = 1, keepdim=True) + 1e-8)

    return mel

def windows_and_labels(mel, phonemes):
    n_mels, T = mel.shape
    frame_dur = C.HOP_LENGTH / C.SAMPLE_RATE  # seconds per frame
    out = []
    for start in range(0, T - C.WIN_FRAMES + 1, C.SHIFT_FRAMES):
        end = start + C.WIN_FRAMES
        t_start = start * frame_dur
        t_end   = end   * frame_dur
        best_vowel, best_overlap = None, 0.0
        for (pmin, pmax, ptext) in phonemes:
            if ptext not in C.PHONEMES:
                continue
            overlap = min(pmax, t_end) - max(pmin, t_start)
            if overlap > best_overlap:
                best_overlap, best_vowel = overlap, ptext
        label = best_vowel if best_vowel is not None else C.NON_PHONEME
        out.append((mel[:, start:end].clone(), C.LABEL2IDX[label]))
    return out

class PhonemeWindowDataset(Dataset):
    def __init__(self, data_dir, max_files=None, verbose=True, standardize=True, silences_same=False):
        # recursively find every .TextGrid at any depth under data_dir
        tg_paths = sorted(str(p) for p in Path(data_dir).rglob('*.TextGrid'))
        if verbose:
            print(f'found {len(tg_paths)} TextGrid files under {data_dir}')
        if max_files is not None:
            tg_paths = tg_paths[:max_files]
        xs, ys = [], []
        for i, tg in enumerate(tg_paths):
            wav = tg[:-len('.TextGrid')] + '.wav'
            if not os.path.exists(wav):
                continue
            try:
                mel = wav_to_logmel(wav, standardize=standardize)
                with open(tg, 'r', encoding='utf-8') as f:
                    phonemes = parse_phonemes(f.read(), silences_same=silences_same)
                for w, lbl in windows_and_labels(mel, phonemes):
                    xs.append(w); ys.append(lbl)
            except Exception as e:
                if verbose: print(f'skip {tg}: {e}')
            if verbose and (i + 1) % 100 == 0:
                print(f'  processed {i+1}/{len(tg_paths)} files, {len(xs)} windows')
        self.X = torch.stack(xs) if xs else torch.empty(0, C.N_MELS, C.WIN_FRAMES)
        self.y = torch.tensor(ys, dtype=torch.long)
        if verbose:
            print(f'total windows: {len(self.y)}')
            counts = torch.bincount(self.y, minlength=C.N_CLASSES).tolist()
            for l, c in zip(C.LABELS, counts):
                print(f'  {l:>9}: {c}')

    def __len__(self):            return len(self.y)
    def __getitem__(self, idx):   return self.X[idx], self.y[idx]