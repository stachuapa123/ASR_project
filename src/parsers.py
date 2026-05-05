from .constants import Constants as C
import numpy as np
from scipy.io import wavfile
import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
from scipy import signal
import warnings
from .augment import augment_audio, SpecAugment

def parse_phonemes(text_grid, silences_same=True):

    phonemes = []
    in_phones = False
    xmin = xmax = None
    for line in text_grid.split("\n"):
        line = line.strip()
        if 'name = "phones"' in line:
            in_phones = True
            continue
        if in_phones and line.startswith("name =") and "phones" not in line:
            break
        if not in_phones:
            continue
        if line.startswith("xmin =") and "intervals" not in line:
            xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax =") and "intervals" not in line:
            xmax = float(line.split("=")[1].strip())
        elif line.startswith("text ="):
            text = line.split("=", 1)[1].strip().strip('"')
            if text == "sp" and silences_same:
                text = "sil"
            if xmin is not None and xmax is not None:
                phonemes.append((xmin, xmax, text))
            xmin = xmax = None
    return phonemes


def wav_to_logmel(wav_path, standardize=True):
    samplerate, audio = wavfile.read(wav_path)
    if np.issubdtype(audio.dtype, np.integer):
        audio = (
            audio.astype(np.float32) / np.iinfo(audio.dtype).max
        )  # jeśli int, to normalizujemy do [-1, 1],
        # np.iinfo(audio.dtype).max zwraca maksymalną wartość dla danego typu danych, np. 32767 dla int16. Dzieląc przez to, skalujemy wartości do zakresu [-1, 1].
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Ensure the feature extractor sees a consistent sample rate.
    if samplerate != C.SAMPLE_RATE:
        warnings.warn(
            f"resampling from {samplerate} Hz to {C.SAMPLE_RATE} Hz for {wav_path}",
            RuntimeWarning,
        )
        n_samples = int(len(audio) * C.SAMPLE_RATE / samplerate)
        audio = signal.resample(audio, n_samples).astype(np.float32)
        samplerate = C.SAMPLE_RATE

    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, n_samples)
    mel = C.decibel_transformer(C.mel_transformer(wav)).squeeze(0)  # (n_mels, T)

    if standardize:
        mel = (mel - mel.mean(dim=1, keepdim=True)) / (
            mel.std(dim=1, keepdim=True) + 1e-8
        )

    return mel


def windows_and_labels(mel, phonemes):
    n_mels, T = mel.shape
    frame_dur = C.HOP_LENGTH / C.SAMPLE_RATE  # seconds per frame
    out = []
    for start in range(0, T - C.WIN_FRAMES + 1, C.SHIFT_FRAMES):
        end = start + C.WIN_FRAMES
        t_start = start * frame_dur
        t_end = end * frame_dur
        best_phone, best_overlap = None, 0.0
        for pmin, pmax, ptext in phonemes:
            if ptext not in C.PHONEMES:
                continue
            overlap = min(pmax, t_end) - max(pmin, t_start)
            if overlap > best_overlap:
                best_overlap, best_phone = overlap, ptext
        label = best_phone if best_phone is not None else C.NON_PHONEME
        out.append((mel[:, start:end].clone(), C.LABEL2IDX[label]))
    return out


def windows_and_labels_center(mel, phonemes):
    n_mels, T = mel.shape
    frame_dur = C.HOP_LENGTH / C.SAMPLE_RATE  # seconds per frame
    out = []
    for start in range(0, T - C.WIN_FRAMES + 1, C.SHIFT_FRAMES):
        end = start + C.WIN_FRAMES
        t_start = start * frame_dur
        t_end = end * frame_dur
        label = C.NON_PHONEME
        t_center = (t_start + t_end) / 2
        for pmin, pmax, ptext in phonemes:
            if pmin <= t_center < pmax and ptext in C.LABEL2IDX:
                label = ptext
                break
        out.append((mel[:, start:end].clone(), C.LABEL2IDX[label]))
    return out

def build_audio_cache(data_dir, output_path, silences_same=True):
    """Buduje cache z surowymi audio + parsedphonemes."""
    
    tg_paths = sorted(str(p) for p in Path(data_dir).rglob('*.TextGrid'))
    
    audio_data = {}        # wav_path -> torch.Tensor (audio samples)
    phoneme_data = {}      # wav_path -> list of (xmin, xmax, label_idx)
    
    for i, tg in enumerate(tg_paths):
        wav_path = tg[:-len('.TextGrid')] + '.wav'
        if not Path(wav_path).exists():
            continue
        
        try:
            sr, audio = wavfile.read(wav_path)
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)
            
            with open(tg, 'r', encoding='utf-8') as f:
                phonemes = parse_phonemes(f.read(), silences_same=silences_same)
            
            audio_data[wav_path] = torch.from_numpy(audio)
            phoneme_data[wav_path] = phonemes
        except Exception as e:
            print(f'skip {tg}: {e}')
        
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(tg_paths)}')
    
    torch.save({'audio': audio_data, 'phonemes': phoneme_data}, output_path)
    print(f'zapisano {len(audio_data)} plików')
class PhonemeWindowDataset(Dataset):
    def __init__(
        self,
        data_dir,
        max_files=None,
        verbose=True,
        standardize=True,
        silences_same=False,
        augment=False,
    ):
        # recursively find every .TextGrid at any depth under data_dir
        tg_paths = sorted(str(p) for p in Path(data_dir).rglob("*.TextGrid"))
        if verbose:
            print(f"found {len(tg_paths)} TextGrid files under {data_dir}")
        if max_files is not None:
            tg_paths = tg_paths[:max_files]
        xs, ys = [], []
        for i, tg in enumerate(tg_paths):
            wav = tg[: -len(".TextGrid")] + ".wav"
            if not os.path.exists(wav):
                continue
            try:
                mel = wav_to_logmel(wav, standardize=standardize)
                with open(tg, "r", encoding="utf-8") as f:
                    phonemes = parse_phonemes(f.read(), silences_same=silences_same)
                for w, lbl in windows_and_labels(mel, phonemes):
                    xs.append(w)
                    ys.append(lbl)
            except Exception as e:
                if verbose:
                    print(f"skip {tg}: {e}")
            if verbose and (i + 1) % 100 == 0:
                print(f"  processed {i + 1}/{len(tg_paths)} files, {len(xs)} windows")
        self.X = torch.stack(xs) if xs else torch.empty(0, C.N_MELS, C.WIN_FRAMES)
        self.y = torch.tensor(ys, dtype=torch.long)
        if verbose:
            print(f"total windows: {len(self.y)}")
            counts = torch.bincount(self.y, minlength=C.N_CLASSES).tolist()
            for l, c in zip(C.LABELS, counts):
                print(f"  {l:>9}: {c}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




    def __getitem__(self, idx):
        wav_path, start, label = self.entries[idx]
        mel = self._get_mel(wav_path)
        T = mel.shape[1]
        end = start + C.WIN_FRAMES
        
        # NEW — ochrona gdy mel jest krótszy niż się spodziewamy
        # (np. po tempo augmentation)
        if end > T:
            if T >= C.WIN_FRAMES:
                # przesuń okno żeby się zmieściło
                end = T
                start = end - C.WIN_FRAMES
            else:
                # mel za krótki nawet na jedno okno — pad zerami
                window = torch.zeros(C.N_MELS, C.WIN_FRAMES)
                window[:, :T] = mel
                return window, label
        
        window = mel[:, start:end].clone()
        return window, label

def build_augmented_cache(data_dir, output_path, n_augmentations=5, 
                          standardize=True, silences_same=True):
    """Buduje cache z N zaaugmentowanymi wariantami każdego pliku.
    
    Cache structure:
        X: (n_total_windows, n_aug, n_mels, win_frames)
        y: (n_total_windows,)
    """
    
    tg_paths = sorted(str(p) for p in Path(data_dir).rglob('*.TextGrid'))
    print(f'znaleziono {len(tg_paths)} plików')
    
    all_augmented_windows = []     # lista list — każdy plik ma N wariantów
    all_labels = []
    
    for i, tg in enumerate(tg_paths):
        wav_path = tg[:-len('.TextGrid')] + '.wav'
        if not Path(wav_path).exists():
            continue
        
        try:
            # 1. Wczytaj audio raz
            sr, audio_orig = wavfile.read(wav_path)
            if np.issubdtype(audio_orig.dtype, np.integer):
                audio_orig = audio_orig.astype(np.float32) / np.iinfo(audio_orig.dtype).max
            else:
                audio_orig = audio_orig.astype(np.float32)
            
            # 2. Wczytaj phoneme labels (raz, niezależne od augmentacji)
            with open(tg, 'r', encoding='utf-8') as f:
                phonemes = parse_phonemes(f.read(), silences_same=silences_same)
            
            # 3. Wygeneruj N zaaugmentowanych wersji
            file_windows_per_aug = []  # [n_aug] -> [n_windows] -> tensor
            file_labels = None         # labels są te same dla wszystkich aug (bez tempo!)
            
            for aug_idx in range(n_augmentations):
                # zaaugmentuj kopię
                audio_aug = augment_audio(audio_orig.copy(), sr)
                
                # zrób mel
                wav_t = torch.from_numpy(audio_aug).unsqueeze(0)
                mel = C.decibel_transformer(C.mel_transformer(wav_t)).squeeze(0)
                if standardize:
                    mel = (mel - mel.mean(dim=1, keepdim=True)) / (
                        mel.std(dim=1, keepdim=True) + 1e-8
                    )
                
                # pocięć na okienka
                windows_with_labels = windows_and_labels(mel, phonemes)
                
                if file_labels is None:
                    # zapisz labels tylko raz (te same dla wszystkich aug)
                    file_labels = [lbl for _, lbl in windows_with_labels]
                
                file_windows_per_aug.append([w for w, _ in windows_with_labels])
            
            # 4. Zsynchronizuj — dla każdego okna mamy n_aug wersji
            # Sprawdź że wszystkie aug mają tyle samo okien
            n_windows = len(file_windows_per_aug[0])
            if not all(len(w) == n_windows for w in file_windows_per_aug):
                print(f'skip {tg}: różne ilości okien między aug')
                continue
            
            # zbuduj tensor (n_windows, n_aug, n_mels, win_frames)
            stacked = torch.stack([
                torch.stack(file_windows_per_aug[a])    # (n_windows, n_mels, win_frames)
                for a in range(n_augmentations)
            ], dim=1)                                    # (n_windows, n_aug, ...)
            
            all_augmented_windows.append(stacked)
            all_labels.extend(file_labels)
            
        except Exception as e:
            print(f'skip {tg}: {e}')
        
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(tg_paths)} files, '
                  f'{sum(s.shape[0] for s in all_augmented_windows)} windows')
    
    X = torch.cat(all_augmented_windows, dim=0)
    y = torch.tensor(all_labels, dtype=torch.long)
    
    print(f'final: X={X.shape}, y={y.shape}')
    print(f'rozmiar: {X.element_size() * X.nelement() / 1e9:.2f} GB')
    
    torch.save({'X': X, 'y': y, 'n_aug': n_augmentations}, output_path)
    print(f'zapisano do {output_path}')

class CachedPhonemeDataset(Dataset):
    def __init__(self, X, y, augment=None):
        self.X = X
        self.y = y
        self.augment = augment
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment is not None:
            x = x.clone()           # nie modyfikuj cache
            x = self.augment(x)
        return x, self.y[idx]
    

class AugmentedCacheDataset(Dataset):
    """Dataset który losowo wybiera jeden z N wariantów."""
    
    def __init__(self, X, y, train=True):
        self.X = X         # (n_windows, n_aug, n_mels, win_frames)
        self.y = y
        self.train = train  # train: random aug, val: zawsze aug 0
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.train:
            aug_idx = torch.randint(self.X.shape[1], (1,)).item()
        else:
            aug_idx = 0    # walidacja deterministycznie
        return self.X[idx, aug_idx], self.y[idx]
    
class DoubledAugmentedCacheDataset(Dataset):
    """Każda próbka pojawia się dwa razy: raz aug, raz nie."""
    
    def __init__(self, X, y, train=True):
        self.X = X
        self.y = y
        self.train = train
    
    def __len__(self):
        # podwajamy tylko podczas treningu
        return len(self.y) * 2 if self.train else len(self.y)
    
    def __getitem__(self, idx):
        if not self.train:
            return self.X[idx, 0], self.y[idx]
        
        actual_idx = idx % len(self.y)
        is_augmented_pass = idx >= len(self.y)
        
        if is_augmented_pass:
            aug_idx = torch.randint(1, self.X.shape[1], (1,)).item()
        else:
            aug_idx = 0          # oryginał
        
        return self.X[actual_idx, aug_idx], self.y[actual_idx]