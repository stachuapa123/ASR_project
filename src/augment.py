
from .constants import Constants as C
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from collections import Counter



def augment_audio(audio, sr=C.SAMPLE_RATE,
                  noiseprob=0.3, gainprob=0.3, tempo_prob=0.0,
                  noise_level=(15, 30),     # SNR dB range
                  gain_range=(-3, 3),        # dB range
                  tempo_range=(0.9, 1.1)):
    """Augmentacja na surowym waveformie.
    Audio może być numpy array albo torch tensor — zwraca numpy array."""
    
    # NEW — konwersja na torch tensor jeśli numpy
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # 1. dodaj szum
    if torch.rand(1).item() < noiseprob:
        snr_db = noise_level[0] + (noise_level[1] - noise_level[0]) * torch.rand(1).item()
        signal_power = audio.pow(2).mean()
        if signal_power > 1e-10:
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(audio) * noise_power.sqrt()
            audio = audio + noise
    
    # 2. zmień głośność
    if torch.rand(1).item() < gainprob:
        gain_db = gain_range[0] + (gain_range[1] - gain_range[0]) * torch.rand(1).item()
        audio = audio * (10 ** (gain_db / 20))
    
    # 3. zmień tempo
    if torch.rand(1).item() < tempo_prob:
        rate = tempo_range[0] + (tempo_range[1] - tempo_range[0]) * torch.rand(1).item()
        audio = torchaudio.functional.speed(audio.unsqueeze(0), sr, rate)[0].squeeze(0)
    
    # 4. clamp i konwersja z powrotem na numpy
    audio = audio.clamp(-1.0, 1.0)
    return audio.numpy() 

class SpecAugment:
    def __init__(self, freq_mask_percent=0.1, time_mask_percent=0.125, p=0.3):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=int(freq_mask_percent * C.N_MELS))
        self.time_mask = T.TimeMasking(time_mask_param=int(time_mask_percent * C.WIN_FRAMES))
        self.p = p
    
    def __call__(self, mel):
        if torch.rand(1).item() < self.p:
            mel = self.freq_mask(mel)
        if torch.rand(1).item() < self.p:
            mel = self.time_mask(mel)
        return mel
    


