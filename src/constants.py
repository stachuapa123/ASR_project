import torchaudio.transforms as T


class Constants:
    # ---- spectrogram ----
    SAMPLE_RATE = 16000          # 16000 pomiarow cisnienia na sekunde
    N_FFT       = 400            # 25 ms — tyle trwa jedna ramka (400 probek przy 16 kHz)
    HOP_LENGTH  = 160             # 10 ms -> 1 spec frame = 10 ms
    N_MELS      = 128

    # ---- classification window ----
    WIN_MS       = 80
    SHIFT_MS     = 20
    FRAME_MS     = HOP_LENGTH * 1000 // SAMPLE_RATE   # 10
    WIN_FRAMES   = WIN_MS   // FRAME_MS               # 8
    SHIFT_FRAMES = SHIFT_MS // FRAME_MS               # 2

    # ---- phonemes ----
    # silences are folded into 'sp' (handled in the parser/labeling step),
    PHONEMES = [
        'S', 'Z', 'a', 'b', 'c', 'd', 'dZ', 'dz', 'dzj', 'e', 'eo5', 'f', 'g',
        'h', 'i', 'i2', 'j', 'k', 'l', 'm', 'n', 'n~', 'o', 'oc5', 'p', 'r',
        's', 'sj', 'sil', 'sp', 't', 'tS', 'tsj', 'u', 'v', 'w', 'z', 'zj',
    ]
    NON_PHONEME = 'oov'
    LABELS    = PHONEMES + [NON_PHONEME]
    LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
    IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}
    N_CLASSES = len(LABELS)

    mel_transformer = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                          hop_length=HOP_LENGTH, n_mels=N_MELS)
    decibel_transformer = T.AmplitudeToDB() #decibel transform
