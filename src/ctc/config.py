class CTCConfig:
    """
    Central configuration for CTC-based Polish ASR
    """

    # Spectrogram parameters
    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 160
    N_MELS = 128

    WIN_MS = 80
    SHIFT_MS = 20
    FRAME_MS = HOP_LENGTH * 1000 // SAMPLE_RATE  # 10
    WIN_FRAMES = WIN_MS // FRAME_MS  # 8
    SHIFT_FRAMES = SHIFT_MS // FRAME_MS

    # Phone labels
    PHONES = [
        "S",  # sz/rz like in przestrzeń
        "Z",  # ż/rz like in rzeka
        "a",
        "b",
        "c",
        "d",
        "dZ",  # dż
        "dz",
        "dzj",  # dź/dzi
        "e",
        "eo5",  # ę
        "f",
        "g",
        "h",  # h/ch
        "i",
        "i2",  # y
        "j",
        "k",
        "l",
        "m",
        "n",
        "n~",  # ń/ni
        "o",
        "oc5",  # ą
        "p",
        "r",
        "s",
        "sj",  # ś/si
        "sil",  # silence
        "t",
        "tS",  # cz
        "tsj",  # ć/ci
        "u",  # u/ó
        "v",  # w
        "w",  # ł
        "z",
        "zj",  # ź/zi
    ]

    BLANK_IDX = 0  # CTC blank token index
    LABEL2IDX = {label: idx + 1 for idx, label in enumerate(PHONES)}
    IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}
    N_CLASSES = len(PHONES) + 1  # Includes blank

    # Time reduction factor of the acoustic model (pooling on time axis)
    TIME_REDUCTION_FACTOR = 2
