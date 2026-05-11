import torch


class CTCConfig:
    """
    Central configuration for CTC-based Polish ASR
    """

    # Spectrogram parameters
    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 160
    N_MELS = 128

    # Phone labels
    PHONES = [
        "S",    # sz/rz like in przestrzeń
        "Z",    # ż/rz like in rzeka
        "a",
        "b",
        "c",
        "d",
        "dZ",   # dż
        "dz",
        "dzj",  # dź/dzi
        "e",
        "eo5",  # ę
        "f",
        "g",
        "h",    # h/ch
        "i",
        "i2",   # y
        "j",
        "k",
        "l",
        "m",
        "n",
        "n~",   # ń/ni
        "o",
        "oc5",  # ą
        "p",
        "r",
        "s",
        "sj",   # ś/si
        "sil",  # silence
        "t",
        "tS",   # cz
        "tsj",  # ć/ci
        "u",    # u/ó
        "v",    # w
        "w",    # ł
        "z",
        "zj",   # ź/zi
    ]

    BLANK_IDX = 0  # CTC blank token index
    LABEL2IDX = {label: idx + 1 for idx, label in enumerate(PHONES)}
    IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}
    N_CLASSES = len(PHONES) + 1  # Includes blank

    # Time reduction factor of the acoustic model (pooling on time axis).
    # With 2x MaxPool2d((2, 2)) time is reduced by ~4.
    TIME_REDUCTION_FACTOR = 4

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
