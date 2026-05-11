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

    # Phoneme labels
    PHONEMES = [
        "S",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "dZ",
        "dz",
        "dzj",
        "e",
        "eo5",
        "f",
        "g",
        "h",
        "i",
        "i2",
        "j",
        "k",
        "l",
        "m",
        "n",
        "n~",
        "o",
        "oc5",
        "p",
        "r",
        "s",
        "sj",
        "sil",
        "t",
        "tS",
        "tsj",
        "u",
        "v",
        "w",
        "z",
        "zj",
    ]

    BLANK_IDX = 0  # CTC blank token index
    LABEL2IDX = {label: idx + 1 for idx, label in enumerate(PHONEMES)}
    IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}
    N_CLASSES = len(PHONEMES) + 1  # Includes blank

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
