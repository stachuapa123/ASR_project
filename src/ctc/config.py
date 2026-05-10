import torch


class CTCConfig:
    """
    Central configuration for CTC-based Polish ASR
    """

    # ---- signal / spectrogram ----
    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 160
    N_MELS = 128

    # ---- phoneme labels ----
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
        "sp",
        "t",
        "tS",
        "tsj",
        "u",
        "v",
        "w",
        "z",
        "zj",
    ]
    NON_PHONEME = "oov"

    LABELS = PHONEMES + [NON_PHONEME]
    LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}
    IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}
    N_CLASSES = len(LABELS)

    # Time reduction factor of the acoustic model (pooling on time axis).
    # With 2x MaxPool2d((2, 2)) time is reduced by ~4.
    TIME_REDUCTION_FACTOR = 4

    @staticmethod
    def get_device() -> torch.device:
        """
        Select a reasonable default device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
