import torch
import torchaudio.transforms as T


class CTCConfig:
    """
    Central configuration for CTC-based Polish ASR:
    - signal / spectrogram parameters
    - phoneme inventory
    - shared transforms
    """

    # ---- signal / spectrogram ----
    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 160
    N_MELS = 128

    # ---- windowing (for frame-based tasks, if needed) ----
    WIN_MS = 80
    SHIFT_MS = 20
    FRAME_MS = HOP_LENGTH * 1000 // SAMPLE_RATE
    WIN_FRAMES = WIN_MS // FRAME_MS
    SHIFT_FRAMES = SHIFT_MS // FRAME_MS

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

    # ---- transforms (CPU by design; move tensors to CUDA in the trainer) ----
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=True,
        power=2.0,
    )
    db_transform = T.AmplitudeToDB(stype="power")

    # Time reduction factor of the acoustic model (pooling on time axis).
    # With two MaxPool2d((2, 2)) layers, time is reduced by ~4.
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

    @staticmethod
    def setup_cuda_optimizations() -> None:
        """
        Enable CUDA/cuDNN optimizations for faster training on modern GPUs.
        Particularly beneficial for RTX 4070 Ti SUPER and similar high-end cards.
        """
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
