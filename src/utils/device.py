"""Device selection shared across models."""

import torch


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
