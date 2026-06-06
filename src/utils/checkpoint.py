"""Checkpoint weight loading shared across models."""

from pathlib import Path
from typing import Any

import torch


def load_weights(
    path: str | Path,
    model: torch.nn.Module,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load a model's weights from a checkpoint into ``model`` in place.

    Accepts both standardized checkpoints (a dict with a ``model_state_dict``
    key plus metadata) and bare ``state_dict`` files. Returns the remaining
    metadata (everything except ``model_state_dict``), or an empty dict.
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        state = ckpt
        meta = {}
    model.load_state_dict(state)
    return meta
