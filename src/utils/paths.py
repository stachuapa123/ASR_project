"""Filesystem discovery helpers shared by the check/build scripts."""

import os


def find_data_dir(base_dir: str) -> str | None:
    """Return the PSD corpus root ``base_dir/data`` if it exists, else None.

    Loaders discover ``*.TextGrid`` recursively, so the root covers every split.
    """
    data_dir = os.path.join(base_dir, "data")
    return data_dir if os.path.isdir(data_dir) else None


def find_ctc_checkpoint(base_dir: str) -> str | None:
    """Latest-by-mtime ``ctc_*.pt`` under ``base_dir/trained_models``, or None."""
    ckpt_dir = os.path.join(base_dir, "trained_models")
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = [
        os.path.join(ckpt_dir, name)
        for name in os.listdir(ckpt_dir)
        if name.startswith("ctc_") and name.endswith(".pt")
    ]
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)
