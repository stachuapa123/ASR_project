"""Filesystem discovery helpers shared by the check/build scripts."""

import os


def find_data_dir(base_dir: str) -> str | None:
    """Return the PSD corpus root ``base_dir/data`` if it exists, else None.

    The dataset loaders discover ``*.TextGrid`` files recursively, so returning
    the top-level ``data`` directory uses **every** split (``1-500`` …
    ``2501-3000``) rather than just the first one.
    """
    data_dir = os.path.join(base_dir, "data")
    return data_dir if os.path.isdir(data_dir) else None


def find_ctc_checkpoint(base_dir: str) -> str | None:
    """Most recently modified ``ctc_*.pt`` checkpoint under ``base_dir/trained_models``.

    Selected by modification time (the latest training run) rather than name
    order, which carries no quality/recency signal. Returns None if there is none.
    """
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
