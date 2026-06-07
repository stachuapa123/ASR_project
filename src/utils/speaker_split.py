"""Shared speaker-disjoint partition for both stages (persisted to data/splits.json).

One global partition keeps the TEST speakers unseen by CTC *and* P2G. Speaker id =
parent directory name of a .wav/.TextGrid.
"""

import json
import random
from pathlib import Path


def three_way_slice(
    items: list, val_frac: float, test_frac: float, rng: random.Random
) -> tuple[list, list, list]:
    """Shuffle a copy of ``items`` and slice it into (train, val, test) lists."""
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    return items[n_test + n_val :], items[n_test : n_test + n_val], items[:n_test]


def split_speakers(
    speakers, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """Partition speaker ids into disjoint (train, val, test) lists (sorted, seeded)."""
    rng = random.Random(seed)
    return three_way_slice(sorted(set(speakers)), val_frac, test_frac, rng)


def save_splits(
    path: str | Path, train: list[str], val: list[str], test: list[str]
) -> None:
    """Write the speaker partition to JSON (the single source of truth)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Read a persisted speaker partition: ``{"train": [...], "val": [...], "test": [...]}``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
