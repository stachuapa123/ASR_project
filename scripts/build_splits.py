"""
Generate the single global speaker partition (``data/splits.json``).

Both the CTC and P2G stages read this file so the TEST speakers are unseen by
the *whole* pipeline. Run once before retraining anything:

    uv run python -m scripts.build_splits --out data/splits.json

Re-running with the same ``--seed`` is deterministic.
"""

import argparse
import os

from src.p2g.data import discover_triples
from src.p2g.config import P2GConfig as P
from src.utils.paths import find_data_dir
from src.utils.speaker_split import split_speakers, save_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the global speaker partition.")
    parser.add_argument("--data-dir", default=None, help="PSD data root (auto if omitted).")
    parser.add_argument("--out", default="data/splits.json", help="Output JSON path.")
    parser.add_argument("--val-frac", type=float, default=P.VAL_SPEAKER_FRAC)
    parser.add_argument("--test-frac", type=float, default=P.TEST_SPEAKER_FRAC)
    parser.add_argument("--seed", type=int, default=P.SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or find_data_dir(project_root)
    if data_dir is None:
        raise SystemExit("No data directory found under ./data")

    speakers = sorted({spk for *_, spk in discover_triples(data_dir)})
    if len(speakers) < 3:
        raise SystemExit(f"Need >=3 speakers for a 3-way split, found {len(speakers)}")

    train, val, test = split_speakers(
        speakers, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    save_splits(args.out, train, val, test)
    print(
        f"[splits] {len(speakers)} speakers -> "
        f"train {len(train)} / val {len(val)} / test {len(test)} -> {args.out}"
    )
    print(f"[splits] val:  {val}")
    print(f"[splits] test: {test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
