"""
Build the P2G (phone-string -> text) dataset as speaker-disjoint JSONL splits.

    uv run python -m scripts.build_p2g_data --mode pred \
        --checkpoint trained_models/ctc_all_augmentations_45epochs.pt --out data/p2g

mode=pred runs the trained CTC model over the audio (error-correcting target);
mode=clean uses oracle phones from the TextGrid alignments (no checkpoint needed).
"""

import argparse
import os

import torch

from src.p2g.data import build_and_save, build_pipeline_dataset
from src.utils.paths import find_data_dir
from src.utils.speaker_split import load_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P2G JSONL dataset.")
    parser.add_argument(
        "--data-dir", default=None, help="PSD data root (auto if omitted)."
    )
    parser.add_argument(
        "--out", default="data/p2g", help="Output dir for train/val/test JSONL."
    )
    parser.add_argument("--mode", choices=["pred", "clean"], default="pred")
    parser.add_argument(
        "--splits",
        default=None,
        help="Shared speaker partition JSON. When set, builds clean-train / "
        "pred-val+test using the global partition (recommended).",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="CTC checkpoint (required for --mode pred / --splits)."
    )
    parser.add_argument("--max-files", type=int, default=None)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or find_data_dir(project_root)
    if data_dir is None:
        raise SystemExit("No data directory found under ./data")
    if args.splits:
        if not args.checkpoint:
            raise SystemExit("--splits requires --checkpoint (val/test use CTC predictions)")
        print(f"[build] data_dir: {data_dir}  partition: {args.splits} (clean-train / pred-eval)")
        counts = build_pipeline_dataset(
            data_root=data_dir,
            out_dir=args.out,
            splits=load_splits(args.splits),
            checkpoint=args.checkpoint,
            device=torch.device(args.device),
            max_files=args.max_files,
        )
    else:
        if args.mode == "pred" and not args.checkpoint:
            raise SystemExit("--mode pred requires --checkpoint")
        print(f"[build] data_dir: {data_dir}  mode: {args.mode}")
        counts = build_and_save(
            data_root=data_dir,
            out_dir=args.out,
            mode=args.mode,
            checkpoint=args.checkpoint,
            device=torch.device(args.device),
            max_files=args.max_files,
        )
    print(f"[build] wrote splits to {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
