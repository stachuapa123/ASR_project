"""
Estimate the CTC phone-error profile on the speaker-disjoint VAL set and save it
to ``data/phone_noise.json`` for P2G training-time augmentation.

    uv run python -m scripts.build_phone_noise \
        --checkpoint trained_models/ctc_speaker_disjoint.pt \
        --splits data/splits.json --out data/phone_noise.json

The VAL speakers are unseen by the CTC, so the measured errors reflect
deployment, not memorized training audio. See src/p2g/phone_noise.py.
"""

import argparse
import os

import torch
import tqdm

from src.ctc.inference import load_ctc_model, wav_to_phone_labels
from src.p2g.data import clean_phone_labels, discover_triples
from src.p2g.phone_noise import PhoneNoiseProfile, save_profile
from src.utils.paths import find_data_dir
from src.utils.speaker_split import load_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CTC phone-noise profile.")
    parser.add_argument("--data-dir", default=None, help="PSD data root (auto if omitted).")
    parser.add_argument("--checkpoint", required=True, help="CTC checkpoint.")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--out", default="data/phone_noise.json")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
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

    speakers = set(load_splits(args.splits)[args.split])
    triples = [t for t in discover_triples(data_dir) if t[3] in speakers]
    if args.max_files is not None:
        triples = triples[: args.max_files]
    if not triples:
        raise SystemExit(f"No files for split '{args.split}' speakers under {data_dir}")

    device = torch.device(args.device)
    ctc = load_ctc_model(args.checkpoint, device)

    pairs: list[tuple[list[str], list[str]]] = []
    for wav, tg, _txt, _spk in tqdm.tqdm(triples, desc="phone-noise"):
        hyp = wav_to_phone_labels(str(wav), ctc, device)
        ref = clean_phone_labels(tg)
        pairs.append((ref, hyp))

    profile = PhoneNoiseProfile.estimate(pairs)
    save_profile(args.out, profile)
    print(
        f"[phone-noise] {len(pairs)} utts on '{args.split}' speakers -> {args.out}\n"
        f"[phone-noise] reproduced PER ~= {profile.expected_per():.3f} "
        f"(this is the deployment noise P2G will train against)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
