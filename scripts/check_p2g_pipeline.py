"""
End-to-end smoke test for the P2G (phone -> text) stage.

Builds a few (phone, text) pairs, fine-tunes a tiny seq2seq for a step or two,
saves+reloads it, transcribes a sample, and reports WER/CER. Wiring check only:
the default model is a tiny random T5, so the numbers are meaningless.

    uv run python -m scripts.check_p2g_pipeline [--data-dir DIR] [--model NAME]

Use a real model (e.g. allegro/plt5-small) + more epochs for a meaningful run.
"""

import argparse
import os
import shutil
import tempfile

import torch

from src.ctc.inference import load_ctc_model, wav_to_phone_labels
from src.p2g.data import build_pairs, split_by_speaker
from src.p2g.model import P2GModel
from src.p2g.train import evaluate, train_p2g
from src.utils.paths import find_ctc_checkpoint, find_data_dir


def log(msg: str) -> None:
    print(f"[check] {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2G pipeline smoke test.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-t5")
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device(args.device)

    data_dir = args.data_dir or find_data_dir(project_root)
    if data_dir is None:
        raise SystemExit("No data directory found under ./data")
    log(f"data_dir: {data_dir}  device: {device}  model: {args.model}")

    # 1. Build clean (phone, text) pairs -- fast, no CTC checkpoint needed.
    rows = build_pairs(data_dir, mode="clean", max_files=args.max_files, progress=False)
    if not rows:
        raise SystemExit("No (phone, text) pairs built from the data dir")
    train_rows, val_rows, test_rows = split_by_speaker(rows)
    log(
        f"pairs: {len(rows)} (train {len(train_rows)} / val {len(val_rows)} / test {len(test_rows)})"
    )
    log(f"example phones: {rows[0]['phones'][:80]}...")
    log(f"example text:   {rows[0]['text'][:80]}")

    tmp_dir = tempfile.mkdtemp(prefix="p2g_check_")
    try:
        # 2. Fine-tune the tiny model and save.
        log("fine-tuning (tiny model, wiring check)...")
        p2g, _ = train_p2g(
            train_rows,
            val_rows or test_rows,
            model_name=args.model,
            output_dir=tmp_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            num_beams=1,
        )

        # 3. Reload from disk and transcribe a sample.
        reloaded = P2GModel.from_pretrained(tmp_dir, device=device)
        sample = (test_rows or val_rows or train_rows)[0]
        pred = reloaded.transcribe(sample["phones"], num_beams=1)
        log(f"sample reference : {sample['text'][:80]}")
        log(f"sample prediction: {pred[:80]}")

        # 4. Report WER/CER on the held-out split.
        eval_rows = test_rows or val_rows or train_rows
        m = evaluate(reloaded, eval_rows, batch_size=args.batch_size, num_beams=1)
        log(
            f"held-out WER {m['wer']:.3f}  CER {m['cer']:.3f} (meaningless for the tiny model)"
        )

        # 5. If a real CTC checkpoint exists, exercise the wav -> phones path too.
        ckpt = find_ctc_checkpoint(project_root)
        if ckpt:
            from src.p2g.data import discover_triples

            wav = discover_triples(data_dir)[0][0]
            ctc = load_ctc_model(ckpt, device)
            labels = wav_to_phone_labels(str(wav), ctc, device)
            log(
                f"CTC phones for {os.path.basename(str(wav))}: {' '.join(labels[:20])}..."
            )
            log(f"end-to-end text: {reloaded.transcribe(labels, num_beams=1)[:80]}")
        else:
            log("no CTC checkpoint found -> skipped wav->phones path")

        log("P2G PIPELINE CHECK OK")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
