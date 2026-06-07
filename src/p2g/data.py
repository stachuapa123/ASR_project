"""Build (phone-string, text) pairs for P2G.

Phone source: ``mode="pred"`` = CTC greedy decode (learn to correct real errors);
``mode="clean"`` = oracle TextGrid phones. Target = sibling ``.txt``. Splits are
speaker-disjoint.
"""

import json
import random
import re
from pathlib import Path

import tqdm

from .config import P2GConfig as P
from src.ctc.config import CTCConfig as C
from src.ctc.inference import load_ctc_model, wav_to_phone_labels
from src.ctc.textgrid import textgrid_to_phone_ids
from src.utils.device import get_device
from src.utils.speaker_split import split_speakers, three_way_slice


_SILENCE = {"sil", "sp"}
_PUNCT_RE = re.compile(r"[^0-9a-ząćęłńóśźż\s]", re.IGNORECASE)


def format_phones(
    labels: list[str], keep_sil: bool = P.KEEP_SIL, sep: str = P.PHONE_SEP
) -> str:
    """Join phone labels into a single string, optionally dropping silences."""
    if not keep_sil:
        labels = [lbl for lbl in labels if lbl not in _SILENCE]
    return sep.join(labels)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (optional target variant)."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return " ".join(text.split())


def read_target(txt_path: str | Path, normalize: bool = P.NORMALIZE_TARGET) -> str:
    """Read a transcript, collapsing whitespace; keeps case + punctuation unless normalized."""
    text = " ".join(Path(txt_path).read_text(encoding="utf-8").split())
    return normalize_text(text) if normalize else text


def clean_phone_labels(tg_path: str | Path) -> list[str]:
    """Oracle phone labels from a TextGrid (includes sil)."""
    return [C.IDX2LABEL[i] for i in textgrid_to_phone_ids(tg_path)]


def discover_triples(data_root: str | Path) -> list[tuple[Path, Path, Path, str]]:
    """Find (wav, TextGrid, txt, speaker) tuples; speaker = the parent directory name."""
    triples: list[tuple[Path, Path, Path, str]] = []
    for tg in sorted(Path(data_root).rglob("*.TextGrid")):
        wav, txt = tg.with_suffix(".wav"), tg.with_suffix(".txt")
        if wav.exists() and txt.exists():
            triples.append((wav, tg, txt, tg.parent.name))
    return triples


def build_pairs(
    data_root: str | Path,
    mode: str = "pred",
    checkpoint: str | Path | None = None,
    device=None,
    keep_sil: bool = P.KEEP_SIL,
    normalize: bool = P.NORMALIZE_TARGET,
    max_files: int | None = None,
    shuffle: bool = False,
    sample_seed: int = P.SEED,
    speakers: set[str] | None = None,
    progress: bool = True,
) -> list[dict]:
    """Build ``{"phones","text","speaker"}`` rows. ``mode="pred"`` needs a CTC
    ``checkpoint``; ``speakers`` filters to those dirs; ``shuffle`` spreads a
    ``max_files`` cap across the corpus."""
    triples = discover_triples(data_root)
    if speakers is not None:
        speakers = set(speakers)
        triples = [t for t in triples if t[3] in speakers]
    if shuffle:
        random.Random(sample_seed).shuffle(triples)
    if max_files is not None:
        triples = triples[:max_files]

    predictor = None
    if mode == "pred":
        if checkpoint is None:
            raise ValueError("mode='pred' requires a CTC checkpoint")

        dev = device or get_device()
        ctc_model = load_ctc_model(checkpoint, dev)

        def predictor(wav):
            return wav_to_phone_labels(wav, ctc_model, dev)
    elif mode != "clean":
        raise ValueError(f"unknown mode: {mode!r} (expected 'pred' or 'clean')")

    rows: list[dict] = []
    iterator = tqdm.tqdm(triples, desc=f"P2G pairs ({mode})") if progress else triples
    for wav, tg, txt, speaker in iterator:
        text = read_target(txt, normalize=normalize)
        if not text.strip():
            continue
        labels = predictor(wav) if mode == "pred" else clean_phone_labels(tg)
        phones = format_phones(labels, keep_sil=keep_sil)
        if not phones.strip():
            continue
        rows.append({"phones": phones, "text": text, "speaker": speaker})
    return rows


def split_rows_by_speakers(
    rows: list[dict],
    val_speakers: set[str],
    test_speakers: set[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition ``rows`` by explicit val/test speaker sets (rest = train); pairs
    with a persisted ``data/splits.json``."""
    val_sp, test_sp = set(val_speakers), set(test_speakers)
    train = [
        r for r in rows if r["speaker"] not in test_sp and r["speaker"] not in val_sp
    ]
    val = [r for r in rows if r["speaker"] in val_sp]
    test = [r for r in rows if r["speaker"] in test_sp]
    return train, val, test


def split_by_speaker(
    rows: list[dict],
    val_frac: float = P.VAL_SPEAKER_FRAC,
    test_frac: float = P.TEST_SPEAKER_FRAC,
    seed: int = P.SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Speaker-disjoint split derived from ``rows`` (per-utterance fallback under 3
    speakers). Prefer ``split_rows_by_speakers`` with a persisted partition."""
    speakers = sorted({r["speaker"] for r in rows})

    if len(speakers) < 3:
        rng = random.Random(seed)
        return three_way_slice(rows, val_frac, test_frac, rng)

    _, val_speakers, test_speakers = split_speakers(speakers, val_frac, test_frac, seed)
    return split_rows_by_speakers(rows, set(val_speakers), set(test_speakers))


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_and_save(
    data_root: str | Path,
    out_dir: str | Path,
    mode: str = "pred",
    checkpoint: str | Path | None = None,
    device=None,
    max_files: int | None = None,
) -> dict[str, int]:
    """Build pairs, split by speaker, write train/val/test JSONL (shuffled so a
    ``max_files`` cap spans many speakers)."""
    rows = build_pairs(
        data_root,
        mode=mode,
        checkpoint=checkpoint,
        device=device,
        max_files=max_files,
        shuffle=True,
    )
    train, val, test = split_by_speaker(rows)

    out_dir = Path(out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    return {"train": len(train), "val": len(val), "test": len(test), "total": len(rows)}


def build_pipeline_dataset(
    data_root: str | Path,
    out_dir: str | Path,
    splits: dict[str, list[str]],
    checkpoint: str | Path,
    device=None,
    normalize: bool = P.NORMALIZE_TARGET,
    max_files: int | None = None,
) -> dict[str, int]:
    """Build train/val/test JSONL on the shared partition (``splits`` from ``load_splits``).

    train = oracle clean phones (corrupted at train time, see ``phone_noise``);
    val/test = real CTC predictions on unseen speakers.
    """
    train_rows = build_pairs(
        data_root, mode="clean", normalize=normalize,
        speakers=set(splits["train"]), max_files=max_files,
    )
    val_rows = build_pairs(
        data_root, mode="pred", checkpoint=checkpoint, device=device,
        normalize=normalize, speakers=set(splits["val"]), max_files=max_files,
    )
    test_rows = build_pairs(
        data_root, mode="pred", checkpoint=checkpoint, device=device,
        normalize=normalize, speakers=set(splits["test"]), max_files=max_files,
    )

    out_dir = Path(out_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "total": len(train_rows) + len(val_rows) + len(test_rows),
    }
