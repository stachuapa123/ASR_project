from pathlib import Path
from collections.abc import Mapping
from .config import CTCConfig as C


def parse_phone_intervals(
    text_grid: str,
    map_sp_to_sil: bool = True,
) -> list[tuple[float, float, str]]:
    """
    Extract (start, end, label) intervals from the "phones" tier of a TextGrid.
    """

    tier_name = "phones"  # Section in TextGrid to look for intervals in

    intervals: list[tuple[float, float, str]] = []

    in_phones = False
    xmin: float | None = None
    xmax: float | None = None

    for line in text_grid.split("\n"):
        line = line.strip()

        if f'name = "{tier_name}"' in line:
            in_phones = True
            continue

        if in_phones and line.startswith("name =") and tier_name not in line:
            break

        if not in_phones:
            continue

        if line.startswith("xmin =") and "intervals" not in line:
            xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax =") and "intervals" not in line:
            xmax = float(line.split("=")[1].strip())
        elif line.startswith("text ="):
            text = line.split("=", 1)[1].strip().strip('"')
            if map_sp_to_sil and text == "sp":
                text = "sil"
            if xmin is not None and xmax is not None:
                intervals.append((xmin, xmax, text))
            xmin = None
            xmax = None

    return intervals


def textgrid_to_phone_ids(
    textgrid_path: str | Path,
    label2idx: Mapping[str, int] = C.LABEL2IDX,
    map_sp_to_sil: bool = True,
) -> list[int]:
    """
    Parse a TextGrid file and map phones to integer indices.
    """

    textgrid_path = Path(textgrid_path)
    with textgrid_path.open("r", encoding="utf-8") as f:
        intervals = parse_phone_intervals(f.read(), map_sp_to_sil=map_sp_to_sil)

    indices: list[int] = []
    for _, _, label in intervals:
        if not label or label not in label2idx:
            continue
        indices.append(label2idx[label])

    return indices
