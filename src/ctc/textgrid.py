def parse_phoneme_intervals(
    text_grid: str,
    map_sp_to_sil: bool = True,
    tier_name: str = "phones",
) -> list[tuple[float, float, str]]:
    """
    Extract (start, end, label) intervals from the "phones" tier of a TextGrid.

    Args:
        text_grid: Entire TextGrid file as a string.
        map_sp_to_sil: If True, map "sp" (pause) to "sil".

    Returns:
        List of (xmin, xmax, label) tuples for each phone interval.
    """

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
