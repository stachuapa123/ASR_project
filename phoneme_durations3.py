import os
import re
import statistics
from collections import defaultdict


def find_textgrid_files(folder, recursive=False):
    files = []
    if recursive:
        for root, dirs, filenames in os.walk(folder):
            dirs.sort()
            for f in sorted(filenames):
                if f.lower().endswith(".textgrid"):
                    files.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".textgrid"):
                files.append(os.path.join(folder, f))
    return files


def parse_textgrid(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tiers = {}
    current_tier = None
    current_intervals = []
    xmin = xmax = text_val = None

    for line in text.splitlines():
        line = line.strip()

        name_match = re.match(r'name = "(.+)"', line)
        if name_match:
            if current_tier:
                tiers[current_tier] = current_intervals
            current_tier = name_match.group(1)
            current_intervals = []
            xmin = xmax = text_val = None
            continue

        if line.startswith("xmin ="):
            xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax ="):
            xmax = float(line.split("=")[1].strip())
        elif line.startswith('text = "'):
            m = re.match(r'text = "(.*)"\s*$', line)
            if m:
                text_val = m.group(1)
                if xmin is not None and xmax is not None:
                    current_intervals.append((xmin, xmax, text_val))
                xmin = xmax = text_val = None

    if current_tier:
        tiers[current_tier] = current_intervals

    return tiers


def show_durations(path):
    """Show per-file phoneme durations and return stats for aggregation."""
    tiers = parse_textgrid(path)

    if "phones" not in tiers:
        print("  [!] No 'phones' tier found in this file.\n")
        return {}

    phones = [(xmin, xmax, text) for xmin, xmax, text in tiers["phones"] if text]

    #print(f"\n{'='*52}")
    #print(f"  File: {os.path.basename(path)}")
    #print(f"  Path: {os.path.dirname(path)}")
    #print(f"{'='*52}")
    #print(f"  {'Phoneme':<10} {'Start':>8}  {'End':>8}  {'Duration':>10}")
    #print(f"  {'-'*10} {'-'*8}  {'-'*8}  {'-'*10}")

    stats = defaultdict(list)

    for xmin, xmax, phone in phones:
        duration_ms = (xmax - xmin) * 1000
        stats[phone].append(duration_ms)
        #print(f"  {phone:<10} {xmin:>8.3f}s {xmax:>8.3f}s {duration_ms:>8.1f} ms")

    #print(f"\n{'='*52}")
    #print("  Per-phoneme statistics (ms)")
    #print(f"{'='*52}")
    #print(f"  {'Phoneme':<10} {'Count':>6}  {'Min':>8}  {'Avg':>8}  {'Std':>8}  {'Max':>8}")
    #print(f"  {'-'*10} {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for phone in sorted(stats):
        durations = stats[phone]
        std = statistics.stdev(durations) if len(durations) > 1 else 0.0
        #print(f"  {phone:<10} {len(durations):>6}  "
        #      f"{min(durations):>7.1f}ms  "
        #      f"{sum(durations)/len(durations):>7.1f}ms  "
        #      f"{std:>7.1f}ms  "
        #      f"{max(durations):>7.1f}ms")

    total = sum(xmax - xmin for xmin, xmax, _ in phones)
    #print(f"\n  Total phoneme time : {total*1000:.1f} ms")
    #print(f"  Phoneme count      : {len(phones)}")
    #print(f"  Average duration   : {total*1000/len(phones):.1f} ms")
    #print()

    return stats


def show_global_summary(global_stats, file_count):
    """Print an aggregate summary table across all parsed files."""
    if not global_stats:
        print("  [!] No data to summarise.\n")
        return

    print(f"\n{'='*68}")
    print(f"  GLOBAL SUMMARY  —  {file_count} file(s)")
    print(f"{'='*68}")
    print(f"  {'Phoneme':<10} {'Count':>7}  {'Total ms':>10}  "
          f"{'Min':>8}  {'Avg':>8}  {'Std':>8}  {'Max':>8}  {'% time':>7}")
    print(f"  {'-'*10} {'-'*7}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}")

    grand_total_ms = sum(sum(d) for d in global_stats.values())

    for phone in sorted(global_stats):
        durations  = global_stats[phone]
        count      = len(durations)
        total_ms   = sum(durations)
        avg_ms     = total_ms / count
        std_ms     = statistics.stdev(durations) if count > 1 else 0.0
        min_ms     = min(durations)
        max_ms     = max(durations)
        pct        = (total_ms / grand_total_ms * 100) if grand_total_ms else 0

        print(f"  {phone:<10} {count:>7}  {total_ms:>9.1f}ms  "
              f"{min_ms:>7.1f}ms  {avg_ms:>7.1f}ms  {std_ms:>7.1f}ms  "
              f"{max_ms:>7.1f}ms  {pct:>6.1f}%")

    grand_count = sum(len(d) for d in global_stats.values())
    print(f"\n  Total phoneme instances : {grand_count}")
    print(f"  Total phoneme time      : {grand_total_ms:.1f} ms  "
          f"({grand_total_ms/1000:.1f} s)")
    print(f"  Unique phonemes         : {len(global_stats)}")
    print()


def analyse_files(files):
    """Run analysis on a list of full file paths and show global summary."""
    global_stats = defaultdict(list)
    skipped = 0
    for path in files:
        file_stats = show_durations(path)
        if file_stats:
            for phone, durations in file_stats.items():
                global_stats[phone].extend(durations)
        else:
            skipped += 1
    show_global_summary(global_stats, len(files) - skipped)
    if skipped:
        print(f"  [!] {skipped} file(s) skipped (no 'phones' tier).\n")


def select_folder():
    current = os.getcwd()
    while True:
        print(f"\n  Current folder: {current}")
        entries = sorted(os.listdir(current))
        dirs = [e for e in entries if os.path.isdir(os.path.join(current, e))]

        print("\n  [0] Use this folder")
        for i, d in enumerate(dirs, 1):
            print(f"  [{i}] {d}/")
        if current != os.path.dirname(current):
            print("  [b] Go back up")

        choice = input("\n  Choice: ").strip().lower()

        if choice == "0":
            return current
        elif choice == "b":
            current = os.path.dirname(current)
        elif choice.isdigit() and 1 <= int(choice) <= len(dirs):
            current = os.path.join(current, dirs[int(choice) - 1])
        else:
            print("  Invalid choice, try again.")


def file_menu(folder, files):
    """Show file list and handle selection. Returns False when user goes back."""
    while True:
        print(f"\n  Found {len(files)} file(s) in: {folder}")
        print(f"  {'#':<5} {'Subfolder':<30} {'Filename'}")
        print(f"  {'-'*5} {'-'*30} {'-'*30}")
        for i, path in enumerate(files, 1):
            rel  = os.path.relpath(os.path.dirname(path), folder)
            name = os.path.basename(path)
            rel  = "" if rel == "." else rel
            print(f"  [{i:<3}] {rel:<30} {name}")

        print(f"\n  [a] Analyse all files + global summary")
        print(f"  [b] Back to folder selection")
        print(f"  [q] Quit")

        choice = input("\n  Select file number: ").strip().lower()

        if choice == "q":
            print("\n  Bye!\n")
            return "quit"
        elif choice == "b":
            return "back"
        elif choice == "a":
            analyse_files(files)
            input("  Press Enter to continue...")
        elif choice.isdigit() and 1 <= int(choice) <= len(files):
            show_durations(files[int(choice) - 1])
            input("  Press Enter to continue...")
        else:
            print("  Invalid choice, try again.")


def main():
    print("\n" + "="*52)
    print("  TextGrid Phoneme Duration Viewer")
    print("="*52)

    while True:
        print("\n  [1] Use current folder (this folder only)")
        print("  [2] Use current folder (include subfolders)")
        print("  [3] Browse to a folder (this folder only)")
        print("  [4] Browse to a folder (include subfolders)")
        print("  [q] Quit")
        mode = input("\n  Choice: ").strip().lower()

        if mode == "q":
            print("\n  Bye!\n")
            break

        if mode in ("1", "2", "3", "4"):
            recursive = mode in ("2", "4")
            folder = os.getcwd() if mode in ("1", "2") else select_folder()
        else:
            print("  Invalid choice.")
            continue

        files = find_textgrid_files(folder, recursive=recursive)

        if not files:
            label = "and subfolders " if recursive else ""
            print(f"\n  No .TextGrid files found in {label}{folder}")
            continue

        result = file_menu(folder, files)
        if result == "quit":
            break


if __name__ == "__main__":
    main()
