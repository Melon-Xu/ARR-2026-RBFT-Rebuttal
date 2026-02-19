#!/usr/bin/env python3
"""
replace_video_paths.py
======================
Replace the hardcoded dataset root prefix in all three RB-FT data files:

  - Open-R1-Video/data/smarthome_grpo.jsonl
  - Qwen-VL-Series-Finetune/rebuttal_scripts/data/reasoning_w_answer.json
  - Qwen-VL-Series-Finetune/rebuttal_scripts/data/sft_label.json

Usage
-----
    # Replace with a custom root (in-place):
    python3 replace_video_paths.py --new-root /your/dataset/root

    # Dry run (show what would change, do not write):
    python3 replace_video_paths.py --new-root /your/dataset/root --dry-run

    # Write to new output files instead of overwriting:
    python3 replace_video_paths.py --new-root /your/dataset/root --suffix _modified

The old prefix that is replaced is:
    /data/meilong/projects/Rational-Bootstrapped-Finetuning/dataset

After replacement the video paths will be:
    <new-root>/SmartHome-Bench-LLM/Videos/Trim_Videos/raw_video/smartbench_XXXX.mp4
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The prefix to replace (everything up to and including "dataset")
OLD_PREFIX = "/data/meilong/projects/Rational-Bootstrapped-Finetuning/dataset"

# Paths to the three data files, expressed relative to this script's directory
# (i.e. relative to ARR-2026-RBFT-Rebuttal/replace_path/)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR   = SCRIPT_DIR.parent          # ARR-2026-RBFT-Rebuttal/

DATA_FILES = [
    REPO_DIR / "Open-R1-Video" / "data" / "smarthome_grpo.jsonl",
    REPO_DIR / "Qwen-VL-Series-Finetune" / "rebuttal_scripts" / "data" / "reasoning_w_answer.json",
    REPO_DIR / "Qwen-VL-Series-Finetune" / "rebuttal_scripts" / "data" / "sft_label.json",
]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def replace_in_string(text: str, old_prefix: str, new_prefix: str) -> str:
    """Replace all occurrences of old_prefix with new_prefix in a string."""
    return text.replace(old_prefix, new_prefix)


def process_jsonl(path: Path, old_prefix: str, new_prefix: str, dry_run: bool, suffix: str) -> int:
    """Process a .jsonl file line-by-line. Returns the number of changed lines."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    changed = 0
    for line in lines:
        new_line = replace_in_string(line, old_prefix, new_prefix)
        if new_line != line:
            changed += 1
        new_lines.append(new_line)

    out_path = path if not suffix else path.with_name(path.stem + suffix + path.suffix)
    if not dry_run:
        out_path.write_text("".join(new_lines), encoding="utf-8")
    return changed


def process_json(path: Path, old_prefix: str, new_prefix: str, dry_run: bool, suffix: str) -> int:
    """Process a .json file (list of dicts). Returns the number of changed records."""
    text = path.read_text(encoding="utf-8")
    new_text = replace_in_string(text, old_prefix, new_prefix)
    changed = text.count(old_prefix)   # number of occurrences replaced

    out_path = path if not suffix else path.with_name(path.stem + suffix + path.suffix)
    if not dry_run and changed > 0:
        out_path.write_text(new_text, encoding="utf-8")
    return changed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Replace hardcoded dataset root paths in RB-FT data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--new-root",
        required=True,
        help=(
            "New dataset root directory. "
            "Replaces '%(old)s'." % {"old": OLD_PREFIX}
        ),
    )
    parser.add_argument(
        "--old-prefix",
        default=OLD_PREFIX,
        help=f"Old prefix to replace (default: {OLD_PREFIX})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without writing any files.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        metavar="SUFFIX",
        help=(
            "If provided, write to a new file with this suffix instead of "
            "overwriting. E.g. '--suffix _new' writes 'foo_new.json'."
        ),
    )
    args = parser.parse_args()

    new_prefix = args.new_root.rstrip("/")
    old_prefix = args.old_prefix.rstrip("/")

    if new_prefix == old_prefix:
        print("[warn] New root is identical to the old prefix — nothing to do.")
        sys.exit(0)

    print(f"Old prefix : {old_prefix}")
    print(f"New prefix : {new_prefix}")
    if args.dry_run:
        print("[dry-run]  No files will be written.\n")
    elif args.suffix:
        print(f"[output]   Writing to *{args.suffix}.{{json,jsonl}} copies.\n")
    else:
        print("[output]   Modifying files in-place.\n")

    total_changes = 0
    for data_file in DATA_FILES:
        if not data_file.exists():
            print(f"  [SKIP]  {data_file}  (file not found)")
            continue

        ext = data_file.suffix.lower()
        if ext == ".jsonl":
            n = process_jsonl(data_file, old_prefix, new_prefix, args.dry_run, args.suffix)
        elif ext == ".json":
            n = process_json(data_file, old_prefix, new_prefix, args.dry_run, args.suffix)
        else:
            print(f"  [SKIP]  {data_file}  (unsupported extension '{ext}')")
            continue

        status = "would change" if args.dry_run else "changed"
        print(f"  [OK]    {data_file.name}  —  {n} occurrence(s) {status}")
        total_changes += n

    print(f"\nTotal occurrences {'that would be' if args.dry_run else ''} replaced: {total_changes}")
    if total_changes == 0:
        print("[info] No occurrences of the old prefix were found.")


if __name__ == "__main__":
    main()
