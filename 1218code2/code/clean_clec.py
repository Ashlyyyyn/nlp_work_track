"""
Utility script for cleaning CLEC (Chinese Learner English Corpus) files.
The script reads `ST3.txt`, removes bracketed error tags, writes the
clean text to `ST3_clean.txt`, and prints a short before/after sample.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple

# Input corpus path relative to the project layout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def locate_clec_dir() -> Path:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_ROOT}")
    for candidate in DATA_ROOT.iterdir():
        clec_dir = candidate / "CLEC"
        if clec_dir.is_dir():
            return clec_dir
    raise FileNotFoundError("No CLEC directory found under data/.")


CLEC_DIR = locate_clec_dir()
INPUT_PATH = CLEC_DIR / "ST3.txt"
# Clean output lives in the same directory with a descriptive suffix.
OUTPUT_PATH = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_clean.txt")

# Regex removes anything that looks like `[xxx]`, such as `[sn2,s]`.
ERROR_TAG_PATTERN = re.compile(r"\[[^\[\]]+\]")
# Number of (original, cleaned) line pairs to display as a sanity check.
SAMPLE_LINE_COUNT = 5


def load_text(path: Path) -> str:
    """Read text as UTF-8 while safely stripping a UTF-8 BOM if present."""
    return path.read_text(encoding="utf-8-sig")


def strip_error_tags(text: str) -> str:
    """Remove CLEC error annotations enclosed in square brackets."""
    return ERROR_TAG_PATTERN.sub("", text)


def save_text(path: Path, text: str) -> None:
    """Persist cleaned text so that downstream tools (e.g., Stanza) can use it."""
    path.write_text(text, encoding="utf-8")


def iter_sample_pairs(original: str, cleaned: str, count: int) -> Iterable[Tuple[str, str]]:
    """Yield up to `count` tuples of (original line, cleaned line) for comparison."""
    orig_lines = original.splitlines()
    clean_lines = cleaned.splitlines()
    upper = min(count, len(orig_lines), len(clean_lines))
    for idx in range(upper):
        yield orig_lines[idx], clean_lines[idx]


def main() -> None:
    """Load, clean, save, and display sample lines."""
    original_text = load_text(INPUT_PATH)
    cleaned_text = strip_error_tags(original_text)
    save_text(OUTPUT_PATH, cleaned_text)

    # Display a few sample lines so we can quickly verify the cleaning effect.
    print(f"Clean text written to: {OUTPUT_PATH}")
    print(f"\nFirst {SAMPLE_LINE_COUNT} line pairs (original vs cleaned):\n")
    for idx, (orig, clean) in enumerate(iter_sample_pairs(original_text, cleaned_text, SAMPLE_LINE_COUNT), start=1):
        print(f"--- Sample line {idx} ---")
        print(f"Original: {orig}")
        print(f"Cleaned : {clean}")
        print()


if __name__ == "__main__":
    main()
