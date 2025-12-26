"""
Parse CLEC ST3 records, keep selected metadata (ID/SCORE/TITLE),
drop metadata tags from the running text, and show before/after samples.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Data source path relative to the project layout.
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
# Parsed JSON output lives next to ST3.txt.
OUTPUT_PATH = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_records.json")

# Regex to pull every angle-bracket tag (metadata such as <TITLE ...>).
ANGLE_TAG_PATTERN = re.compile(r"<[^>]+>")
# Reuse the earlier rule to strip CLEC error annotations like [sn2,s].
SQUARE_TAG_PATTERN = re.compile(r"\[[^\[\]]+\]")
# Records are separated by blank lines in ST3.
RECORD_SPLIT_PATTERN = re.compile(r"\r?\n\s*\r?\n")
# Number of sample pairs we print for quick verification.
SAMPLE_COUNT = 3


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def parse_score(value: str) -> Optional[int]:
    """Convert the SCORE payload to int when possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def strip_tags(text: str) -> str:
    """Remove metadata and error tags to produce clean essay text."""
    without_angle = ANGLE_TAG_PATTERN.sub(" ", text)
    without_square = SQUARE_TAG_PATTERN.sub(" ", without_angle)
    return normalize_whitespace(without_square)


def parse_record(block: str) -> Optional[Dict[str, Any]]:
    """Return a dictionary with id, score, title, and cleaned text."""
    metadata: Dict[str, Any] = {}
    for tag in ANGLE_TAG_PATTERN.findall(block):
        inner = tag[1:-1].strip()
        if not inner:
            continue
        key, _, value = inner.partition(" ")
        key = key.upper()
        value = value.strip()
        if key == "ID" and value:
            metadata["id"] = value
        elif key == "SCORE":
            metadata["score"] = parse_score(value)
        elif key == "TITLE" and value:
            metadata["title"] = value

    clean_text = strip_tags(block)
    if not clean_text or "id" not in metadata:
        return None

    metadata.setdefault("title", "")
    metadata.setdefault("score", None)
    metadata["text"] = clean_text
    return metadata


def iter_records(raw_text: str) -> Iterable[str]:
    """Yield non-empty record blocks separated by blank lines."""
    for chunk in RECORD_SPLIT_PATTERN.split(raw_text):
        stripped = chunk.strip()
        if stripped:
            yield stripped


def snippet(text: str, limit: int = 200) -> str:
    """Compress whitespace and truncate long text for logging."""
    clean = normalize_whitespace(text)
    return clean if len(clean) <= limit else f"{clean[:limit].rstrip()}..."


def main() -> None:
    raw_text = INPUT_PATH.read_text(encoding="utf-8-sig")
    parsed_records: List[Dict[str, Any]] = []
    sample_pairs: List[Tuple[str, Dict[str, Any]]] = []

    for block in iter_records(raw_text):
        record = parse_record(block)
        if record is None:
            continue
        parsed_records.append(record)
        if len(sample_pairs) < SAMPLE_COUNT:
            sample_pairs.append((block, record))

    OUTPUT_PATH.write_text(json.dumps(parsed_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Parsed {len(parsed_records)} records.")
    print(f"Structured output saved to: {OUTPUT_PATH}")
    print("\nSample before/after pairs:\n")
    for idx, (raw_block, record) in enumerate(sample_pairs, start=1):
        print(f"--- Record {idx} ---")
        print(f"Original snippet: {snippet(raw_block)}")
        print(f"Cleaned snippet : {snippet(record['text'])}")
        print(
            f"Metadata        : "
            f"id={record['id']}, score={record['score']}, title={record['title']}"
        )
        print()

    print("First 2 structured entries:")
    print(json.dumps(parsed_records[:2], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
