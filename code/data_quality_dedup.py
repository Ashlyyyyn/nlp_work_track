from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\\s+", " ", s)
    s = re.sub(r"[^\\w\\s\\u4e00-\\u9fff]", "", s)
    return s


def shingles(s: str, k: int = 5) -> set[str]:
    s = normalize_text(s)
    if len(s) <= k:
        return {s}
    return {s[i : i + k] for i in range(len(s) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a.union(b))
    return inter / max(union, 1)


def stable_hash(s: str) -> str:
    return blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class DedupResult:
    kept: list[str]
    removed: list[tuple[int, int, float]]


def dedup_texts(texts: Iterable[str], near_threshold: float = 0.9) -> DedupResult:
    kept: list[str] = []
    kept_shingles: list[set[str]] = []
    kept_hashes: set[str] = set()
    removed: list[tuple[int, int, float]] = []

    for i, t in enumerate(texts):
        t_norm = normalize_text(t)
        h = stable_hash(t_norm)
        if h in kept_hashes:
            removed.append((i, -1, 1.0))
            continue

        sh = shingles(t_norm)
        best_j = -1
        best_sim = 0.0
        for j, sh_j in enumerate(kept_shingles):
            sim = jaccard(sh, sh_j)
            if sim > best_sim:
                best_sim = sim
                best_j = j

        if best_sim >= near_threshold:
            removed.append((i, best_j, best_sim))
            continue

        kept.append(t)
        kept_shingles.append(sh)
        kept_hashes.add(h)

    return DedupResult(kept=kept, removed=removed)


def main() -> None:
    texts = [
        "OpenAI released a new model.",
        "OpenAI released a new model!",
        "OpenAI releases a new model today.",
        "完全不同的一句话。",
        "  完全不同的一句话  ",
    ]
    res = dedup_texts(texts, near_threshold=0.9)
    print("kept:")
    for t in res.kept:
        print("-", t)
    print("removed (i, matched_j, sim):")
    for i, j, sim in res.removed:
        print("-", i, j, f"{sim:.2f}")


if __name__ == "__main__":
    main()
