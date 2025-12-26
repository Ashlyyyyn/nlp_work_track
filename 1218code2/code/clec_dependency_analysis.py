"""
Run dependency parsing on CLEC ST3 essays, extract nsubj/amod relations,
and summarize their frequency overall and by score groups.
"""
from __future__ import annotations

import json
import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import stanza

# Project layout paths.
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
RECORDS_PATH = CLEC_DIR / "ST3_records.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "ST3_dependency_collocations.json"

TARGET_RELATIONS = {"nsubj", "amod"}
TOP_LIMIT = 30


@dataclass
class DependencyMatch:
    """Container for a single dependency relation occurrence."""

    essay_id: str
    title: str
    score: Optional[int]
    relation: str
    head: str
    head_lemma: str
    dependent: str
    dependent_lemma: str
    sentence: str

    def as_tuple(self) -> Tuple[str, str, str, str]:
        """Match the requested output format (head, relation, dependent, sentence)."""
        return self.head, self.relation, self.dependent, self.sentence

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Serialize to JSON-friendly structure."""
        return {
            "essay_id": self.essay_id,
            "title": self.title,
            "score": self.score,
            "relation": self.relation,
            "head": self.head,
            "head_lemma": self.head_lemma,
            "dependent": self.dependent,
            "dependent_lemma": self.dependent_lemma,
            "sentence": self.sentence,
        }


@dataclass
class CorpusTotals:
    essay_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    dependency_count: int = 0
    match_count: int = 0
    filtered_match_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "essay_count": self.essay_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "dependency_count": self.dependency_count,
            "match_count": self.match_count,
            "filtered_match_count": self.filtered_match_count,
        }


@dataclass(frozen=True)
class FilterConfig:
    filter_numeric_lemma: bool = False
    filter_nonalpha_lemma: bool = False
    min_lemma_len: Optional[int] = None
    exclude_lemmas: Set[str] = frozenset()

    def enabled(self) -> bool:
        return (
            self.filter_numeric_lemma
            or self.filter_nonalpha_lemma
            or self.min_lemma_len is not None
            or bool(self.exclude_lemmas)
        )

    @staticmethod
    def _is_numeric_like(lemma: str) -> bool:
        cleaned = lemma.strip().strip(".,:;!?'\"“”‘’()[]{}<>")
        cleaned = cleaned.replace("–", "").replace("—", "").replace("-", "").replace("/", "")
        cleaned = cleaned.replace("%", "").replace(",", "").replace(".", "")
        return bool(cleaned) and cleaned.isdigit()

    @staticmethod
    def _has_alpha(lemma: str) -> bool:
        return any(ch.isalpha() for ch in lemma)

    def excludes(self, lemma: str) -> bool:
        lemma = lemma.strip().lower()
        if not lemma:
            return True
        if lemma in self.exclude_lemmas:
            return True
        if self.min_lemma_len is not None and len(lemma) < self.min_lemma_len:
            return True
        if self.filter_numeric_lemma and self._is_numeric_like(lemma):
            return True
        if self.filter_nonalpha_lemma and not self._has_alpha(lemma):
            return True
        return False

    def excludes_match(self, head_lemma: str, dependent_lemma: str) -> bool:
        return self.excludes(head_lemma) or self.excludes(dependent_lemma)


def ensure_pipeline(
    lang: str = "en",
    use_gpu: Optional[bool] = None,
    *,
    resources_dir: Optional[Path] = None,
    download_models: bool = True,
    download_json: bool = True,
    resources_url: Optional[str] = None,
    model_url: Optional[str] = None,
) -> stanza.Pipeline:
    """
    Download the language model if needed and return a ready pipeline.

    Stanza caches downloaded models, so invoking download on every run is cheap.
    """
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        torch = None  # type: ignore[assignment]
        cuda_available = False

    if use_gpu is None:
        use_gpu = cuda_available
    elif use_gpu and not cuda_available:
        print("Warning: --use-gpu requested but CUDA is not available; falling back to CPU.", flush=True)
        use_gpu = False

    model_dir = str(resources_dir) if resources_dir is not None else None

    if download_models:
        try:
            stanza.download(
                lang,
                model_dir=model_dir,
                verbose=False,
                download_json=download_json,
                resources_url=resources_url or stanza.resources.common.DEFAULT_RESOURCES_URL,
                model_url=model_url or "default",
            )
        except Exception as exc:
            raise RuntimeError(
                "Stanza model download failed. If your server has no internet access, "
                "download the models on a machine with internet, copy the `stanza_resources/` "
                "folder to the server, then rerun with `--no-download-models --resources-dir <path>` "
                "(or set `STANZA_RESOURCES_DIR`)."
            ) from exc

    return stanza.Pipeline(
        lang=lang,
        dir=model_dir,
        processors="tokenize,pos,lemma,depparse",
        use_gpu=bool(use_gpu),
        verbose=False,
    )


def normalize_word(word: Optional["stanza.models.common.doc.Word"]) -> str:
    """Return a lower-cased lemma fallback to surface form when lemma is missing."""
    if word is None:
        return "root"
    lemma = word.lemma or word.text or ""
    return lemma.lower()


def sentence_text(sentence: "stanza.models.common.doc.Sentence") -> str:
    """Prefer the original sentence text; fall back to joining tokens."""
    if sentence.text:
        return sentence.text
    return " ".join(word.text for word in sentence.words)


def determine_cutoff(records: List[Dict[str, Optional[int]]]) -> Optional[float]:
    """Median score is used to split low/high groups."""
    numeric_scores = [record["score"] for record in records if isinstance(record.get("score"), int)]
    if not numeric_scores:
        return None
    return float(median(numeric_scores))


def score_group(score: Optional[int], cutoff: Optional[float]) -> Optional[str]:
    """Return 'low' or 'high' depending on the cutoff."""
    if cutoff is None or score is None:
        return None
    return "low" if score < cutoff else "high"


def iter_dependency_matches(
    records: List[Dict[str, object]],
    pipeline: stanza.Pipeline,
    *,
    cutoff: Optional[float] = None,
    filter_config: Optional[FilterConfig] = None,
    essay_stats_callback: Optional[Callable[[Optional[str], int, int, int], None]] = None,
    filter_reject_callback: Optional[Callable[[Optional[str]], None]] = None,
    log_every: Optional[int] = 20,
) -> Iterable[DependencyMatch]:
    """Yield DependencyMatch entries for each requested relation."""
    total = len(records)
    for index, record in enumerate(records, start=1):
        text = record.get("text", "")
        essay_id = str(record.get("id", ""))
        score = record.get("score")
        title = record.get("title", "")

        if not text:
            continue

        group = score_group(score if isinstance(score, int) else None, cutoff)
        doc = pipeline(text)
        word_count = sum(len(sentence.words) for sentence in doc.sentences)
        sentence_count = len(doc.sentences)
        dependency_count = sum(len(sentence.dependencies) for sentence in doc.sentences)
        if essay_stats_callback is not None:
            essay_stats_callback(group, word_count, sentence_count, dependency_count)

        for sentence in doc.sentences:
            sent_text = sentence_text(sentence)
            for governor, relation, dependent in sentence.dependencies:
                if relation not in TARGET_RELATIONS:
                    continue
                if governor is None or dependent is None:
                    continue
                head_lemma = normalize_word(governor)
                dep_lemma = normalize_word(dependent)
                if filter_config is not None and filter_config.enabled() and filter_config.excludes_match(
                    head_lemma, dep_lemma
                ):
                    if filter_reject_callback is not None:
                        filter_reject_callback(group)
                    continue
                yield DependencyMatch(
                    essay_id=essay_id,
                    title=title,
                    score=score if isinstance(score, int) else None,
                    relation=relation,
                    head=governor.text,
                    head_lemma=head_lemma,
                    dependent=dependent.text,
                    dependent_lemma=dep_lemma,
                    sentence=sent_text,
                )

        if log_every and log_every > 0 and index % log_every == 0:
            if total:
                print(f"Processed {index}/{total} essays...", flush=True)
            else:
                print(f"Processed {index} essays...", flush=True)


def top_entries(counter: Counter, limit: int = TOP_LIMIT) -> List[Dict[str, object]]:
    """Return a sorted top-N list of collocations."""
    entries: List[Dict[str, object]] = []
    for (head_lemma, relation, dep_lemma), count in counter.most_common(limit):
        entries.append(
            {
                "head_lemma": head_lemma,
                "relation": relation,
                "dependent_lemma": dep_lemma,
                "count": count,
            }
        )
    return entries


def rate_per_1k(count: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return (count / denominator) * 1000.0


def top_entries_normalized(
    counter: Counter,
    totals: CorpusTotals,
    limit: int = TOP_LIMIT,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for (head_lemma, relation, dep_lemma), count in counter.most_common(limit):
        entries.append(
            {
                "head_lemma": head_lemma,
                "relation": relation,
                "dependent_lemma": dep_lemma,
                "count": count,
                "count_per_1k_words": rate_per_1k(count, totals.word_count),
                "count_per_1k_dependencies": rate_per_1k(count, totals.dependency_count),
            }
        )
    return entries


def differential_entries(
    low_counter: Counter,
    high_counter: Counter,
    limit: int = TOP_LIMIT,
) -> List[Dict[str, object]]:
    """Highlight collocations with the largest absolute difference."""
    entries: List[Tuple[int, int, Tuple[str, str, str]]] = []
    for key in set(low_counter) | set(high_counter):
        low_value = low_counter.get(key, 0)
        high_value = high_counter.get(key, 0)
        diff = high_value - low_value
        abs_diff = abs(diff)
        if abs_diff == 0:
            continue
        entries.append((abs_diff, diff, key))

    # Sort by absolute difference desc, then by signed difference desc, then key.
    entries.sort(key=lambda item: (-item[0], -item[1], item[2]))

    formatted: List[Dict[str, object]] = []
    for abs_diff, diff, (head_lemma, relation, dep_lemma) in entries[:limit]:
        formatted.append(
            {
                "head_lemma": head_lemma,
                "relation": relation,
                "dependent_lemma": dep_lemma,
                "low_count": low_counter.get((head_lemma, relation, dep_lemma), 0),
                "high_count": high_counter.get((head_lemma, relation, dep_lemma), 0),
                "difference": diff,
                "absolute_difference": abs_diff,
            }
        )
    return formatted


def calculate_log_likelihood(count1: int, total1: int, count2: int, total2: int) -> Tuple[float, str]:
    """
    Calculate log-likelihood (LL) and significance level.
    Reference: Rayson, P., & Garside, R. (2000). Comparing corpora using frequency profiling.
    """
    if count1 + count2 == 0:
        return 0.0, "n.s."

    total = total1 + total2
    if total == 0:
        return 0.0, "n.s."

    # Expected frequencies.
    expected1 = total1 * (count1 + count2) / total
    expected2 = total2 * (count1 + count2) / total

    # Avoid log(0) errors by skipping zero-count terms.
    term1 = count1 * math.log(count1 / expected1) if count1 > 0 and expected1 > 0 else 0.0
    term2 = count2 * math.log(count2 / expected2) if count2 > 0 and expected2 > 0 else 0.0

    ll = 2 * (term1 + term2)

    # Significance thresholds for dof=1.
    if ll < 3.84:
        sig = "n.s."
    elif ll < 6.63:
        sig = "p<0.05"
    elif ll < 10.83:
        sig = "p<0.01"
    else:
        sig = "p<0.001"

    return round(ll, 2), sig


def differential_entries_normalized(
    low_counter: Counter,
    high_counter: Counter,
    low_totals: CorpusTotals,
    high_totals: CorpusTotals,
    limit: int = TOP_LIMIT,
) -> List[Dict[str, object]]:
    entries = differential_entries(low_counter, high_counter, limit)
    normalized: List[Dict[str, object]] = []
    total_words_low = low_totals.word_count
    total_words_high = high_totals.word_count
    for entry in entries:
        low_count = int(entry["low_count"])
        high_count = int(entry["high_count"])
        ll_score, significance = calculate_log_likelihood(
            low_count,
            total_words_low,
            high_count,
            total_words_high,
        )
        low_per_1k_words = rate_per_1k(low_count, low_totals.word_count)
        high_per_1k_words = rate_per_1k(high_count, high_totals.word_count)
        low_per_1k_deps = rate_per_1k(low_count, low_totals.dependency_count)
        high_per_1k_deps = rate_per_1k(high_count, high_totals.dependency_count)
        normalized.append(
            {
                **entry,
                "low_count_per_1k_words": low_per_1k_words,
                "high_count_per_1k_words": high_per_1k_words,
                "difference_per_1k_words": (
                    None
                    if low_per_1k_words is None or high_per_1k_words is None
                    else high_per_1k_words - low_per_1k_words
                ),
                "log_likelihood": ll_score,
                "significance": significance,
                "low_count_per_1k_dependencies": low_per_1k_deps,
                "high_count_per_1k_dependencies": high_per_1k_deps,
                "difference_per_1k_dependencies": (
                    None
                    if low_per_1k_deps is None or high_per_1k_deps is None
                    else high_per_1k_deps - low_per_1k_deps
                ),
            }
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLEC dependency collocations using Stanza."
    )
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=None,
        help=(
            "Directory for Stanza models (same as `STANZA_RESOURCES_DIR`). "
            "If omitted, Stanza uses its default."
        ),
    )
    parser.add_argument(
        "--resources-url",
        type=str,
        default=None,
        help=(
            "URL to fetch Stanza `resources.json` (advanced; use this if GitHub raw is blocked). "
            "Defaults to Stanza's built-in value."
        ),
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=None,
        help=(
            "Model download URL template (advanced; use this if HuggingFace is blocked). "
            "It must contain `{resources_version}`, `{lang}`, `{filename}` placeholders."
        ),
    )
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--download-models",
        dest="download_models",
        action="store_true",
        default=True,
        help="Download/refresh required Stanza models (default).",
    )
    download_group.add_argument(
        "--no-download-models",
        dest="download_models",
        action="store_false",
        help="Do not attempt any model downloads (useful on offline servers).",
    )
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument(
        "--download-json",
        dest="download_json",
        action="store_true",
        default=True,
        help="Download/refresh `resources.json` (default).",
    )
    json_group.add_argument(
        "--no-download-json",
        dest="download_json",
        action="store_false",
        help="Skip downloading `resources.json` (requires it already exists in resources dir).",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        default=None,
        help="Force Stanza to run on GPU (requires torch+CUDA).",
    )
    gpu_group.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        default=None,
        help="Force Stanza to run on CPU.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on the number of essays to process (useful for quick tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination JSON path. Defaults to ST3_dependency_collocations.json next to the source file.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Print a progress update after this many essays (set to 0 to disable).",
    )
    parser.add_argument(
        "--filter-noise",
        action="store_true",
        help=(
            "Enable default noise filtering (numeric-like lemmas, lemmas with no letters, and min lemma length 2)."
        ),
    )
    parser.add_argument(
        "--filter-numeric-lemma",
        action="store_true",
        help="Drop matches where head/dependent lemma is numeric-like (e.g., 40, 3.14, 40–).",
    )
    parser.add_argument(
        "--filter-nonalpha-lemma",
        action="store_true",
        help="Drop matches where head/dependent lemma contains no alphabetic characters.",
    )
    parser.add_argument(
        "--min-lemma-len",
        type=int,
        default=None,
        help="Drop matches where head/dependent lemma is shorter than this length (e.g., 2).",
    )
    parser.add_argument(
        "--exclude-lemmas",
        type=str,
        default="",
        help="Comma-separated list of lemmas to exclude (case-insensitive).",
    )
    parser.add_argument(
        "--exclude-lemmas-file",
        type=Path,
        default=None,
        help="Optional file containing lemmas to exclude (one per line, # comments allowed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    if args.max_records is not None:
        records = records[: args.max_records]
    print(f"Loaded {len(records)} essays from {RECORDS_PATH} (max_records={args.max_records}).")

    cutoff = determine_cutoff(records)

    pipeline = ensure_pipeline(
        "en",
        use_gpu=args.use_gpu,
        resources_dir=args.resources_dir,
        download_models=bool(args.download_models),
        download_json=bool(args.download_json),
        resources_url=args.resources_url,
        model_url=args.model_url,
    )

    overall_counter: Counter = Counter()
    group_counters: Dict[str, Counter] = {"low": Counter(), "high": Counter()}
    matches: List[DependencyMatch] = []

    totals_overall = CorpusTotals()
    totals_by_group: Dict[str, CorpusTotals] = {
        "low": CorpusTotals(),
        "high": CorpusTotals(),
        "unscored": CorpusTotals(),
    }

    exclude_lemmas: Set[str] = set()
    if args.exclude_lemmas:
        exclude_lemmas |= {part.strip().lower() for part in args.exclude_lemmas.split(",") if part.strip()}
    if args.exclude_lemmas_file is not None:
        for line in args.exclude_lemmas_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            exclude_lemmas.add(cleaned.lower())

    min_lemma_len = args.min_lemma_len
    filter_numeric = args.filter_numeric_lemma
    filter_nonalpha = args.filter_nonalpha_lemma
    if args.filter_noise:
        filter_numeric = True if not filter_numeric else filter_numeric
        filter_nonalpha = True if not filter_nonalpha else filter_nonalpha
        if min_lemma_len is None:
            min_lemma_len = 2

    filter_config = FilterConfig(
        filter_numeric_lemma=filter_numeric,
        filter_nonalpha_lemma=filter_nonalpha,
        min_lemma_len=min_lemma_len,
        exclude_lemmas=frozenset(exclude_lemmas),
    )

    def record_totals(group: Optional[str], word_count: int, sentence_count: int, dependency_count: int) -> None:
        totals_overall.essay_count += 1
        totals_overall.word_count += word_count
        totals_overall.sentence_count += sentence_count
        totals_overall.dependency_count += dependency_count

        group_key = group if group in ("low", "high") else "unscored"
        group_totals = totals_by_group[group_key]
        group_totals.essay_count += 1
        group_totals.word_count += word_count
        group_totals.sentence_count += sentence_count
        group_totals.dependency_count += dependency_count

    def record_filter_reject(group: Optional[str]) -> None:
        totals_overall.filtered_match_count += 1
        group_key = group if group in ("low", "high") else "unscored"
        totals_by_group[group_key].filtered_match_count += 1

    for match in iter_dependency_matches(
        records,
        pipeline,
        cutoff=cutoff,
        filter_config=filter_config,
        essay_stats_callback=record_totals,
        filter_reject_callback=record_filter_reject,
        log_every=args.log_every,
    ):
        matches.append(match)
        key = (match.head_lemma, match.relation, match.dependent_lemma)
        overall_counter[key] += 1
        totals_overall.match_count += 1

        group = score_group(match.score, cutoff)
        if group:
            group_counters[group][key] += 1
            totals_by_group[group].match_count += 1
        else:
            totals_by_group["unscored"].match_count += 1

    dependencies: List[Dict[str, object]] = []
    for match in matches:
        dependencies.append(
            {
                "head": match.head,
                "relation": match.relation,
                "dependent": match.dependent,
                "sentence": match.sentence,
                "essay_id": match.essay_id,
                "title": match.title,
                "score": match.score,
                "head_lemma": match.head_lemma,
                "dependent_lemma": match.dependent_lemma,
            }
        )

    results = {
        "metadata": {
            "source": str(RECORDS_PATH),
            "processed_essay_count": len(records),
            "match_count": len(matches),
            "filtered_match_count": totals_overall.filtered_match_count,
            "target_relations": sorted(TARGET_RELATIONS),
            "score_cutoff": cutoff,
            "score_group_definition": (
                "score < cutoff => low, score >= cutoff => high" if cutoff is not None else None
            ),
            "max_records": args.max_records,
            "stanza_use_gpu": bool(getattr(pipeline, "use_gpu", False)),
            "stanza_resources_dir": str(args.resources_dir) if args.resources_dir else None,
            "stanza_download_models": bool(args.download_models),
            "filters": {
                "enabled": filter_config.enabled(),
                "filter_numeric_lemma": filter_config.filter_numeric_lemma,
                "filter_nonalpha_lemma": filter_config.filter_nonalpha_lemma,
                "min_lemma_len": filter_config.min_lemma_len,
                "exclude_lemmas_count": len(filter_config.exclude_lemmas),
                "exclude_lemmas_file": str(args.exclude_lemmas_file) if args.exclude_lemmas_file else None,
            },
        },
        "summary": {
            "overall": {
                **totals_overall.to_dict(),
                "collocation_tokens": int(sum(overall_counter.values())),
                "collocation_types": int(len(overall_counter)),
                "type_token_ratio": (
                    None
                    if sum(overall_counter.values()) == 0
                    else len(overall_counter) / sum(overall_counter.values())
                ),
            },
            "low": {
                **totals_by_group["low"].to_dict(),
                "collocation_tokens": int(sum(group_counters["low"].values())),
                "collocation_types": int(len(group_counters["low"])),
                "type_token_ratio": (
                    None
                    if sum(group_counters["low"].values()) == 0
                    else len(group_counters["low"]) / sum(group_counters["low"].values())
                ),
            },
            "high": {
                **totals_by_group["high"].to_dict(),
                "collocation_tokens": int(sum(group_counters["high"].values())),
                "collocation_types": int(len(group_counters["high"])),
                "type_token_ratio": (
                    None
                    if sum(group_counters["high"].values()) == 0
                    else len(group_counters["high"]) / sum(group_counters["high"].values())
                ),
            },
            "unscored": totals_by_group["unscored"].to_dict(),
        },
        "dependencies": dependencies,
        "overall_top_30": top_entries(overall_counter, TOP_LIMIT),
        "overall_top_30_normalized": top_entries_normalized(overall_counter, totals_overall, TOP_LIMIT),
        "group_comparison": {
            "low_top_30": top_entries(group_counters["low"], TOP_LIMIT),
            "high_top_30": top_entries(group_counters["high"], TOP_LIMIT),
            "low_top_30_normalized": top_entries_normalized(
                group_counters["low"], totals_by_group["low"], TOP_LIMIT
            ),
            "high_top_30_normalized": top_entries_normalized(
                group_counters["high"], totals_by_group["high"], TOP_LIMIT
            ),
            "differential_top_30": differential_entries(
                group_counters["low"], group_counters["high"], TOP_LIMIT
            ),
            "differential_top_30_normalized": differential_entries_normalized(
                group_counters["low"],
                group_counters["high"],
                totals_by_group["low"],
                totals_by_group["high"],
                TOP_LIMIT,
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(matches)} dependency relations across {len(records)} essays.")
    print(f"Score cutoff: {cutoff!r}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
