"""Generate CLEC dependency collocation visualizations from JSON results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "ST3_dependency_collocations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matplotlib visualizations for CLEC dependency collocations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to ST3_dependency_collocations.json (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store generated PNG files (default: alongside the input JSON).",
    )
    parser.add_argument(
        "--top-n-group",
        type=int,
        default=15,
        help="Number of collocations to show in the low/high comparison chart (default: 15).",
    )
    parser.add_argument(
        "--write-tables",
        action="store_true",
        help="Also write CSV tables for normalized counts and type/token summary.",
    )
    return parser.parse_args()


def format_label(head: str, relation: str, dependent: str) -> str:
    return f"{head} {relation} {dependent}"


def plot_overall(entries: List[Dict[str, object]], output_path: Path) -> None:
    if not entries:
        print("No overall entries found in JSON data; skipping overall plot.")
        return

    labels = [
        format_label(item["head_lemma"], item["relation"], item["dependent_lemma"])
        for item in entries
    ]
    counts = [item["count"] for item in entries]

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, counts, color="#4c72b0")
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title("Overall Top 30 Dependency Collocations")
    for idx, value in enumerate(counts):
        ax.text(value + max(counts) * 0.01, idx, str(value), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved overall chart to {output_path}")


def build_group_table(
    low_entries: Iterable[Dict[str, object]],
    high_entries: Iterable[Dict[str, object]],
) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    table: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for group_name, entries in (("low", low_entries), ("high", high_entries)):
        for entry in entries:
            key = (
                str(entry["head_lemma"]),
                str(entry["relation"]),
                str(entry["dependent_lemma"]),
            )
            table.setdefault(key, {"low": 0, "high": 0})[group_name] = int(entry["count"])
    return table


def plot_group_comparison(
    low_entries: List[Dict[str, object]],
    high_entries: List[Dict[str, object]],
    limit: int,
    output_path: Path,
) -> None:
    table = build_group_table(low_entries, high_entries)
    if not table:
        print("No group comparison data found; skipping low/high chart.")
        return

    ranked = sorted(
        table.items(),
        key=lambda item: (-(item[1]["low"] + item[1]["high"]), item[0]),
    )[:limit]

    labels = [format_label(*key) for key, _ in ranked]
    low_counts = [values["low"] for _, values in ranked]
    high_counts = [values["high"] for _, values in ranked]
    indices = range(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar([i - width / 2 for i in indices], low_counts, width, label="Low", color="#dd8452")
    ax.bar([i + width / 2 for i in indices], high_counts, width, label="High", color="#4c72b0")
    ax.set_xticks(list(indices), labels=labels, rotation=45, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Top {limit} Collocations: Low vs High Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved group comparison chart to {output_path}")


def plot_differential(entries: List[Dict[str, object]], output_path: Path) -> None:
    if not entries:
        print("No differential entries found; skipping differential chart.")
        return

    labels = [
        format_label(entry["head_lemma"], entry["relation"], entry["dependent_lemma"])
        for entry in entries
    ]
    diffs = [entry["difference"] for entry in entries]
    colors = ["#4c72b0" if diff > 0 else "#dd8452" for diff in diffs]

    fig, ax = plt.subplots(figsize=(10, 12))
    positions = list(range(len(labels)))
    ax.barh(positions, diffs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("High - Low Frequency")
    ax.set_title("Differential Top 30 Collocations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved differential chart to {output_path}")


def write_csv_table(output_path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(data: Dict[str, object], output_dir: Path) -> None:
    overall_norm = data.get("overall_top_30_normalized", [])
    if isinstance(overall_norm, list) and overall_norm:
        write_csv_table(
            output_dir / "overall_top_30_normalized.csv",
            overall_norm,
            [
                "head_lemma",
                "relation",
                "dependent_lemma",
                "count",
                "count_per_1k_words",
                "count_per_1k_dependencies",
            ],
        )

    group_data = data.get("group_comparison", {})
    if isinstance(group_data, dict):
        for name in ("low", "high"):
            entries = group_data.get(f"{name}_top_30_normalized", [])
            if isinstance(entries, list) and entries:
                write_csv_table(
                    output_dir / f"{name}_top_30_normalized.csv",
                    entries,
                    [
                        "head_lemma",
                        "relation",
                        "dependent_lemma",
                        "count",
                        "count_per_1k_words",
                        "count_per_1k_dependencies",
                    ],
                )

        diff_norm = group_data.get("differential_top_30_normalized", [])
        if isinstance(diff_norm, list) and diff_norm:
            write_csv_table(
                output_dir / "differential_top_30_normalized.csv",
                diff_norm,
                [
                    "head_lemma",
                    "relation",
                    "dependent_lemma",
                    "low_count",
                    "high_count",
                    "difference",
                    "low_count_per_1k_words",
                    "high_count_per_1k_words",
                    "difference_per_1k_words",
                    "log_likelihood",
                    "significance",
                    "low_count_per_1k_dependencies",
                    "high_count_per_1k_dependencies",
                    "difference_per_1k_dependencies",
                ],
            )

    summary = data.get("summary", {})
    if isinstance(summary, dict) and summary:
        rows: List[Dict[str, object]] = []
        for key in ("overall", "low", "high", "unscored"):
            section = summary.get(key)
            if isinstance(section, dict):
                rows.append({"group": key, **section})
        if rows:
            fieldnames = ["group"]
            for row in rows:
                for col in row.keys():
                    if col not in fieldnames:
                        fieldnames.append(col)
            write_csv_table(output_dir / "type_token_summary.csv", rows, fieldnames)


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_overall(data.get("overall_top_30", []), output_dir / "overall_top_30.png")
    group_data = data.get("group_comparison", {})
    plot_group_comparison(
        group_data.get("low_top_30", []),
        group_data.get("high_top_30", []),
        args.top_n_group,
        output_dir / "group_top_15.png",
    )
    plot_differential(
        group_data.get("differential_top_30", []),
        output_dir / "differential_top_30.png",
    )

    if args.write_tables:
        write_tables(data, output_dir)
        print(f"Saved CSV tables to {output_dir}")


if __name__ == "__main__":
    main()
