from __future__ import annotations

from collections import Counter

import numpy as np

from common import Paths, load_dataset, save_json


def describe_lengths(texts: list[str]) -> dict:
    lengths = np.array([len(t) for t in texts], dtype=np.int32)
    if lengths.size == 0:
        return {"count": 0}
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
    }


def main() -> None:
    paths = Paths()
    paths.ensure()

    df = load_dataset(paths.data_csv)

    label_counts = Counter(df["label"].tolist())
    lengths_all = describe_lengths(df["text"].tolist())

    per_label = {}
    for label, g in df.groupby("label"):
        per_label[label] = describe_lengths(g["text"].tolist())

    payload = {
        "n_samples": int(len(df)),
        "n_labels": int(df["label"].nunique()),
        "label_counts": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))),
        "lengths_all": lengths_all,
        "lengths_per_label": per_label,
    }

    save_json(payload, paths.results_dir / "dataset_stats.json")

    lines = ["# Dataset stats", "", f"Samples: {payload['n_samples']}", f"Labels: {payload['n_labels']}", "", "## Label counts"]
    for k, v in payload["label_counts"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Text length (all)")
    for k, v in payload["lengths_all"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## Text length (per label)")
    for label, stats in payload["lengths_per_label"].items():
        lines.append(f"\n### {label}")
        for k, v in stats.items():
            lines.append(f"- {k}: {v}")

    (paths.results_dir / "dataset_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {paths.results_dir / 'dataset_stats.json'}")
    print(f"Wrote: {paths.results_dir / 'dataset_stats.md'}")


if __name__ == "__main__":
    main()