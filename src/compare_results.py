from __future__ import annotations

from pathlib import Path

import json

from common import Paths


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    paths = Paths()
    paths.ensure()

    items = [
        ("TFIDF+LR", paths.results_dir / "tfidf_lr_metrics.json"),
        ("TFIDF+SVM", paths.results_dir / "tfidf_svm_metrics.json"),
        ("W2V+LR", paths.results_dir / "w2v_lr_metrics.json"),
    ]

    rows = []
    for name, p in items:
        m = load_metrics(p)
        if m is None:
            continue
        rows.append((name, m.get("macro_f1"), m.get("n_train"), m.get("n_test")))

    lines = ["# Model comparison", "", "| Model | Macro-F1 | Train | Test |", "|---|---:|---:|---:|"]
    for name, f1, n_train, n_test in rows:
        lines.append(f"| {name} | {f1:.4f} | {n_train} | {n_test} |")

    out_path = paths.results_dir / "model_comparison.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()