from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import Paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top TF-IDF features for linear models.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/tfidf_lr.joblib",
        help="Path to a joblib model file.",
    )
    parser.add_argument("--topn", type=int, default=30, help="Top-N features per class.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths()
    paths.ensure()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}. Train a model first.")

    pipe = joblib.load(model_path)
    if "tfidf" not in pipe.named_steps or "clf" not in pipe.named_steps:
        raise ValueError("Expected a Pipeline with steps: tfidf -> clf.")

    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        raise ValueError("Classifier must expose coef_ (e.g., LogisticRegression or LinearSVC).")

    feature_names = tfidf.get_feature_names_out()
    coef = clf.coef_
    classes = clf.classes_.tolist()

    rows: list[dict] = []
    if coef.shape[0] == 1 and len(classes) == 2:
        weights = coef[0]
        order_pos = weights.argsort()[::-1][: args.topn]
        order_neg = weights.argsort()[: args.topn]
        for rank, idx in enumerate(order_pos, start=1):
            rows.append(
                {"class": classes[1], "feature": feature_names[idx], "weight": float(weights[idx]), "rank": rank}
            )
        for rank, idx in enumerate(order_neg, start=1):
            rows.append(
                {"class": classes[0], "feature": feature_names[idx], "weight": float(weights[idx]), "rank": rank}
            )
    else:
        for class_idx, class_name in enumerate(classes):
            weights = coef[class_idx]
            order = weights.argsort()[::-1][: args.topn]
            for rank, idx in enumerate(order, start=1):
                rows.append(
                    {"class": class_name, "feature": feature_names[idx], "weight": float(weights[idx]), "rank": rank}
                )

    out_df = pd.DataFrame(rows)
    model_stem = model_path.stem
    out_path = paths.results_dir / f"top_features_{model_stem}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
