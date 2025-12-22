from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import Paths, load_dataset, train_test_split_stratified, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze accuracy by text length bucket.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/tfidf_lr.joblib",
        help="Path to a joblib model file.",
    )
    return parser.parse_args()


def bucket_length(n: int) -> str:
    if n <= 20:
        return "0-20"
    if n <= 50:
        return "21-50"
    if n <= 100:
        return "51-100"
    return "101+"


def main() -> None:
    args = parse_args()
    paths = Paths()
    paths.ensure()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}. Train a model first.")

    df = load_dataset(paths.data_csv)
    train_df, test_df = train_test_split_stratified(df, test_size=0.2, seed=42)

    pipe = joblib.load(model_path)
    pred = pipe.predict(test_df["text"].tolist())

    out = test_df.copy()
    out["pred"] = pred
    out["length"] = out["text"].apply(len)
    out["bucket"] = out["length"].apply(bucket_length)
    out["is_correct"] = out["label"] == out["pred"]

    summary = (
        out.groupby("bucket", sort=False)
        .agg(count=("is_correct", "size"), acc=("is_correct", "mean"), avg_len=("length", "mean"))
        .reset_index()
    )

    model_stem = model_path.stem
    out_path = paths.results_dir / f"length_buckets_{model_stem}.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")

    payload = summary.to_dict(orient="records")
    save_json(payload, paths.results_dir / f"length_buckets_{model_stem}.json")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
