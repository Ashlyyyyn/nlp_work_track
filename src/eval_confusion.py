from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from common import Paths, load_dataset, train_test_split_stratified, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model with confusion matrix and report.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/tfidf_lr.joblib",
        help="Path to a joblib model file.",
    )
    return parser.parse_args()


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
    y_true = test_df["label"].tolist()
    y_pred = pipe.predict(test_df["text"].tolist())

    labels = sorted(df["label"].unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, digits=4)
    report_txt = classification_report(y_true, y_pred, labels=labels, digits=4)

    model_stem = model_path.stem
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = paths.results_dir / f"confusion_matrix_{model_stem}.csv"
    cm_df.to_csv(cm_path, index=True, encoding="utf-8")

    report_path = paths.results_dir / f"{model_stem}_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")

    save_json(report_dict, paths.results_dir / f"{model_stem}_report.json")

    print(f"Wrote: {cm_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
