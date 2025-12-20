from __future__ import annotations

import joblib
import pandas as pd

from common import Paths, load_dataset, train_test_split_stratified


def main() -> None:
    paths = Paths()
    paths.ensure()

    model_path = paths.models_dir / "tfidf_lr.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run train_tfidf_lr.py first.")

    df = load_dataset(paths.data_csv)
    train_df, test_df = train_test_split_stratified(df, test_size=0.2, seed=42)

    pipe = joblib.load(model_path)
    pred = pipe.predict(test_df["text"].tolist())

    out = test_df.copy()
    out["pred"] = pred
    out["is_correct"] = out["label"] == out["pred"]

    errors = out[~out["is_correct"]].reset_index(drop=True)
    out_path = paths.results_dir / "tfidf_errors.csv"
    errors.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()