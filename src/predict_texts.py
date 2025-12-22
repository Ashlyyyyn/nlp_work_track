from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import Paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict labels for input texts.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/tfidf_lr.joblib",
        help="Path to a joblib model file.",
    )
    parser.add_argument("--text", type=str, default="", help="Single text to predict.")
    parser.add_argument("--file", type=str, default="", help="Path to a text file (one line per sample).")
    return parser.parse_args()


def load_texts(text_arg: str, file_arg: str) -> list[str]:
    if text_arg:
        return [text_arg]
    if file_arg:
        p = Path(file_arg)
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")
        lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line]
    raise ValueError("Provide --text or --file.")


def main() -> None:
    args = parse_args()
    paths = Paths()
    paths.ensure()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}. Train a model first.")

    texts = load_texts(args.text, args.file)
    pipe = joblib.load(model_path)
    preds = pipe.predict(texts)

    out_df = pd.DataFrame({"text": texts, "pred": preds})
    model_stem = model_path.stem
    out_path = paths.results_dir / f"predictions_{model_stem}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    for text, pred in zip(texts, preds):
        print(f"[{pred}] {text}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
