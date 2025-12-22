from __future__ import annotations

import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from common import Paths, load_dataset, set_seed, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold evaluation for TF-IDF + LogisticRegression.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    paths = Paths()
    paths.ensure()

    df = load_dataset(paths.data_csv)
    X = df["text"].tolist()
    y = df["label"].tolist()

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    scores: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=200000)),
                ("clf", LogisticRegression(max_iter=2000, n_jobs=1, class_weight="balanced")),
            ]
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = f1_score(y_test, pred, average="macro")
        scores.append(float(score))
        print(f"Fold {fold_idx}: Macro-F1 = {score:.4f}")

    payload = {
        "model": "TFIDF + LogisticRegression",
        "k": int(args.k),
        "macro_f1_scores": scores,
        "macro_f1_mean": float(np.mean(scores)),
        "macro_f1_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    }

    save_json(payload, paths.results_dir / "tfidf_lr_kfold.json")
    print(f"Mean Macro-F1: {payload['macro_f1_mean']:.4f}")


if __name__ == "__main__":
    main()
