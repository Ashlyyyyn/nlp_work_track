from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from common import Paths, load_dataset, train_test_split_stratified, set_seed, save_json


def main() -> None:
    set_seed(42)
    paths = Paths()
    paths.ensure()

    df = load_dataset(paths.data_csv)
    train_df, test_df = train_test_split_stratified(df, test_size=0.2, seed=42)

    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=200000)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    macro_f1 = f1_score(y_test, pred, average="macro")
    report = classification_report(y_test, pred, digits=4)

    print("Macro-F1:", macro_f1)
    print(report)

    model_path = paths.models_dir / "tfidf_nb.joblib"
    joblib.dump(pipe, model_path)

    save_json(
        {
            "model": "TFIDF + MultinomialNB",
            "macro_f1": float(macro_f1),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "vectorizer": {"analyzer": "char", "ngram_range": [2, 4], "min_df": 2},
            "clf": {"alpha": 0.5},
        },
        paths.results_dir / "tfidf_nb_metrics.json",
    )

    (paths.results_dir / "tfidf_nb_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
