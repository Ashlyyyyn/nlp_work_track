from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from common import Paths, load_dataset, train_test_split_stratified, set_seed, save_json


def char_tokenize(text: str) -> List[str]:
    # 简单且稳定：按字符（也可以替换为分词）
    text = text.strip()
    return [ch for ch in text if not ch.isspace()]


def doc_vector(model: Word2Vec, tokens: List[str]) -> np.ndarray:
    vecs = []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def main() -> None:
    set_seed(42)
    paths = Paths()
    paths.ensure()

    df = load_dataset(paths.data_csv)
    train_df, test_df = train_test_split_stratified(df, test_size=0.2, seed=42)

    train_tokens = [char_tokenize(t) for t in train_df['text'].tolist()]
    test_tokens = [char_tokenize(t) for t in test_df['text'].tolist()]

    # 训练 Word2Vec（小数据也能跑，主要用于“对比+过程证据”）
    w2v = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=2,
        sg=1,  # skip-gram
        epochs=10,
    )

    X_train = np.stack([doc_vector(w2v, toks) for toks in train_tokens], axis=0)
    X_test = np.stack([doc_vector(w2v, toks) for toks in test_tokens], axis=0)
    y_train = train_df['label'].tolist()
    y_test = test_df['label'].tolist()

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, pred, average='macro')
    report = classification_report(y_test, pred, digits=4)

    print('Macro-F1:', macro_f1)
    print(report)

    # 保存：w2v + clf
    w2v_path = paths.models_dir / 'w2v.model'
    clf_path = paths.models_dir / 'w2v_lr.joblib'
    w2v.save(str(w2v_path))
    joblib.dump(clf, clf_path)

    save_json(
        {
            'model': 'Word2Vec(char) + MeanPooling + LogisticRegression',
            'macro_f1': float(macro_f1),
            'n_train': int(len(train_df)),
            'n_test': int(len(test_df)),
            'w2v': {'vector_size': 100, 'window': 5, 'min_count': 2, 'epochs': 10, 'sg': 1},
        },
        paths.results_dir / 'w2v_lr_metrics.json',
    )
    (paths.results_dir / 'w2v_lr_report.txt').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()