from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from common import Paths


def top_features_per_class(model_path: Path, top_k: int = 30) -> str:
    pipe = joblib.load(model_path)
    vec = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']

    feature_names = np.array(vec.get_feature_names_out())
    classes = clf.classes_
    coefs = clf.coef_  # shape: (n_classes, n_features) for multinomial

    lines = []
    for i, c in enumerate(classes):
        w = coefs[i]
        top_idx = np.argsort(w)[-top_k:][::-1]
        lines.append(f"\n## Class: {c}")
        for rank, j in enumerate(top_idx, start=1):
            lines.append(f"{rank:02d}. {feature_names[j]}  ({w[j]:.4f})")
    return '\n'.join(lines)


def main() -> None:
    paths = Paths()
    paths.ensure()

    model_path = paths.models_dir / 'tfidf_lr.joblib'
    if not model_path.exists():
        raise FileNotFoundError('Run train_tfidf_lr.py first.')

    out = top_features_per_class(model_path, top_k=30)
    out_path = paths.results_dir / 'tfidf_top_features.md'
    out_path.write_text(out, encoding='utf-8')
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()