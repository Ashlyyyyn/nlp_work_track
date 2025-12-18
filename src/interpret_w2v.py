from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from gensim.models import Word2Vec

from common import Paths, load_dataset, train_test_split_stratified


def char_tokenize(text: str) -> List[str]:
    text = text.strip()
    return [ch for ch in text if not ch.isspace()]


def build_class_centroids(w2v: Word2Vec, df, top_docs_per_class: int = 200) -> Dict[str, np.ndarray]:
    centroids: Dict[str, List[np.ndarray]] = {}
    for label, g in df.groupby('label'):
        vecs = []
        for t in g['text'].tolist()[:top_docs_per_class]:
            toks = char_tokenize(t)
            toks = [x for x in toks if x in w2v.wv]
            if toks:
                vecs.append(np.mean([w2v.wv[x] for x in toks], axis=0))
        if vecs:
            centroids[label] = np.mean(np.stack(vecs, axis=0), axis=0)
    return {k: v.astype(np.float32) for k, v in centroids.items()}


def nearest_tokens(w2v: Word2Vec, vec: np.ndarray, topn: int = 20):
    return w2v.wv.similar_by_vector(vec, topn=topn)


def main() -> None:
    paths = Paths()
    paths.ensure()

    w2v_path = paths.models_dir / 'w2v.model'
    if not w2v_path.exists():
        raise FileNotFoundError('Run train_w2v_lr.py first.')

    w2v = Word2Vec.load(str(w2v_path))
    df = load_dataset(paths.data_csv)
    train_df, _ = train_test_split_stratified(df, test_size=0.2, seed=42)

    # 1) 近邻示例：你可以把 seed_words 换成你任务里代表性的词/字
    seed_words = ['的', '了', '股', '学', '他']
    lines = ['# Word2Vec interpretability\n', '## Nearest neighbors (seed tokens)\n']
    for w in seed_words:
        if w in w2v.wv:
            nn = w2v.wv.most_similar(w, topn=10)
            lines.append(f"- **{w}**: " + ', '.join([f"{x}({s:.2f})" for x, s in nn]))
        else:
            lines.append(f"- **{w}**: [OOV]")
    lines.append('\n## Class prototype neighbors (centroid -> nearest tokens)\n')

    # 2) 类别原型：按类别聚合文档向量，取质心，再找最相似 token
    centroids = build_class_centroids(w2v, train_df, top_docs_per_class=200)
    for label, cvec in centroids.items():
        nn = nearest_tokens(w2v, cvec, topn=15)
        lines.append(f"\n### {label}")
        lines.append(', '.join([f"{x}({s:.2f})" for x, s in nn]))

    out_path = paths.results_dir / 'w2v_interpretability.md'
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()