from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    root: Path = Path('.')
    data_csv: Path = Path('data/dataset.csv')
    results_dir: Path = Path('results')
    models_dir: Path = Path('models')

    def ensure(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def load_dataset(csv_path: str | os.PathLike) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset file: {p}. Please create data/dataset.csv")
    df = pd.read_csv(p)
    for col in ['text', 'label']:
        if col not in df.columns:
            raise ValueError(f"dataset.csv must contain columns: text,label. Missing: {col}")
    df = df.dropna(subset=['text', 'label']).copy()
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    return df


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 简化版分层切分：按 label 分组后抽样
    rng = np.random.default_rng(seed)
    test_idx: List[int] = []
    for _, g in df.groupby('label'):
        idx = g.index.to_numpy()
        n_test = max(1, int(round(len(idx) * test_size)))
        pick = rng.choice(idx, size=n_test, replace=False)
        test_idx.extend(pick.tolist())
    test_idx = sorted(set(test_idx))
    test_df = df.loc[test_idx].reset_index(drop=True)
    train_df = df.drop(index=test_idx).reset_index(drop=True)
    return train_df, test_df


def save_json(obj: Dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)