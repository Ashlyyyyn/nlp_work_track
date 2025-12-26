from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


@dataclass
class SimpleRNNLM:
    vocab: list[str]
    char_to_id: dict[str, int]
    w_xh: np.ndarray
    w_hh: np.ndarray
    b_h: np.ndarray
    w_hy: np.ndarray
    b_y: np.ndarray

    @property
    def hidden_dim(self) -> int:
        return int(self.w_hh.shape[0])

    def step(self, x_id: int, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(self.vocab),), dtype=np.float64)
        x[x_id] = 1.0
        h_new = np.tanh(self.w_xh.T @ x + self.w_hh @ h + self.b_h)
        y = self.w_hy @ h_new + self.b_y
        p = softmax(y)
        return h_new, p


def make_tiny_lm(text: str, hidden_dim: int = 32, seed: int = 0) -> SimpleRNNLM:
    vocab = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(vocab)}
    v = len(vocab)
    rng = np.random.default_rng(seed)
    w_xh = rng.standard_normal((v, hidden_dim), dtype=np.float64) / np.sqrt(v)
    w_hh = rng.standard_normal((hidden_dim, hidden_dim), dtype=np.float64) / np.sqrt(hidden_dim)
    b_h = np.zeros((hidden_dim,), dtype=np.float64)
    w_hy = rng.standard_normal((v, hidden_dim), dtype=np.float64) / np.sqrt(hidden_dim)
    b_y = np.zeros((v,), dtype=np.float64)
    return SimpleRNNLM(vocab=vocab, char_to_id=char_to_id, w_xh=w_xh, w_hh=w_hh, b_h=b_h, w_hy=w_hy, b_y=b_y)


def sample(lm: SimpleRNNLM, start: str, length: int = 80, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    h = np.zeros((lm.hidden_dim,), dtype=np.float64)
    out = [start]
    x_id = lm.char_to_id[start]
    for _ in range(length - 1):
        h, p = lm.step(x_id, h)
        x_id = int(rng.choice(len(lm.vocab), p=p))
        out.append(lm.vocab[x_id])
    return "".join(out)


def main() -> None:
    text = "hello rnn language model\n"
    lm = make_tiny_lm(text, hidden_dim=32)
    print("vocab:", lm.vocab)
    print("sample:", sample(lm, start="h", length=60))
    print("Note: this demo samples from random weights; add training for a real LM.")


if __name__ == "__main__":
    main()

