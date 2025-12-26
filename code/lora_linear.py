from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Linear:
    w: np.ndarray  # [in, out]
    b: np.ndarray  # [out]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b


@dataclass
class LoRALinear:
    base: Linear
    a: np.ndarray  # [in, r]
    b: np.ndarray  # [r, out]
    alpha: float

    @property
    def r(self) -> int:
        return int(self.a.shape[1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        scale = self.alpha / max(self.r, 1)
        return self.base(x) + (x @ self.a @ self.b) * scale


def init_linear(in_dim: int, out_dim: int, seed: int) -> Linear:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((in_dim, out_dim), dtype=np.float64) / np.sqrt(in_dim)
    b = np.zeros((out_dim,), dtype=np.float64)
    return Linear(w=w, b=b)


def init_lora(base: Linear, r: int = 4, alpha: float = 8.0, seed: int = 1) -> LoRALinear:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((base.w.shape[0], r), dtype=np.float64) * 0.01
    b = rng.standard_normal((r, base.w.shape[1]), dtype=np.float64) * 0.01
    return LoRALinear(base=base, a=a, b=b, alpha=alpha)


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, 16), dtype=np.float64)

    base = init_linear(16, 8, seed=0)
    lora = init_lora(base, r=4, alpha=8.0, seed=123)

    y_base = base(x)
    y_lora = lora(x)
    print("base output shape:", y_base.shape)
    print("lora output shape:", y_lora.shape)
    print("mean |delta|:", float(np.mean(np.abs(y_lora - y_base))))


if __name__ == "__main__":
    main()

