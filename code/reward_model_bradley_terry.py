from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bradley_terry_loss(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> float:
    if reward_chosen.shape != reward_rejected.shape:
        raise ValueError("reward_chosen and reward_rejected must have same shape")
    diff = reward_chosen - reward_rejected
    p = sigmoid(diff)
    return float(-np.mean(np.log(p + 1e-12)))


def main() -> None:
    rng = np.random.default_rng(0)
    reward_chosen = rng.standard_normal((16,), dtype=np.float64)
    reward_rejected = rng.standard_normal((16,), dtype=np.float64)
    loss = bradley_terry_loss(reward_chosen, reward_rejected)
    print(f"Bradley-Terry preference loss: {loss:.4f}")


if __name__ == "__main__":
    main()

