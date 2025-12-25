from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PPOBatch:
    obs: np.ndarray
    act: np.ndarray
    old_logp: np.ndarray
    adv: np.ndarray
    ret: np.ndarray


def ppo_clip_objective(logp: np.ndarray, old_logp: np.ndarray, adv: np.ndarray, clip_eps: float = 0.2) -> float:
    ratio = np.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return float(np.mean(np.minimum(unclipped, clipped)))


def demo() -> None:
    rng = np.random.default_rng(0)
    batch = PPOBatch(
        obs=rng.standard_normal((32, 8), dtype=np.float64),
        act=rng.integers(0, 4, size=(32,), dtype=np.int64),
        old_logp=rng.standard_normal((32,), dtype=np.float64) * 0.1,
        adv=rng.standard_normal((32,), dtype=np.float64),
        ret=rng.standard_normal((32,), dtype=np.float64),
    )
    logp = batch.old_logp + rng.standard_normal((32,), dtype=np.float64) * 0.05
    obj = ppo_clip_objective(logp, batch.old_logp, batch.adv)
    print(f"PPO clipped objective (demo): {obj:.4f}")
    print("Note: this is a runnable skeleton; plug in policy/value networks to train.")


if __name__ == "__main__":
    demo()

