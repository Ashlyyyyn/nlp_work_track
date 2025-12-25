from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def clip_contrastive_loss(image_emb: np.ndarray, text_emb: np.ndarray, temperature: float = 0.07) -> float:
    if image_emb.shape != text_emb.shape:
        raise ValueError("image_emb and text_emb must have same shape [B, D]")

    image_emb = l2_normalize(image_emb)
    text_emb = l2_normalize(text_emb)

    logits = (image_emb @ text_emb.T) / temperature  # [B, B]
    probs_i2t = softmax(logits, axis=1)
    probs_t2i = softmax(logits.T, axis=1)

    b = image_emb.shape[0]
    target = np.arange(b)
    loss_i2t = -np.mean(np.log(probs_i2t[np.arange(b), target] + 1e-12))
    loss_t2i = -np.mean(np.log(probs_t2i[np.arange(b), target] + 1e-12))
    return float((loss_i2t + loss_t2i) / 2.0)


def main() -> None:
    rng = np.random.default_rng(0)
    b, d = 8, 64
    image_emb = rng.standard_normal((b, d), dtype=np.float64)
    text_emb = rng.standard_normal((b, d), dtype=np.float64)

    loss = clip_contrastive_loss(image_emb, text_emb)
    print(f"CLIP contrastive loss (random embeddings): {loss:.4f}")


if __name__ == "__main__":
    main()

