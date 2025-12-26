from __future__ import annotations

import numpy as np


def rnn_forward(
    x: np.ndarray,
    h0: np.ndarray,
    wxh: np.ndarray,
    whh: np.ndarray,
    why: np.ndarray,
    bh: np.ndarray,
    by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple RNN forward pass.

    Shapes:
        x: (batch, time, input_dim)
        h0: (batch, hidden_dim)
        wxh: (input_dim, hidden_dim)
        whh: (hidden_dim, hidden_dim)
        why: (hidden_dim, output_dim)
        bh: (hidden_dim,)
        by: (output_dim,)
    Returns:
        h: (batch, time, hidden_dim)
        y: (batch, time, output_dim)
    """
    batch, time_steps, _ = x.shape
    hidden_dim = h0.shape[1]
    output_dim = by.shape[0]

    h = np.zeros((batch, time_steps, hidden_dim), dtype=x.dtype)
    y = np.zeros((batch, time_steps, output_dim), dtype=x.dtype)

    h_t = h0
    for t in range(time_steps):
        x_t = x[:, t, :]
        h_t = np.tanh(x_t @ wxh + h_t @ whh + bh)
        y[:, t, :] = h_t @ why + by
        h[:, t, :] = h_t

    return h, y


def demo() -> None:
    rng = np.random.default_rng(0)
    batch, time_steps, input_dim = 2, 4, 3
    hidden_dim, output_dim = 5, 2

    x = rng.normal(size=(batch, time_steps, input_dim)).astype(np.float32)
    h0 = np.zeros((batch, hidden_dim), dtype=np.float32)
    wxh = rng.normal(size=(input_dim, hidden_dim)).astype(np.float32)
    whh = rng.normal(size=(hidden_dim, hidden_dim)).astype(np.float32)
    why = rng.normal(size=(hidden_dim, output_dim)).astype(np.float32)
    bh = np.zeros((hidden_dim,), dtype=np.float32)
    by = np.zeros((output_dim,), dtype=np.float32)

    h, y = rnn_forward(x, h0, wxh, whh, why, bh, by)
    print("h shape:", h.shape)
    print("y shape:", y.shape)
    print("y[0, -1]:", y[0, -1])


if __name__ == "__main__":
    demo()
