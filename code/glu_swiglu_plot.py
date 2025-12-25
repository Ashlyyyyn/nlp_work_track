from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


@dataclass(frozen=True)
class Curves:
    x: np.ndarray
    glu_gate: np.ndarray
    swiglu_gate: np.ndarray


def compute_curves(n: int = 1000, x_min: float = -6.0, x_max: float = 6.0) -> Curves:
    x = np.linspace(x_min, x_max, n, dtype=np.float64)
    return Curves(x=x, glu_gate=sigmoid(x), swiglu_gate=silu(x))


def write_csv(curves: Curves, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "sigmoid(x)", "silu(x)"])
        for xi, si, sli in zip(curves.x, curves.glu_gate, curves.swiglu_gate):
            w.writerow([float(xi), float(si), float(sli)])


def try_plot(curves: Curves, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(curves.x, curves.glu_gate, label="GLU gate: sigmoid(x)")
    plt.plot(curves.x, curves.swiglu_gate, label="SwiGLU gate: silu(x)")
    plt.axhline(0.0, linewidth=0.8, color="black", alpha=0.3)
    plt.axvline(0.0, linewidth=0.8, color="black", alpha=0.3)
    plt.title("GLU vs SwiGLU gates (1D)")
    plt.xlabel("x")
    plt.ylabel("gate(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


def main() -> None:
    curves = compute_curves()
    img_path = Path("results") / "glu_swiglu_gates.png"
    csv_path = Path("results") / "glu_swiglu_gates.csv"

    if try_plot(curves, img_path):
        print(f"Wrote: {img_path}")
        return

    write_csv(curves, csv_path)
    print("matplotlib not available; wrote CSV instead.")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
