from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchifyResult:
    image: np.ndarray
    patches: np.ndarray
    tokens: np.ndarray


def patchify(image: np.ndarray, patch_size: int) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("image must be HxWxC")
    h, w, c = image.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("H and W must be divisible by patch_size")

    grid_h = h // patch_size
    grid_w = w // patch_size
    patches = (
        image.reshape(grid_h, patch_size, grid_w, patch_size, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_h * grid_w, patch_size * patch_size * c)
    )
    return patches


def linear_project(x: np.ndarray, out_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((x.shape[1], out_dim), dtype=np.float64) / np.sqrt(x.shape[1])
    return x @ w


def demo(h: int = 224, w: int = 224, c: int = 3, patch: int = 16, d_model: int = 64) -> PatchifyResult:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
    patches = patchify(image.astype(np.float64) / 255.0, patch_size=patch)
    tokens = linear_project(patches, out_dim=d_model)
    return PatchifyResult(image=image, patches=patches, tokens=tokens)


def main() -> None:
    res = demo()
    n_patches, patch_dim = res.patches.shape
    print(f"image: {tuple(res.image.shape)}")
    print(f"patches: {n_patches} x {patch_dim}")
    print(f"tokens: {tuple(res.tokens.shape)}  (patches projected to d_model)")


if __name__ == "__main__":
    main()

