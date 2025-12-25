from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from huffman_tree import build_codes, build_huffman_tree  # noqa: E402


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class HierSoftmax:
    in_dim: int
    codes: dict[str, list[int]]
    paths: dict[str, list[int]]
    w_nodes: np.ndarray  # [n_nodes, in_dim]

    def prob(self, h: np.ndarray, token: str) -> float:
        code = self.codes[token]
        path = self.paths[token]
        p = 1.0
        for bit, node_id in zip(code, path):
            logit = float(self.w_nodes[node_id] @ h)
            p_bit = float(sigmoid(np.array(logit)))
            p *= p_bit if bit == 1 else (1.0 - p_bit)
        return float(p)


def build_node_index(root) -> tuple[dict[object, int], list[object]]:
    nodes: list[object] = []
    idx: dict[object, int] = {}

    stack = [root]
    while stack:
        n = stack.pop()
        if n in idx:
            continue
        idx[n] = len(nodes)
        nodes.append(n)
        if getattr(n, "left", None) is not None:
            stack.append(n.left)
        if getattr(n, "right", None) is not None:
            stack.append(n.right)
    return idx, nodes


def build_paths(root, codes: dict[str, list[int]]) -> dict[str, list[int]]:
    node_to_id, _ = build_node_index(root)
    paths: dict[str, list[int]] = {}

    def walk(node, token: str, acc: list[int]) -> bool:
        if node.symbol == token:
            paths[token] = acc.copy()
            return True
        if node.left is not None:
            acc.append(node_to_id[node.left])
            if walk(node.left, token, acc):
                return True
            acc.pop()
        if node.right is not None:
            acc.append(node_to_id[node.right])
            if walk(node.right, token, acc):
                return True
            acc.pop()
        return False

    for tok in codes.keys():
        walk(root, tok, [])
    return paths


def init_hsoftmax(vocab: dict[str, float], in_dim: int = 16, seed: int = 0) -> HierSoftmax:
    root = build_huffman_tree(vocab)
    codes = build_codes(root)
    paths = build_paths(root, codes)

    rng = np.random.default_rng(seed)
    node_to_id, nodes = build_node_index(root)
    n_nodes = len(nodes)
    w_nodes = rng.standard_normal((n_nodes, in_dim), dtype=np.float64) * 0.01
    return HierSoftmax(in_dim=in_dim, codes=codes, paths=paths, w_nodes=w_nodes)


def main() -> None:
    vocab = {"a": 10, "b": 8, "c": 6, "d": 4, "e": 2}
    hs = init_hsoftmax(vocab, in_dim=8)
    h = np.ones((hs.in_dim,), dtype=np.float64)

    for tok in sorted(vocab.keys()):
        print(tok, "code:", "".join(map(str, hs.codes[tok])), "p(h,tok):", f"{hs.prob(h, tok):.6f}")


if __name__ == "__main__":
    main()
