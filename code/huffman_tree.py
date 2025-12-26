from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Optional
from typing import Iterable


@dataclass(frozen=True)
class HuffmanNode:
    weight: float
    symbol: Optional[str] = None
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None


def build_huffman_tree(freqs: dict[str, float]) -> HuffmanNode:
    heap: list[tuple[float, int, HuffmanNode]] = []
    uid = 0
    for sym, w in freqs.items():
        heappush(heap, (float(w), uid, HuffmanNode(weight=float(w), symbol=sym)))
        uid += 1

    if not heap:
        raise ValueError("empty freqs")

    while len(heap) > 1:
        w1, _, n1 = heappop(heap)
        w2, _, n2 = heappop(heap)
        parent = HuffmanNode(weight=w1 + w2, left=n1, right=n2)
        heappush(heap, (parent.weight, uid, parent))
        uid += 1
    return heap[0][2]


def build_codes(root: HuffmanNode) -> dict[str, list[int]]:
    codes: dict[str, list[int]] = {}

    def dfs(node: HuffmanNode, path: list[int]) -> None:
        if node.symbol is not None:
            codes[node.symbol] = path.copy()
            return
        if node.left is not None:
            path.append(0)
            dfs(node.left, path)
            path.pop()
        if node.right is not None:
            path.append(1)
            dfs(node.right, path)
            path.pop()

    dfs(root, [])
    return codes


def demo_symbols(symbols: Iterable[str]) -> dict[str, float]:
    freqs: dict[str, float] = {}
    for s in symbols:
        freqs[s] = freqs.get(s, 0.0) + 1.0
    return freqs


def main() -> None:
    text = "hierarchical softmax uses a huffman tree"
    freqs = demo_symbols([c for c in text if c != " "])
    root = build_huffman_tree(freqs)
    codes = build_codes(root)
    for sym in sorted(codes.keys())[:12]:
        print(sym, "".join(map(str, codes[sym])))
    print(f"symbols: {len(codes)}; tree weight: {root.weight:.1f}")


if __name__ == "__main__":
    main()
