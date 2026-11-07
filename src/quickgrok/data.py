"""Modular arithmetic dataset from "Grokking" (Power et al., 2022), plus composition in S5.

Every pair (a, b) with a, b in [0, p) gives one example `a op b = c (mod p)`.
Inputs are token sequences `[a, b, EQ]` with `EQ = p`, so the vocab size is p + 1.
For "s5" the symbols are the 120 permutations of 5 elements, so p must be 120 (`S5_ORDER`).
"""

import random
from collections.abc import Callable
from itertools import permutations
from typing import Literal

Operation = Literal["add", "sub", "mul", "div", "s5"]
Example = tuple[tuple[int, int, int], int]

# Symbol i of "s5" is S5[i]: the permutation x -> S5[i][x] of 0..4, in lexicographic order.
S5 = list(permutations(range(5)))
S5_ORDER = len(S5)
S5_INDEX = {perm: i for i, perm in enumerate(S5)}


def compose_s5(a: int, b: int) -> int:
    """Index of a∘b, the permutation x -> a(b(x)) (apply b first, then a)."""
    return S5_INDEX[tuple(S5[a][S5[b][x]] for x in range(5))]


OPS: dict[Operation, Callable[[int, int, int], int]] = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    "div": lambda a, b, p: (a * pow(b, -1, p)) % p,
    "s5": lambda a, b, p: compose_s5(a, b),
}


def make_dataset(
    p: int = 97,
    op: Operation = "add",
    train_frac: float = 0.3,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    """Build all `a op b mod p` examples and split them into train and validation.

    `p` should be prime (needed for "div"). For "div", b = 0 is skipped. For "s5", `p` is the
    number of symbols and must be `S5_ORDER` (raises ValueError otherwise).
    Returns (train, val), each a list of ((a, b, EQ), c) with EQ = p.
    """
    if op == "s5" and p != S5_ORDER:
        raise ValueError(f"op s5 needs p = {S5_ORDER}, got {p}")
    examples = [
        ((a, b, p), OPS[op](a, b, p))
        for a in range(p)
        for b in range(p)
        if not (op == "div" and b == 0)
    ]
    random.Random(seed).shuffle(examples)
    n_train = int(train_frac * len(examples))
    return examples[:n_train], examples[n_train:]


if __name__ == "__main__":
    train, val = make_dataset(p=97, op="div", train_frac=0.5)
    assert len(train) + len(val) == 97 * 96
    assert not {x for x, _ in train} & {x for x, _ in val}
    for (a, b, eq), c in train + val:
        assert eq == 97 and (c * b) % 97 == a
    train, _ = make_dataset(p=113, op="add")
    assert all((a + b) % 113 == c for (a, b, _), c in train)
    print(f"ok: {len(train)} train examples, first: {train[0]}")
