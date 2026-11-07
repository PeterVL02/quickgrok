"""Modular arithmetic dataset from "Grokking" (Power et al., 2022).

Every pair (a, b) with a, b in [0, p) gives one example `a op b = c (mod p)`.
Inputs are token sequences `[a, b, EQ]` with `EQ = p`, so the vocab size is p + 1.
"""

import random
from collections.abc import Callable
from typing import Literal

Operation = Literal["add", "sub", "mul", "div"]
Example = tuple[tuple[int, int, int], int]

OPS: dict[Operation, Callable[[int, int, int], int]] = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    "div": lambda a, b, p: (a * pow(b, -1, p)) % p,
}


def make_dataset(
    p: int = 97,
    op: Operation = "add",
    train_frac: float = 0.3,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    """Build all `a op b mod p` examples and split them into train and validation.

    `p` should be prime (needed for "div"). For "div", b = 0 is skipped.
    Returns (train, val), each a list of ((a, b, EQ), c) with EQ = p.
    """
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
