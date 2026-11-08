"""Check the modular-arithmetic dataset split and labels."""

from quickgrok.data import make_dataset


def test_make_dataset() -> None:
    """Splits partition all pairs without overlap and every label solves its equation."""
    train, val = make_dataset(p=97, op="div", train_frac=0.5)
    assert len(train) + len(val) == 97 * 96
    assert not {x for x, _ in train} & {x for x, _ in val}
    for (a, b, eq), c in train + val:
        assert eq == 97 and (c * b) % 97 == a
    train, _ = make_dataset(p=113, op="add")
    assert all((a + b) % 113 == c for (a, b, _), c in train)
