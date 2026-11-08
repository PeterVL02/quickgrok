"""Check the random-label control, the firing rule, log-MAE and the excluded loss."""

import math

import torch

from quickgrok.monitor import excluded_loss, fire_step, log_mae
from quickgrok.train import make_tensors


def test_random_labels() -> None:
    """Training targets are a reproducible permutation; inputs and test labels are untouched."""
    plain = make_tensors(97, "add", 0.3, 3, "cpu")
    shuffled = make_tensors(97, "add", 0.3, 3, "cpu", random_labels=True)
    again = make_tensors(97, "add", 0.3, 3, "cpu", random_labels=True)
    x_tr, y_tr, x_val, y_val = plain
    assert torch.equal(shuffled[0], x_tr)
    assert torch.equal(shuffled[2], x_val) and torch.equal(shuffled[3], y_val)
    assert torch.equal(shuffled[1].sort().values, y_tr.sort().values)
    assert (shuffled[1] != y_tr).float().mean() > 0.9
    assert all(torch.equal(a, b) for a, b in zip(shuffled, again))
    other = make_tensors(97, "add", 0.3, 4, "cpu", random_labels=True)[1]
    assert not torch.equal(other, make_tensors(97, "add", 0.3, 4, "cpu")[1])


def test_fire_step() -> None:
    """The first probe step meeting the threshold, inclusive; None when it never does."""
    steps = [0, 50, 100, 150, 200]
    assert fire_step(steps, [0.0, 0.05, 0.1, 0.3, 0.0], 0.1) == 100
    assert fire_step(steps, [None, 0.2, 0.3, 0.3, 0.0], 0.1) == 50
    assert fire_step(steps, [0.0, 0.05, 0.09, 0.0, 0.0], 0.1) is None
    assert fire_step(steps, [300.0, 200.0, 99.0, 50.0, 40.0], 99.0, below=True) == 100


def test_log_mae() -> None:
    """Hand-made: errors ln 2, ln 2 and 0 average to (2/3) ln 2."""
    value = log_mae([2000.0, 500.0, 300.0], [1000.0, 1000.0, 300.0])
    assert math.isclose(value, 2 / 3 * math.log(2))


def test_excluded_loss() -> None:
    """Logits made only of a key frequency become uniform (loss ln p); other freqs survive."""
    p, k = 97, 7
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    c = torch.arange(p)
    logits = 10 * torch.cos(2 * math.pi * k * ((a + b)[:, None] - c).double() / p)
    y = (a + b) % p
    assert math.isclose(excluded_loss(logits, y, [k, 3]), math.log(p), rel_tol=1e-9)
    plain = torch.nn.functional.cross_entropy(logits, y).item()
    assert math.isclose(excluded_loss(logits, y, [3, 5]), plain, rel_tol=1e-9)
