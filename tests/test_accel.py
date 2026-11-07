"""Check the probe split, the harmonic share and the hook API. Run: uv run python tests/test_accel.py"""

import json
import math
import tempfile
from pathlib import Path

import torch
from torch import nn

from quickgrok.accel import Accelerator, Hook, ProbeHook
from quickgrok.data import make_dataset
from quickgrok.models import build_model
from quickgrok.regress import fit_neurons, harmonic_share, make_grid
from quickgrok.train import to_tensors


def test_probe_split() -> None:
    """Fit and check rows partition the training rows and contain no test pair; probe runs."""
    p = 97
    train, val = make_dataset(p, "add", 0.3, seed=100)
    x_train, _ = to_tensors(train, "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.jsonl"
        hook = ProbeHook(x_train, p, path, every=2, seed=100)
        pairs = [tuple(r) for r in x_train[:, :2].tolist()]
        fit = {q for q, m in zip(pairs, hook.fit_rows.tolist()) if m}
        check = {q for q, m in zip(pairs, hook.fit_rows.tolist()) if not m}
        assert not fit & check and fit | check == {(a, b) for (a, b, _), _ in train}
        assert not (fit | check) & {(a, b) for (a, b, _), _ in val}
        assert abs(len(fit) / len(train) - 0.8) < 0.01

        model = build_model("mlp", p)
        opt = torch.optim.SGD(model.parameters())
        accel = Accelerator(model, [hook])
        for step in range(3):
            accel.after_step(step, opt)
        records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in records] == [0, 2]
    assert all(len(r["harmonic_share"]) == len(r["r2_check"]) == 512 for r in records)


def test_harmonic_share() -> None:
    """A square wave in k*a mod p is ~fully explained by harmonics of k, not by k alone."""
    p, k = 97, 7
    a, b = make_grid(p)
    wave = torch.sign(torch.cos(2 * math.pi * k * a.double() / p))
    fit = fit_neurons(wave[:, None], a, b, p, "trig", 0, [0])[0]
    base, share = harmonic_share(torch.tensor(fit["freq_shares"]), p)
    assert fit["top_freq"] == k and base.item() == k
    # Ideal square wave: harmonics k and 3k hold (1 + 1/9) / (pi^2 / 8) = 0.90 of the variance.
    assert share.item() > 0.89 and fit["top_freq_share"] < 0.82
    assert harmonic_share(torch.tensor(fit["freq_shares"]), p, 20)[1] > 0.98


def test_optimizer_replacement() -> None:
    """A hook returning an optimizer replaces it; a no-op hook keeps it."""
    model = nn.Linear(2, 2)
    old, new = torch.optim.SGD(model.parameters()), torch.optim.Adam(model.parameters())

    class SwapHook(Hook):
        def after_step(
            self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
        ) -> torch.optim.Optimizer | None:
            """Swap in `new` at step 1."""
            return new if step == 1 else None

    accel = Accelerator(model, [Hook(), SwapHook(), Hook()])
    assert accel.after_step(0, old) is old
    assert accel.after_step(1, old) is new


if __name__ == "__main__":
    test_probe_split()
    test_harmonic_share()
    test_optimizer_replacement()
    print("ok")
