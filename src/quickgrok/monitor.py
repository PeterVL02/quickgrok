"""Train-only grokking monitors and cheap baselines (reports/preregistration_monitor.md).

`MonitorHook` logs one `MonitorRecord` every `every` steps, from hidden layer 0 of `GrokMLP`
and the training rows only (test inputs and labels are never touched):
  m1    fraction of neurons with harmonic share >= 0.5 and check R^2 >= 0.8 (cyclic add only)
  m2    median self-expression skill of the input columns, (capture - null) / (1 - null)
  norm  total parameter norm of the effective weights (B3)
  b4    fraction of neurons whose input columns have Fourier captured energy >= 0.8 (add only)
  b5    excluded train loss: logits without their key-frequency components (add only)
"""

import json
import math
import time
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quickgrok.accel import (
    ProbeHook,
    captured_energy,
    effective_state_dict,
    make_basis,
    self_expr_basis,
)
from quickgrok.models import GrokMLP

M1_SHARE, M1_R2 = 0.5, 0.8
B4_ENERGY = 0.8
OMP_Q = 8  # other columns per self-expression fit, as in scripts/basis_capture.py
N_KEY_FREQS = 5
N_HARMONICS = 4  # M1's harmonic share and B4's basis (1 + 2 * 4 columns)


class MonitorRecord(TypedDict):
    step: int
    m1: float | None
    m2: float
    norm: float
    b4: float | None
    b5: float | None
    key_freqs: list[int] | None  # B5's key frequencies at this step
    seconds: float  # wall-clock cost of this call


def self_expr_capture(W: Tensor, q: int = OMP_Q) -> Tensor:
    """Capture (d,) of each mean-removed column of W (n, d) by q other columns (greedy OMP)."""
    basis = self_expr_basis(W, list(range(W.shape[1])), q)
    return captured_energy((basis, basis), W.T, W.T)


def self_expr_null(n: int, d: int, seed: int = 0) -> float:
    """Median self-expression capture over the columns of Gaussian U, V of shape (n, d)."""
    g = torch.Generator().manual_seed(seed)
    U, V = (torch.randn(n, d, generator=g, dtype=torch.float64) for _ in "uv")
    return float(torch.cat([self_expr_capture(U), self_expr_capture(V)]).nanmedian())


def key_freqs(U: Tensor, V: Tensor, n_keys: int = N_KEY_FREQS) -> list[int]:
    """The `n_keys` frequencies 1..n//2 with the most DFT energy in the columns of U, V (n, d)."""
    power = sum(
        (torch.fft.rfft(W - W.mean(0), dim=0).abs() ** 2).sum(1) for W in (U, V)
    )
    assert isinstance(power, Tensor)
    return (power[1 : len(U) // 2 + 1].topk(n_keys).indices + 1).tolist()


def excluded_loss(logits: Tensor, y: Tensor, freqs: list[int]) -> float:
    """Cross-entropy after removing each row's cos/sin(2 pi k c / p) components, k in `freqs`.

    `logits` (N, p) over the answers c = 0..p-1. The cos and sin columns are orthogonal on
    the full answer grid with squared norm p / 2, so the projection is basis @ basis.T * 2 / p.
    """
    p = logits.shape[1]
    c = torch.arange(p, device=logits.device, dtype=logits.dtype)
    w = (
        2
        * math.pi
        * torch.tensor(freqs, device=c.device, dtype=c.dtype)[:, None]
        * c
        / p
    )
    basis = torch.cat([torch.cos(w), torch.sin(w)]).T  # (p, 2 * len(freqs))
    kept = logits - logits @ basis @ basis.T * (2 / p)
    return F.cross_entropy(kept, y).item()


def fire_step(
    steps: list[int], values: list[float | None], threshold: float, below: bool = False
) -> int | None:
    """First step whose value is >= `threshold` (<= with `below`), or None if never."""
    return next(
        (
            s
            for s, v in zip(steps, values)
            if v is not None and (v <= threshold if below else v >= threshold)
        ),
        None,
    )


def log_mae(forecasts: list[float], actual: list[float]) -> float:
    """Mean absolute error of the natural logs: mean |ln forecast - ln actual|."""
    errors = [abs(math.log(f) - math.log(a)) for f, a in zip(forecasts, actual)]
    return sum(errors) / len(errors)


class MonitorHook(ProbeHook):
    """Every `every` steps, log a `MonitorRecord` to `out_path` (JSON lines, truncated here).

    Runs in `after_backward`, i.e. on the weights before the optimizer step, so a record at
    step s describes the same weights as train.py's metrics at step s. `cyclic` (the task is
    a + b mod p) turns on m1, b4 and b5. M1 uses `ProbeHook`'s 80/20 fit/check split of the
    training rows; b5 uses the training targets `y_train` as the model sees them.
    """

    def __init__(
        self,
        x_train: Tensor,
        y_train: Tensor,
        p: int,
        out_path: Path,
        every: int,
        seed: int,
        cyclic: bool,
    ) -> None:
        super().__init__(x_train, p, out_path, 0, every, N_HARMONICS, seed)
        self.y_train = y_train
        self.cyclic = cyclic
        self.seed = seed
        self.null: float | None = None  # Gaussian null of M2, for the model's shape

    def measure(self, step: int, model: GrokMLP) -> MonitorRecord:
        """Compute every score on the current weights."""
        start = time.perf_counter()
        U, V = (
            e.weight.detach().cpu().double() for e in (model.embed_a, model.embed_b)
        )
        if self.null is None:
            self.null = self_expr_null(*U.shape)
        caps = torch.cat([self_expr_capture(U), self_expr_capture(V)])
        m2 = float(((caps - self.null) / (1 - self.null)).nanmedian())
        weights = effective_state_dict(model).values()
        norm = torch.cat([w.flatten() for w in weights]).norm().item()
        m1 = b4 = b5 = None
        freqs = None
        if self.cyclic:
            record = self.probe(step, model)
            m1 = sum(
                s is not None and r is not None and s >= M1_SHARE and r >= M1_R2
                for s, r in zip(record["harmonic_share"], record["r2_check"])
            ) / len(record["harmonic_share"])
            neurons = list(range(U.shape[1]))
            bases, _, _ = make_basis("fourier", U, V, neurons, 1 + 2 * N_HARMONICS, 0)
            b4 = (captured_energy(bases, U.T, V.T) >= B4_ENERGY).double().mean().item()
            freqs = key_freqs(U, V)
            with torch.no_grad():
                b5 = excluded_loss(model(self.x_train), self.y_train, freqs)
        return {
            "step": step,
            "m1": m1,
            "m2": m2,
            "norm": norm,
            "b4": b4,
            "b5": b5,
            "key_freqs": freqs,
            "seconds": time.perf_counter() - start,
        }

    def after_backward(self, step: int, model: nn.Module) -> None:
        """Every `every` steps, measure and append a JSON line."""
        if step % self.every:
            return
        assert isinstance(model, GrokMLP), "the monitor needs GrokMLP"
        with self.out_path.open("a") as f:
            f.write(json.dumps(self.measure(step, model)) + "\n")

    def after_step(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer | None:
        """No-op: the monitor runs before the step, in `after_backward`."""
        return None
