"""Training hooks for grokking accelerators, usable from any PyTorch training loop.

Call `accel.after_backward(step)` between `loss.backward()` and `opt.step()`, and
`opt = accel.after_step(step, opt)` after `opt.step()`.
"""

import json
import math
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor, nn

from quickgrok.regress import (
    build_features,
    fit_neurons,
    harmonic_share,
    record_activations,
)
from quickgrok.train import GrokMLP, GrokTransformer


class Hook:
    def after_backward(self, step: int, model: nn.Module) -> None:
        """Called after gradients are computed and before the optimizer step; may edit gradients."""

    def after_step(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer | None:
        """Called after the optimizer step; may edit weights or modules.

        Return a replacement optimizer (e.g. after adding parameters), or None to keep it.
        """
        return None


class Accelerator:
    def __init__(self, model: nn.Module, hooks: list[Hook]) -> None:
        self.model = model
        self.hooks = hooks

    def after_backward(self, step: int) -> None:
        """Run every hook's `after_backward` in order."""
        for hook in self.hooks:
            hook.after_backward(step, self.model)

    def after_step(
        self, step: int, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        """Run every hook's `after_step` in order, passing on any replacement optimizer."""
        for hook in self.hooks:
            optimizer = hook.after_step(step, self.model, optimizer) or optimizer
        return optimizer


class ProbeRecord(TypedDict):
    step: int
    top_freq: list[int | None]  # harmonic base k* per neuron; None for a dead neuron
    harmonic_share: list[float | None]  # variance share over harmonics of k*
    r2_check: list[float | None]  # R^2 on the check split of the training rows


def split_fit_check(n: int, seed: int, fit_frac: float = 0.8) -> Tensor:
    """Bool mask over n rows: True for the fit rows (a fixed random `fit_frac`), False for check."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[order[: int(fit_frac * n)]] = True
    return mask


def drop_nan(values: list[float]) -> list[float | None]:
    """Replace nan by None so the list is valid JSON."""
    return [None if math.isnan(v) else v for v in values]


class ProbeHook(Hook):
    """Every `every` steps, fit all neurons of one layer on Fourier features of the training inputs.

    Only `x_train` is ever fed to the model. Appends one `ProbeRecord` per probe to `out_path`
    (JSON lines). The file is truncated on construction.
    """

    def __init__(
        self,
        x_train: Tensor,
        p: int,
        out_path: Path,
        layer: int = 0,
        every: int = 500,
        n_harmonics: int = 4,
        seed: int = 0,
    ) -> None:
        self.x_train = x_train
        self.p = p
        self.out_path = out_path
        self.layer = layer
        self.every = every
        self.n_harmonics = n_harmonics
        self.fit_rows = split_fit_check(len(x_train), seed)
        self.a, self.b = x_train[:, 0].cpu(), x_train[:, 1].cpu()
        self.features = build_features(self.a, self.b, p, "trig", 0)
        self.pinv = torch.linalg.pinv(self.features[0][self.fit_rows])
        out_path.write_text("")

    def probe(self, step: int, model: GrokMLP | GrokTransformer) -> ProbeRecord:
        """Fit every neuron now and return the record."""
        acts = record_activations(model, self.x_train, self.layer).cpu().double()
        neurons = list(range(acts.shape[1]))
        fits = fit_neurons(
            acts,
            self.a,
            self.b,
            self.p,
            "trig",
            0,
            neurons,
            fit_mask=self.fit_rows,
            features=self.features,
            pinv=self.pinv,
        )
        n_k = (self.p - 1) // 2
        shares = torch.tensor(
            [f["freq_shares"] or [math.nan] * n_k for f in fits], dtype=torch.float64
        )
        base, share = harmonic_share(shares, self.p, self.n_harmonics)
        alive = [bool(f["freq_shares"]) for f in fits]
        return {
            "step": step,
            "top_freq": [k if ok else None for k, ok in zip(base.tolist(), alive)],
            "harmonic_share": drop_nan(share.tolist()),
            "r2_check": drop_nan([f["r2_heldout"] for f in fits]),
        }

    def after_step(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer | None:
        """Probe and append a JSON line every `every` steps."""
        if step % self.every == 0:
            assert isinstance(model, GrokMLP | GrokTransformer)
            with self.out_path.open("a") as f:
                f.write(json.dumps(self.probe(step, model)) + "\n")
        return None
