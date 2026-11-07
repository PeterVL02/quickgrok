"""Training hooks for grokking accelerators, usable from any PyTorch training loop.

Call `accel.after_backward(step)` between `loss.backward()` and `opt.step()`, and
`opt = accel.after_step(step, opt)` after `opt.step()`.
"""

import json
import math
import random
from pathlib import Path
from typing import Literal, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from quickgrok.models import GrokMLP, GrokTransformer
from quickgrok.regress import (
    build_features,
    fit_neurons,
    fold_freq,
    harmonic_share,
    record_activations,
)


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

    def log_probe(self, step: int, model: nn.Module) -> ProbeRecord:
        """Probe now, append the record to `out_path` as a JSON line and return it."""
        assert isinstance(model, GrokMLP | GrokTransformer)
        record = self.probe(step, model)
        with self.out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def after_step(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer | None:
        """Probe and append a JSON line every `every` steps."""
        if step % self.every == 0:
            self.log_probe(step, model)
        return None


SubstitutionMode = Literal["project", "module"]


class SubstitutionRecord(TypedDict):
    step: int
    mode: SubstitutionMode
    neurons: list[int]  # newly substituted at this step
    k_fitted: list[int]  # fitted base frequency k* per new neuron
    k: list[int]  # frequency used: k_fitted, or a random other one with random_k
    train_loss_before: float
    train_loss_after: float
    n_substituted_total: int


def harmonic_basis(ks: list[int], p: int, n_harmonics: int) -> Tensor:
    """Per-neuron basis on x = 0..p-1, shape (len(ks), p, 1 + 2 * n_harmonics), float32.

    Columns: 1, then cos and sin(2 pi f x / p) for each f in fold(m * k), m = 1..n_harmonics,
    duplicates dropped. A dropped duplicate (only possible for small p) leaves zero columns.
    """
    x = torch.arange(p, dtype=torch.float64)
    basis = torch.zeros(len(ks), p, 1 + 2 * n_harmonics, dtype=torch.float64)
    basis[:, :, 0] = 1
    for i, k in enumerate(ks):
        freqs = dict.fromkeys(fold_freq(m * k, p) for m in range(1, n_harmonics + 1))
        for j, f in enumerate(freqs):
            basis[i, :, 1 + 2 * j] = torch.cos(2 * math.pi * f * x / p)
            basis[i, :, 2 + 2 * j] = torch.sin(2 * math.pi * f * x / p)
    return basis.float()


def fit_coef(basis: Tensor, cols: Tensor) -> Tensor:
    """Least-squares coefficients (n, n_basis) of columns `cols` (n, p) on `basis` (n, p, n_basis)."""
    coef = torch.linalg.pinv(basis.double()) @ cols.double()[..., None]
    return coef[..., 0].to(cols.dtype)


def expand(basis: Tensor, coef: Tensor) -> Tensor:
    """Basis expansion (n, p) of coefficients `coef` (n, n_basis)."""
    return (basis @ coef[..., None])[..., 0]


class HarmonicColumns(nn.Module):
    """Parametrization of an embedding weight (p, d): columns `idx` become `basis @ coef`."""

    idx: Tensor
    basis: Tensor

    def __init__(self, p: int, n_basis: int, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("idx", torch.zeros(0, dtype=torch.long, device=device))
        self.register_buffer("basis", torch.zeros(0, p, n_basis, device=device))
        self.coef = nn.Parameter(torch.zeros(0, n_basis, device=device))

    def forward(self, W: Tensor) -> Tensor:
        """Return W with columns `idx` replaced by their basis expansions."""
        return W.index_copy(1, self.idx, expand(self.basis, self.coef).T)

    def grow(self, idx: Tensor, basis: Tensor, W: Tensor) -> None:
        """Add columns `idx` with their bases; new coef rows are the least-squares fit to W[:, idx].

        Replaces `coef` by a new Parameter, so the optimizer must be rebuilt.
        """
        self.idx = torch.cat([self.idx, idx])
        self.basis = torch.cat([self.basis, basis])
        new = fit_coef(basis, W[:, idx].T)
        self.coef = nn.Parameter(torch.cat([self.coef.detach(), new]))


def rebuild_optimizer(
    old: torch.optim.Optimizer,
    model: nn.Module,
    grown: dict[nn.Parameter, nn.Parameter],
) -> torch.optim.Optimizer:
    """New optimizer of the same class and hyperparameters over the model's current parameters.

    State is copied for every parameter `old` already tracks. `grown` maps an old Parameter to the
    longer one that replaced it: its state tensors keep the old rows and get zero rows appended
    (the `step` counter is kept).
    """
    # ponytail: one param group, which is all train.py builds; map groups if that changes
    assert len(old.param_groups) == 1, "only single-group optimizers are supported"
    new = type(old)(model.parameters())
    new.param_groups[0].update(
        {k: v for k, v in old.param_groups[0].items() if k != "params"}
    )
    source = {q: q for q in old.state} | {n: o for o, n in grown.items()}
    for q in new.param_groups[0]["params"]:
        src = source.get(q)
        if src is None or src not in old.state:
            continue
        new.state[q] = {
            name: torch.cat([v, v.new_zeros(len(q) - len(v), *v.shape[1:])])
            if src is not q and v.shape == src.shape
            else v
            for name, v in old.state[src].items()
        }
    return new


def effective_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """The state dict the model would have without parametrizations (effective weights)."""
    state = {
        k: v for k, v in model.state_dict().items() if ".parametrizations." not in k
    }
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module):
            for tensor_name in module.parametrizations:
                state[f"{name}.{tensor_name}"] = getattr(module, tensor_name).detach()
    return state


class SubstitutionHook(ProbeHook):
    """Probe like `ProbeHook`, then substitute neurons that pass the trigger for the first time.

    Trigger: harmonic share >= `tau_share` and check-split R^2 >= `tau_r2`. Both input columns
    U[:, i], V[:, i] of a triggered neuron are projected onto the harmonics of its k*.
    "project" overwrites them in place and lets them train freely afterwards; "module" keeps them
    in that span for good (trainable coefficients via `HarmonicColumns`). Only GrokMLP layer 0.
    `random_k` (control) swaps each k* for a uniformly random other frequency, seeded by `seed`.
    `sub_all_at` (a probe step) ignores the thresholds: every live neuron is substituted at that
    step and none at any other. Appends one `SubstitutionRecord` per event to `log_path` (truncated on construction).
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
        *,
        mode: SubstitutionMode,
        tau_share: float,
        tau_r2: float,
        y_train: Tensor,
        log_path: Path,
        random_k: bool = False,
        sub_all_at: int | None = None,
    ) -> None:
        if layer != 0:
            raise ValueError(f"substitution only supports GrokMLP layer 0, got {layer}")
        if sub_all_at is not None and sub_all_at % every:
            raise ValueError(
                f"sub_all_at={sub_all_at} is not a probe step (every {every})"
            )
        super().__init__(x_train, p, out_path, layer, every, n_harmonics, seed)
        self.mode = mode
        self.tau_share = tau_share
        self.tau_r2 = tau_r2
        self.y_train = y_train
        self.log_path = log_path
        self.random_k = random_k
        self.sub_all_at = sub_all_at
        self.rng = random.Random(seed)
        self.substituted: set[int] = set()
        log_path.write_text("")

    def train_loss(self, model: nn.Module) -> float:
        """Full-batch cross-entropy on the training set."""
        with torch.no_grad():
            return F.cross_entropy(model(self.x_train), self.y_train).item()

    def substitute(
        self,
        model: GrokMLP,
        neurons: list[int],
        ks: list[int],
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer | None:
        """Apply the mode to `neurons` with base frequencies `ks`; return a new optimizer if needed."""
        device = model.bias.device
        idx = torch.tensor(neurons, device=device)
        basis = harmonic_basis(ks, self.p, self.n_harmonics).to(device)
        embeds = (model.embed_a, model.embed_b)
        if self.mode == "project":
            with torch.no_grad():
                for e in embeds:
                    e.weight[:, idx] = expand(
                        basis, fit_coef(basis, e.weight[:, idx].T)
                    ).T
            return None
        grown: dict[nn.Parameter, nn.Parameter] = {}
        for e in embeds:
            if not parametrize.is_parametrized(e, "weight"):
                columns = HarmonicColumns(self.p, basis.shape[-1], device)
                parametrize.register_parametrization(e, "weight", columns)
            columns = e.parametrizations.weight[0]
            old = columns.coef
            with torch.no_grad():
                columns.grow(idx, basis, e.weight)
            grown[old] = columns.coef
        return rebuild_optimizer(optimizer, model, grown)

    def after_step(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer | None:
        """Every `every` steps: probe, substitute newly triggered neurons and log the event."""
        if step % self.every:
            return None
        if not isinstance(model, GrokMLP):
            raise TypeError(f"substitution only supports GrokMLP, got {type(model)}")
        record = self.log_probe(step, model)
        all_now = step == self.sub_all_at
        if self.sub_all_at is not None and not all_now:
            return None
        picks = [
            (i, k)
            for i, (k, share, r2) in enumerate(
                zip(record["top_freq"], record["harmonic_share"], record["r2_check"])
            )
            if k is not None
            and i not in self.substituted
            and (
                all_now
                or (
                    share is not None
                    and r2 is not None
                    and share >= self.tau_share
                    and r2 >= self.tau_r2
                )
            )
        ]
        if not picks:
            return None
        neurons, fitted = [i for i, _ in picks], [k for _, k in picks]
        n_k = (self.p - 1) // 2
        ks = [
            self.rng.choice([j for j in range(1, n_k + 1) if j != k])
            if self.random_k
            else k
            for k in fitted
        ]
        loss_before = self.train_loss(model)
        new_optimizer = self.substitute(model, neurons, ks, optimizer)
        self.substituted.update(neurons)
        event: SubstitutionRecord = {
            "step": step,
            "mode": self.mode,
            "neurons": neurons,
            "k_fitted": fitted,
            "k": ks,
            "train_loss_before": loss_before,
            "train_loss_after": self.train_loss(model),
            "n_substituted_total": len(self.substituted),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")
        return new_optimizer
