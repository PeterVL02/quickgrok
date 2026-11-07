"""Training hooks for grokking accelerators, usable from any PyTorch training loop.

Call `accel.after_backward(step)` between `loss.backward()` and `opt.step()`, and
`opt = accel.after_step(step, opt)` after `opt.step()`.
"""

import json
import math
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from quickgrok.models import GrokMLP, GrokTransformer
from quickgrok.regress import (
    build_features,
    fit_neurons,
    fold_freq,
    harmonic_matrix,
    harmonic_share,
    record_activations,
    score_r2,
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
        n_k = self.p // 2
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
# "probe": harmonics of the activation fit's k* (T6). The others are chosen from the weights.
BasisName = Literal["probe", "fourier", "poly", "svd", "random", "auto"]
AUTO_CANDIDATES: list[BasisName] = ["svd", "fourier", "poly"]  # order breaks ties


class EventFields(TypedDict):
    # probe: fitted k* per new neuron; other bases: the fourier k, None for non-fourier neurons
    k_fitted: list[int | None]
    k: list[int | None]  # frequency used: k_fitted, or a random other one with random_k
    basis: NotRequired[list[BasisName]]  # not "probe": basis per new neuron
    energy: NotRequired[
        list[float | None]
    ]  # not "probe": captured energy per new neuron
    r2_check: NotRequired[list[float | None]]  # not "probe": check R^2 per new neuron


class SubstitutionRecord(EventFields):
    step: int
    mode: SubstitutionMode
    neurons: list[int]  # newly substituted at this step
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


def unit_rms(basis: Tensor) -> Tensor:
    """Scale every column of `basis` (..., n, m) to RMS 1; zero columns stay zero."""
    return basis / basis.pow(2).mean(-2, keepdim=True).sqrt().clamp_min(1e-12)


def fourier_basis(u: Tensor, v: Tensor, m: int = 9) -> tuple[Tensor, list[int]]:
    """Per-neuron harmonic basis (k, n, m) whose base frequency maximises the captured energy.

    `u`, `v` (k, n) are a neuron's two input columns. For each base k the energy of the
    mean-removed columns at the (m - 1) // 2 folded harmonics of k is summed from their DFT,
    which on the full grid of n symbols is exactly the energy the basis captures.
    Returns (basis with unit-RMS columns, base frequency per neuron).
    """
    n = u.shape[-1]
    u0, v0 = (c - c.mean(-1, keepdim=True) for c in (u, v))
    power = (torch.fft.rfft(u0).abs() ** 2 + torch.fft.rfft(v0).abs() ** 2)[
        :, 1 : n // 2 + 1
    ]
    if n % 2 == 0:
        power[:, -1] /= 2  # the Nyquist frequency has no mirror image
    harmonics = harmonic_matrix(n, (m - 1) // 2, power.dtype)
    ks = ((power @ harmonics.T).argmax(-1) + 1).tolist()
    return unit_rms(harmonic_basis(ks, n, (m - 1) // 2).to(u.dtype)), ks


def poly_basis(n_neurons: int, n: int, m: int = 9) -> Tensor:
    """Legendre polynomials of degree 0..m-1 in the symbol index rescaled to [-1, 1]: (k, n, m)."""
    legendre = np.polynomial.legendre.legvander(np.linspace(-1, 1, n), m - 1)
    return unit_rms(torch.from_numpy(legendre)).expand(n_neurons, n, m)


def svd_vectors(U: Tensor, V: Tensor) -> Tensor:
    """Left singular vectors (n, n - 1) of the mean-removed [U, V] (n, 2d), largest first.

    Removing each column's mean makes every vector orthogonal to the constant; the last
    vector (zero singular value, the constant direction) is dropped.
    """
    W = torch.cat([U, V], 1)
    left = torch.linalg.svd(W - W.mean(0), full_matrices=False)[0]
    return left[:, : W.shape[0] - 1]


def svd_basis(u: Tensor, v: Tensor, vectors: Tensor, m: int = 9) -> Tensor:
    """Per neuron: constant plus the m - 1 of `vectors` (n, r) that capture most of u and v.

    Score of a vector = squared projection of u plus that of v. Returns (k, n, m), unit-RMS columns.
    """
    top = ((u @ vectors) ** 2 + (v @ vectors) ** 2).topk(m - 1).indices  # (k, m - 1)
    chosen = vectors.T[top].transpose(1, 2)  # (k, n, m - 1)
    return unit_rms(torch.cat([torch.ones_like(chosen[..., :1]), chosen], -1))


def random_basis(neurons: list[int], n: int, m: int, seed: int) -> Tensor:
    """Control: constant plus m - 1 random orthonormal directions orthogonal to it, (k, n, m).

    The directions of neuron i depend only on (seed, i).
    """
    bases = []
    for i in neurons:
        g = torch.Generator().manual_seed(seed * 100_003 + i)
        q = torch.randn(n, m - 1, generator=g, dtype=torch.float64)
        q = torch.linalg.qr(q - q.mean(0)).Q
        bases.append(torch.cat([torch.ones(n, 1, dtype=torch.float64), q], 1))
    return unit_rms(torch.stack(bases))


def project(basis: Tensor, cols: Tensor) -> Tensor:
    """Least-squares projection (k, n) of columns `cols` (k, n) onto `basis` (k, n, m)."""
    return expand(basis, fit_coef(basis, cols))


def captured_energy(basis: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Share (k,) of the energy of the mean-removed columns u, v (k, n) inside `basis` (k, n, m).

    The basis contains the constant, so this is ||P[u, v] - mean||^2 / ||[u, v] - mean||^2.
    nan for a neuron whose columns are both constant.
    """
    num, den = torch.zeros(len(u), dtype=u.dtype), torch.zeros(len(u), dtype=u.dtype)
    for c in (u, v):
        mean = c.mean(-1, keepdim=True)
        num = num + ((project(basis, c) - mean) ** 2).sum(-1)
        den = den + ((c - mean) ** 2).sum(-1)
    return torch.where(den > 1e-12, num / den.clamp_min(1e-12), torch.nan)


def make_basis(
    name: BasisName, U: Tensor, V: Tensor, neurons: list[int], m: int, seed: int
) -> tuple[Tensor, list[BasisName], list[int | None]]:
    """Basis (k, n, m) for `neurons` of the input weights U, V (n, d), chosen from the weights.

    Returns (basis, basis name per neuron, fourier k per neuron or None). "auto" picks per
    neuron the candidate in `AUTO_CANDIDATES` with the highest captured energy (ties: svd).
    Raises ValueError for "probe", which is built from activation fits instead.
    """
    u, v = U[:, neurons].T, V[:, neurons].T
    n = U.shape[0]
    candidates: dict[BasisName, Tensor] = {}
    ks: list[int | None] = [None] * len(neurons)
    if name in ("fourier", "auto"):
        candidates["fourier"], fourier_ks = fourier_basis(u, v, m)
        ks = list(fourier_ks)
    if name in ("poly", "auto"):
        candidates["poly"] = poly_basis(len(neurons), n, m).to(u.dtype)
    if name in ("svd", "auto"):
        candidates["svd"] = svd_basis(u, v, svd_vectors(U, V), m)
    if name == "random":
        candidates["random"] = random_basis(neurons, n, m, seed).to(u.dtype)
    if name == "probe":
        raise ValueError("the probe basis comes from activation fits, not weights")
    if name != "auto":
        return candidates[name], [name] * len(neurons), ks
    stack = torch.stack([candidates[c] for c in AUTO_CANDIDATES])  # (3, k, n, m)
    energy = torch.stack([captured_energy(b, u, v) for b in stack]).nan_to_num(-1)
    best = energy.argmax(0).tolist()  # first maximum, so ties go to svd
    names = [AUTO_CANDIDATES[j] for j in best]
    basis = stack[torch.tensor(best), torch.arange(len(neurons))]
    return basis, names, [k if c == "fourier" else None for k, c in zip(ks, names)]


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
    # Hyperparameters are copied from the old param group below, so the class needs only params.
    make = cast(Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer], type(old))
    new = make(model.parameters())
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
            for tensor_name in cast(nn.ModuleDict, module.parametrizations):
                state[f"{name}.{tensor_name}"] = getattr(module, tensor_name).detach()
    return state


class SubstitutionHook(ProbeHook):
    """Probe like `ProbeHook`, then substitute neurons that pass the trigger for the first time.

    Both input columns U[:, i], V[:, i] of a triggered neuron are projected onto a basis of
    m = 1 + 2 * n_harmonics columns. "project" overwrites them in place and lets them train
    freely afterwards; "module" keeps them in that span for good (trainable coefficients via
    `HarmonicColumns`). Only GrokMLP layer 0.
    `basis="probe"` (T6): the harmonics of the activation fit's k*; trigger: harmonic share
    >= `tau_share` and check-split R^2 >= `tau_r2`. `random_k` (control, probe only) swaps each
    k* for a uniformly random other frequency, seeded by `seed`.
    Any other basis is chosen from the weights by `make_basis`; trigger: captured energy
    >= `tau_energy` and `check_r2` >= `tau_r2`.
    `sub_all_at` (a probe step) ignores the thresholds: every live neuron is substituted at that
    step and none at any other. Appends one `SubstitutionRecord` per event to `log_path`
    (truncated on construction).
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
        basis: BasisName = "probe",
        tau_energy: float = 0.8,
    ) -> None:
        if layer != 0:
            raise ValueError(f"substitution only supports GrokMLP layer 0, got {layer}")
        if sub_all_at is not None and sub_all_at % every:
            raise ValueError(
                f"sub_all_at={sub_all_at} is not a probe step (every {every})"
            )
        if random_k and basis != "probe":
            raise ValueError("random_k only applies to basis='probe'")
        super().__init__(x_train, p, out_path, layer, every, n_harmonics, seed)
        self.mode: SubstitutionMode = mode
        self.tau_share = tau_share
        self.tau_r2 = tau_r2
        self.y_train = y_train
        self.log_path = log_path
        self.random_k = random_k
        self.sub_all_at = sub_all_at
        self.basis: BasisName = basis
        self.tau_energy = tau_energy
        self.seed = seed
        self.check_a, self.check_b = self.a[~self.fit_rows], self.b[~self.fit_rows]
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
        basis = harmonic_basis(ks, self.p, self.n_harmonics)
        return self.apply_basis(model, neurons, basis, optimizer)

    def apply_basis(
        self,
        model: GrokMLP,
        neurons: list[int],
        basis: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer | None:
        """Apply the mode to `neurons` with per-neuron bases (k, n, m); return a new optimizer if needed."""
        device = model.bias.device
        idx = torch.tensor(neurons, device=device)
        basis = basis.to(device=device, dtype=model.bias.dtype)
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
            plist = cast(nn.ModuleDict, e.parametrizations)["weight"]
            columns = cast(HarmonicColumns, cast(nn.ModuleList, plist)[0])
            old = columns.coef
            with torch.no_grad():
                columns.grow(idx, basis, e.weight)
            grown[old] = columns.coef
        return rebuild_optimizer(optimizer, model, grown)

    def input_weights(self, model: GrokMLP) -> tuple[Tensor, Tensor, Tensor]:
        """Effective U, V (n, d) and bias (d,) of layer 0 as float64 CPU tensors."""
        U, V, bias = (
            w.detach().cpu().double()
            for w in (model.embed_a.weight, model.embed_b.weight, model.bias)
        )
        return U, V, bias

    def check_r2(self, model: GrokMLP, neurons: list[int], basis: Tensor) -> Tensor:
        """R^2 (k,) of each neuron's activation against its activation with u, v projected.

        Measured on the check split of the training rows only (`check_a`, `check_b`).
        nan for a neuron that is constant there.
        """
        U, V, bias = self.input_weights(model)
        u, v, c = U[:, neurons].T, V[:, neurons].T, bias[neurons, None]

        def activate(u: Tensor, v: Tensor) -> Tensor:
            """Layer-0 activations (N_check, k) for input columns u, v (k, n)."""
            return torch.relu(u[:, self.check_a] + v[:, self.check_b] + c).T

        return score_r2(activate(u, v), activate(project(basis, u), project(basis, v)))

    def pick_probe(
        self, record: ProbeRecord, live: list[int], all_now: bool
    ) -> tuple[list[int], Tensor, EventFields]:
        """Probe-basis trigger: (neurons, their harmonic bases, the event's per-neuron fields)."""
        picks = [
            (i, k)
            for i in live
            if (k := record["top_freq"][i]) is not None
            and (
                all_now
                or (
                    (share := record["harmonic_share"][i]) is not None
                    and (r2 := record["r2_check"][i]) is not None
                    and share >= self.tau_share
                    and r2 >= self.tau_r2
                )
            )
        ]
        neurons, fitted = [i for i, _ in picks], [k for _, k in picks]
        n_k = self.p // 2
        ks = [
            self.rng.choice([j for j in range(1, n_k + 1) if j != k])
            if self.random_k
            else k
            for k in fitted
        ]
        basis = harmonic_basis(ks, self.p, self.n_harmonics)
        return neurons, basis, {"k_fitted": list(fitted), "k": list(ks)}

    def pick_generic(
        self, model: GrokMLP, live: list[int], all_now: bool
    ) -> tuple[list[int], Tensor, EventFields]:
        """Weight-basis trigger: (neurons, their bases, the event's per-neuron fields)."""
        U, V, _ = self.input_weights(model)
        m = 1 + 2 * self.n_harmonics
        basis, names, ks = make_basis(self.basis, U, V, live, m, self.seed)
        energy = captured_energy(basis, U[:, live].T, V[:, live].T)
        r2 = self.check_r2(model, live, basis)
        keep = [
            j
            for j in range(len(live))
            if all_now or (energy[j] >= self.tau_energy and r2[j] >= self.tau_r2)
        ]
        fields: EventFields = {
            "k_fitted": [ks[j] for j in keep],
            "k": [ks[j] for j in keep],
            "basis": [names[j] for j in keep],
            "energy": drop_nan(energy[keep].tolist()),
            "r2_check": drop_nan(r2[keep].tolist()),
        }
        return [live[j] for j in keep], basis[keep], fields

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
        live = [
            i
            for i, k in enumerate(record["top_freq"])
            if k is not None and i not in self.substituted
        ]
        if not live:
            return None
        neurons, basis, fields = (
            self.pick_probe(record, live, all_now)
            if self.basis == "probe"
            else self.pick_generic(model, live, all_now)
        )
        if not neurons:
            return None
        loss_before = self.train_loss(model)
        new_optimizer = self.apply_basis(model, neurons, basis, optimizer)
        self.substituted.update(neurons)
        event: SubstitutionRecord = {
            "step": step,
            "mode": self.mode,
            "neurons": neurons,
            **fields,
            "train_loss_before": loss_before,
            "train_loss_after": self.train_loss(model),
            "n_substituted_total": len(self.substituted),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")
        return new_optimizer
