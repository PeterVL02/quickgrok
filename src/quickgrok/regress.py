"""Fit regressions to MLP neuron activations as a function of the inputs (a, b). Runs on CPU.

Each neuron is evaluated at the "=" position on all p*p inputs and fit by least squares in one of
two bases:
  poly  monomials x^i y^j (i + j <= degree), with x, y = a, b rescaled to [-1, 1]
  trig  for every frequency k: cos/sin(w a), cos/sin(w b) and their 4 pairwise products,
        w = 2*pi*k/p. This is a degree-2 polynomial in Fourier features; it contains
        cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb), which grokked add/sub nets are known to use.
On the full grid every trig column is orthogonal, so the explained variance splits exactly per k.

Run: uv run python -m quickgrok.regress runs/p97_add/step_*.pt --neurons 0 1 2 --basis trig
"""

import argparse
import math
from pathlib import Path
from typing import Literal, TypedDict, get_args

import torch
from torch import Tensor

from quickgrok.train import GrokModel, get_activation_layers, load_model

Basis = Literal["poly", "trig"]


class NeuronFit(TypedDict):
    neuron: int
    r2: float  # on the fitted rows; nan for a dead (constant) neuron
    r2_heldout: float  # on rows outside fit_mask; nan when no mask is given
    coef: list[float]  # one per column of the design matrix
    top_freq: int | None  # trig only: frequency k explaining the most variance
    top_freq_share: float | None  # trig only: share of explained variance from top_freq
    # trig only: share per frequency k = 1..(p-1)/2; else empty
    freq_shares: list[float]


def make_grid(p: int) -> tuple[Tensor, Tensor]:
    """Return all (a, b) pairs as two int tensors of length p*p, row index = a*p + b."""
    a, b = torch.cartesian_prod(torch.arange(p), torch.arange(p)).T
    return a, b


def record_neurons(
    model: GrokModel, neurons: list[int] | None = None, layer: int = 0
) -> Tensor:
    """Return activations of hidden layer `layer` on all p*p inputs, shape (p*p, n_neurons).

    `neurons=None` means every neuron in the layer. For the transformer, the only layer is the
    MLP after attention, read at the "=" position.
    """
    p = model.unembed.out_features
    a, b = make_grid(p)
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    acts: list[Tensor] = []

    def keep(_m: torch.nn.Module, _i: tuple[Tensor, ...], out: Tensor) -> None:
        out = out[:, -1] if out.dim() == 3 else out  # transformer: (N, seq, d)
        acts.append(out if neurons is None else out[:, neurons])

    handle = get_activation_layers(model)[layer].register_forward_hook(keep)
    with torch.no_grad():
        model(x)
    handle.remove()
    return acts[0].double()


def build_features(
    a: Tensor, b: Tensor, p: int, basis: Basis, degree: int
) -> tuple[Tensor, list[int]]:
    """Return the design matrix (float64, constant column first) and each column's frequency k.

    The frequency is 0 for the constant column and for every column of the poly basis.
    """
    if basis == "poly":
        x, y = 2 * a.double() / (p - 1) - 1, 2 * b.double() / (p - 1) - 1
        cols = [x**i * y ** (d - i) for d in range(degree + 1) for i in range(d + 1)]
        return torch.stack(cols, dim=1), [0] * len(cols)
    cols, freqs = [torch.ones(len(a), dtype=torch.float64)], [0]
    for k in range(1, (p - 1) // 2 + 1):
        w = 2 * math.pi * k / p
        ca, sa = torch.cos(w * a.double()), torch.sin(w * a.double())
        cb, sb = torch.cos(w * b.double()), torch.sin(w * b.double())
        cols += [ca, sa, cb, sb, ca * cb, ca * sb, sa * cb, sa * sb]
        freqs += [k] * 8
    return torch.stack(cols, dim=1), freqs


def score_r2(y: Tensor, y_hat: Tensor) -> Tensor:
    """Per-column R^2 of predictions; nan where `y` is constant (dead neuron)."""
    ss_res = ((y - y_hat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return torch.where(ss_tot > 1e-12, 1 - ss_res / ss_tot, torch.nan)


def fit_neurons(
    acts: Tensor,
    a: Tensor,
    b: Tensor,
    p: int,
    basis: Basis,
    degree: int,
    neurons: list[int],
    fit_mask: Tensor | None = None,
) -> list[NeuronFit]:
    """Least-squares fit each column of `acts` (p*p, n) on the chosen basis and score it with R^2.

    With `fit_mask` (bool, p*p), only those rows are used for fitting; `r2` is then in-sample on
    them and `r2_heldout` is measured on the other rows. Without it, `r2_heldout` is nan.
    """
    X, freqs = build_features(a, b, p, basis, degree)
    mask = torch.ones(len(a), dtype=torch.bool) if fit_mask is None else fit_mask
    X_fit, y_fit = X[mask], acts[mask]
    coef = torch.linalg.lstsq(
        X_fit, y_fit, driver="gelsd"
    ).solution  # (features, neurons)
    r2 = score_r2(y_fit, X_fit @ coef)
    r2_heldout = (
        score_r2(acts[~mask], X[~mask] @ coef)
        if fit_mask is not None
        else torch.full_like(r2, torch.nan)
    )
    # Variance explained per column; split exactly by frequency when columns are orthogonal
    # (true on the full grid, approximately true on a random subset).
    energy = coef**2 * (X_fit**2).sum(0, keepdim=True).T
    per_k = torch.zeros(max(freqs) + 1, len(neurons), dtype=torch.float64)
    per_k.index_add_(0, torch.tensor(freqs), energy)
    per_k[0] = 0  # constant term explains no variance
    shares = per_k / per_k.sum(0).clamp_min(1e-12)

    fits: list[NeuronFit] = []
    for j, neuron in enumerate(neurons):
        alive = basis == "trig" and not math.isnan(r2[j].item())
        top_freq = int(shares[:, j].argmax()) if alive else None
        fits.append(
            {
                "neuron": neuron,
                "r2": r2[j].item(),
                "r2_heldout": r2_heldout[j].item(),
                "coef": coef[:, j].tolist(),
                "top_freq": top_freq,
                "top_freq_share": shares[top_freq, j].item() if top_freq else None,
                "freq_shares": shares[1:, j].tolist() if alive else [],
            }
        )
    return fits


def main() -> None:
    """Parse CLI arguments and print a fit table per checkpoint."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--neurons", type=int, nargs="+", required=True)
    parser.add_argument("--basis", choices=get_args(Basis), default="trig")
    parser.add_argument("--degree", type=int, default=6, help="poly basis only")
    parser.add_argument("--layer", type=int, default=0, help="hidden layer index")
    args = parser.parse_args()

    for path in args.checkpoints:
        model, ckpt = load_model(path)
        p = ckpt["p"]
        a, b = make_grid(p)
        acts = record_neurons(model, args.neurons, args.layer)
        fits = fit_neurons(acts, a, b, p, args.basis, args.degree, args.neurons)
        print(
            f"\n{path}  step {ckpt['step']}  train_acc {ckpt['train_acc']:.3f}  val_acc {ckpt['val_acc']:.3f}"
        )
        for f in fits:
            extra = (
                f"  top_freq {f['top_freq']} ({f['top_freq_share']:.0%})"
                if f["top_freq"]
                else ""
            )
            print(f"  neuron {f['neuron']:4d}  R^2 {f['r2']:.3f}{extra}")


if __name__ == "__main__":
    main()
