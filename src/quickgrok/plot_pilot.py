"""Pilot figures: training curves and a regression probe of MLP neurons over training.

Run: uv run python -m quickgrok.plot_pilot runs/pilot/seed0 runs/pilot/seed1 runs/pilot/seed2
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from quickgrok.data import Operation, make_dataset
from quickgrok.models import load_model
from quickgrok.regress import (
    build_features,
    fit_neurons,
    make_grid,
    record_neurons,
)
from quickgrok.train import Metrics

# Reference palette (dataviz skill), light mode
TRAIN, VAL = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROK_ACC = 0.99


class ProbeResult(TypedDict):
    seed: int
    steps: list[int]
    val_acc: list[float]
    trig_r2_fit: list[
        tuple[float, float, float]
    ]  # (q25, median, q75) over live neurons
    trig_r2_heldout: list[tuple[float, float, float]]
    poly_r2_heldout: list[tuple[float, float, float]]
    top_share: list[tuple[float, float, float]]
    spectrum: list[list[float]]  # per checkpoint: mean share of neuron variance per k


def style_axes(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_history(run_dir: Path) -> list[Metrics]:
    """Read the metrics log written by train.py."""
    return json.loads((run_dir / "metrics.json").read_text())


def find_grok_step(history: list[Metrics], threshold: float = GROK_ACC) -> int | None:
    """First logged step with val accuracy >= threshold, or None if never reached."""
    return next((m["step"] for m in history if m["val_acc"] >= threshold), None)


def quartiles(x: Tensor) -> tuple[float, float, float]:
    """(q25, median, q75) ignoring nan."""
    q = np.nanquantile(x.numpy(), [0.25, 0.5, 0.75])
    return float(q[0]), float(q[1]), float(q[2])


def make_train_mask(p: int, op: Operation, train_frac: float, seed: int) -> Tensor:
    """Bool mask over the p*p grid (row a*p + b) marking the training pairs."""
    train, _ = make_dataset(p, op, train_frac, seed)
    mask = torch.zeros(p * p, dtype=torch.bool)
    mask[[a * p + b for (a, b, _), _ in train]] = True
    return mask


def probe_run(run_dir: Path, layer: int = 0, poly_degree: int = 6) -> ProbeResult:
    """Fit every MLP neuron at every checkpoint: trig and poly on train pairs, trig on full grid."""
    paths = sorted(run_dir.glob("step_*.pt"))
    model, ckpt = load_model(paths[0])
    p = ckpt["p"]
    a, b = make_grid(p)
    mask = make_train_mask(p, ckpt["op"], ckpt["train_frac"], ckpt["seed"])
    neurons = list(range(record_neurons(model, layer=layer).shape[1]))
    result: ProbeResult = {
        "seed": ckpt["seed"],
        "steps": [],
        "val_acc": [],
        "trig_r2_fit": [],
        "trig_r2_heldout": [],
        "poly_r2_heldout": [],
        "top_share": [],
        "spectrum": [],
    }
    for path in paths:
        model, ckpt = load_model(path)
        acts = record_neurons(model, neurons, layer)
        trig = fit_neurons(acts, a, b, p, "trig", 0, neurons, fit_mask=mask)
        poly = fit_neurons(acts, a, b, p, "poly", poly_degree, neurons, fit_mask=mask)
        full = [
            f
            for f in fit_neurons(acts, a, b, p, "trig", 0, neurons)
            if f["freq_shares"]
        ]
        result["steps"].append(ckpt["step"])
        result["val_acc"].append(ckpt["val_acc"])
        result["trig_r2_fit"].append(quartiles(torch.tensor([f["r2"] for f in trig])))
        result["trig_r2_heldout"].append(
            quartiles(torch.tensor([f["r2_heldout"] for f in trig]))
        )
        result["poly_r2_heldout"].append(
            quartiles(torch.tensor([f["r2_heldout"] for f in poly]))
        )
        result["top_share"].append(
            quartiles(torch.tensor([f["top_freq_share"] for f in full]))
        )
        result["spectrum"].append(
            torch.tensor([f["freq_shares"] for f in full]).mean(0).tolist()
        )
    return result


def mark_grok(ax: plt.Axes, step: int | None) -> None:
    """Dashed vertical line at the grokking step."""
    if step is not None:
        ax.axvline(step, color=MUTED, linestyle="--", linewidth=1)


def plot_training_curves(
    histories: list[list[Metrics]], seeds: list[int], out: Path
) -> None:
    """Rows: loss, accuracy, weight norm. Columns: seeds. Train blue, test orange."""
    fig, axes = plt.subplots(
        3, len(histories), figsize=(4 * len(histories), 8), sharex=True, sharey="row"
    )
    axes = np.atleast_2d(axes.T).T if len(histories) == 1 else axes
    for col, (hist, seed) in enumerate(zip(histories, seeds)):
        h = [m for m in hist if m["step"] > 0]  # log x-axis
        steps = [m["step"] for m in h]
        grok = find_grok_step(hist)
        ax_loss, ax_acc, ax_norm = axes[:, col]
        ax_loss.plot(
            steps, [m["train_loss"] for m in h], color=TRAIN, lw=2, label="train"
        )
        ax_loss.plot(steps, [m["val_loss"] for m in h], color=VAL, lw=2, label="test")
        ax_loss.set_yscale("log")
        ax_acc.plot(
            steps, [m["train_acc"] for m in h], color=TRAIN, lw=2, label="train"
        )
        ax_acc.plot(steps, [m["val_acc"] for m in h], color=VAL, lw=2, label="test")
        ax_norm.plot(steps, [m["weight_norm"] for m in h], color=MUTED, lw=2)
        title = (
            f"seed {seed}: test acc ≥ {GROK_ACC:.0%} at step {grok:,}"
            if grok
            else f"seed {seed}"
        )
        ax_loss.set_title(title, fontsize=10, color=INK, loc="left")
        for ax in (ax_loss, ax_acc, ax_norm):
            ax.set_xscale("log")
            style_axes(ax)
            mark_grok(ax, grok)
        ax_norm.set_xlabel("training step (log)", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel("cross-entropy loss (log)", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("accuracy", color=MUTED, fontsize=9)
    axes[2, 0].set_ylabel("parameter L2 norm", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Grokking on a + b mod 97: train fits early, test accuracy jumps much later",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_band(
    ax: plt.Axes, x: list[int], q: list[tuple[float, float, float]], **kw: str
) -> None:
    """Median line with an interquartile band."""
    arr = np.array(q)
    ax.fill_between(x, arr[:, 0], arr[:, 2], color=kw["color"], alpha=0.15, linewidth=0)
    ax.plot(x, arr[:, 1], lw=2, **kw)


def plot_probe(results: list[ProbeResult], groks: list[int | None], out: Path) -> None:
    """Rows: test accuracy, neuron fit R^2, dominant-frequency share. Columns: seeds."""
    fig, axes = plt.subplots(
        3, len(results), figsize=(4 * len(results), 8), sharex=True, sharey="row"
    )
    axes = np.atleast_2d(axes.T).T if len(results) == 1 else axes
    for col, (r, grok) in enumerate(zip(results, groks)):
        ax_acc, ax_r2, ax_share = axes[:, col]
        ax_acc.plot(r["steps"], r["val_acc"], color=VAL, lw=2)
        plot_band(
            ax_r2,
            r["steps"],
            r["trig_r2_heldout"],
            color=TRAIN,
            label="Fourier, held-out",
        )
        plot_band(
            ax_r2,
            r["steps"],
            r["trig_r2_fit"],
            color=TRAIN,
            linestyle=":",
            label="Fourier, fitted rows",
        )
        plot_band(
            ax_r2,
            r["steps"],
            r["poly_r2_heldout"],
            color=VAL,
            label="polynomial, held-out",
        )
        plot_band(ax_share, r["steps"], r["top_share"], color=TRAIN)
        ax_acc.set_title(f"seed {r['seed']}", fontsize=10, color=INK, loc="left")
        for ax in (ax_acc, ax_r2, ax_share):
            style_axes(ax)
            mark_grok(ax, grok)
        ax_r2.set_ylim(-0.1, 1.02)
        ax_share.set_xlabel("training step", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel("test accuracy", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("neuron fit R² (median, IQR)", color=MUTED, fontsize=9)
    axes[2, 0].set_ylabel("variance share of top frequency", color=MUTED, fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle(
        "Regression probe of all hidden neurons in one layer, fit on training pairs only",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_spectrum(r: ProbeResult, grok: int | None, out: Path) -> None:
    """Heatmap: frequency k vs step, value = mean share of neuron variance at k."""
    spec = np.array(r["spectrum"]).T  # (k, checkpoint)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    step_w = r["steps"][1] - r["steps"][0]
    extent = (
        r["steps"][0] - step_w / 2,
        r["steps"][-1] + step_w / 2,
        0.5,
        spec.shape[0] + 0.5,
    )
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap="Blues", extent=extent)
    mark_grok(ax, grok)
    ax.set_xlabel("training step", color=MUTED, fontsize=9)
    ax.set_ylabel("frequency k", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean share of neuron variance", color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(
        f"seed {r['seed']}: neurons concentrate on a few frequencies around grokking (dashed)",
        fontsize=11,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_neuron_maps(
    run_dir: Path, grok: int | None, out: Path, layer: int = 0, n_neurons: int = 4
) -> None:
    """Activation of the most active neurons over (a, b) at a few steps, plus the final Fourier fit."""
    paths = sorted(run_dir.glob("step_*.pt"))
    final_model, ckpt = load_model(paths[-1])
    p = ckpt["p"]
    a, b = make_grid(p)
    final = record_neurons(final_model, layer=layer)
    all_neurons = list(range(final.shape[1]))
    final_fits = fit_neurons(final, a, b, p, "trig", 0, all_neurons)
    # The most active neuron for each of the first n distinct dominant frequencies.
    neurons, seen = [], set()
    for j in final.var(0).argsort(descending=True).tolist():
        k = final_fits[j]["top_freq"]
        if k is not None and k not in seen:
            neurons.append(j)
            seen.add(k)
        if len(neurons) == n_neurons:
            break
    grok = grok or ckpt["step"]
    wanted = [1000, max(1000, grok - 4000), grok, ckpt["step"]]
    by_step = {int(q.stem.removeprefix("step_")): q for q in paths}
    wanted = [
        min(by_step, key=lambda t: abs(t - s)) for s in wanted
    ]  # nearest checkpoint
    cols = [
        (f"step {s:,}", record_neurons(load_model(by_step[s])[0], neurons, layer))
        for s in wanted
    ]
    X, _ = build_features(a, b, p, "trig", 0)
    fits = fit_neurons(cols[-1][1], a, b, p, "trig", 0, neurons)
    coef = torch.tensor([f["coef"] for f in fits], dtype=torch.float64).T
    cols.append((f"Fourier fit of step {ckpt['step']:,}", X @ coef))

    fig, axes = plt.subplots(
        n_neurons, len(cols), figsize=(2.6 * len(cols), 2.5 * n_neurons)
    )
    for row, neuron in enumerate(neurons):
        # Order a and b by k*x mod p so a frequency-k wave shows as smooth stripes.
        order = np.argsort((fits[row]["top_freq"] * np.arange(p)) % p)
        for col, (label, acts) in enumerate(cols):
            ax = axes[row, col]
            img = acts[:, row].reshape(p, p).numpy()[np.ix_(order, order)]
            ax.imshow(
                img, cmap="Blues", vmin=0, vmax=max(img.max(), 1e-9), origin="lower"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel(
                    f"neuron {neuron}\na (by k·a mod p)", fontsize=9, color=MUTED
                )
            if row == n_neurons - 1:
                ax.set_xlabel("b (by k·b mod p)", fontsize=9, color=MUTED)
        r2 = fits[row]["r2"]
        axes[row, -1].text(
            0.03,
            0.03,
            f"R² {r2:.2f}, k={fits[row]['top_freq']}",
            transform=axes[row, -1].transAxes,
            fontsize=8,
            color=INK,
            backgroundcolor="white",
        )
    fig.suptitle(
        f"seed {ckpt['seed']}: each panel scaled to its own max; grokking at step {grok:,}",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Make all pilot figures for the given run directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("reports/figures/pilot"))
    parser.add_argument("--layer", type=int, default=0, help="hidden layer to probe")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    histories = [load_history(r) for r in args.runs]
    groks = [find_grok_step(h) for h in histories]
    seeds = [load_model(min(r.glob("step_*.pt")))[1]["seed"] for r in args.runs]
    plot_training_curves(histories, seeds, args.out / "training_curves.png")
    results = [probe_run(r, args.layer) for r in args.runs]
    (args.out / "probe.json").write_text(json.dumps(results))
    plot_probe(results, groks, args.out / "probe_r2.png")
    plot_spectrum(results[0], groks[0], args.out / "frequency_spectrum.png")
    plot_neuron_maps(args.runs[0], groks[0], args.out / "neuron_maps.png", args.layer)
    for seed, grok in zip(seeds, groks):
        print(f"seed {seed}: test acc >= {GROK_ACC:.0%} at step {grok}")


if __name__ == "__main__":
    main()
