"""How much of each input column does each candidate basis capture, over training? (G1b)

Report: reports/generic_basis.md. Offline, CPU only. For every checkpoint of the vanilla MLP runs of add, mul and s5, and every
input column (u_i and v_i separately, mean removed), the captured share of its energy by:
  fourier     harmonics of the column's best base frequency, 8 directions (knows cyclicity)
  svd_global  the 8 left singular vectors of the mean-removed [U, V] that capture it most
  levels_L    optimal 1-D k-means of the column's values into L groups (L - 1 directions)
  self_expr   greedy orthogonal matching pursuit: 8 other columns of the same matrix
Nulls: the step-0 checkpoint and Gaussian matrices of the same shape. Skill = (capture - null) /
(1 - null) with the Gaussian null's median. Full-grid analysis of weights; no data rows are used.

Run: uv run python scripts/basis_capture.py   (~4 min; --cached replots from runs/g1/basis_capture.pt)
Writes reports/figures/g1/basis_capture.png and prints the table.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from quickgrok.accel import (
    captured_energy,
    fourier_basis,
    levels_basis,
    self_expr_basis,
    svd_basis,
    svd_vectors,
)
from quickgrok.models import GrokMLP, load_model
from quickgrok.plot_compare import ARM_COLORS
from quickgrok.plot_pilot import INK, MUTED, find_grok_step, load_history, style_axes

TASKS: dict[str, list[Path]] = {
    "add": [Path(f"runs/t6/vanilla/seed{s}") for s in (0, 1, 2)],
    "mul": [Path(f"runs/g1/mul_vanilla/seed{s}") for s in (100, 101, 102)],
    "s5 (train 0.5)": [
        Path(f"runs/g1/s5_vanilla_tf0.5/seed{s}") for s in (100, 101, 102)
    ],
}
BASES = ["fourier", "svd_global", "levels_2", "levels_3", "self_expr"]
M = 9  # constant + 8 directions, as in the substitution hook
OUT = Path("reports/figures/g1/basis_capture.png")
CACHE = Path("runs/g1/basis_capture.pt")  # computed captures, for --cached replots


class RunCapture(TypedDict):
    steps: list[int]
    grok: int | None
    # basis -> one array of per-column captures (2 * width,) per checkpoint
    captures: dict[str, list[np.ndarray]]


def capture_fourier(W: Tensor) -> Tensor:
    """Capture (d,) of each column of W by the harmonics of its own best base frequency."""
    cols = W.T
    basis, _ = fourier_basis(cols, cols, M)
    return captured_energy((basis, basis), cols, cols)


def capture_svd(W: Tensor, vectors: Tensor) -> Tensor:
    """Capture (d,) of each column of W by its best 8 of the shared singular `vectors`."""
    cols = W.T
    basis = svd_basis(cols, cols, vectors, M)
    return captured_energy((basis, basis), cols, cols)


def capture_levels(W: Tensor, n_levels: int) -> Tensor:
    """Capture (d,) of each column by its optimal 1-D k-means into `n_levels` groups (2 or 3)."""
    cols = W.T
    basis = levels_basis(cols, n_levels)
    return captured_energy((basis, basis), cols, cols)


def capture_self_expr(W: Tensor) -> Tensor:
    """Capture (d,) of each column by 8 other columns of W chosen by greedy OMP."""
    cols = W.T
    basis = self_expr_basis(W, list(range(W.shape[1])), M - 1)
    return captured_energy((basis, basis), cols, cols)


def capture_all(U: Tensor, V: Tensor) -> dict[str, Tensor]:
    """Per-column captures (2d,) of u columns then v columns, for every basis."""
    vectors = svd_vectors(U, V)
    makers: dict[str, Callable[[Tensor], Tensor]] = {
        "fourier": capture_fourier,
        "svd_global": lambda W: capture_svd(W, vectors),
        "levels_2": lambda W: capture_levels(W, 2),
        "levels_3": lambda W: capture_levels(W, 3),
        "self_expr": capture_self_expr,
    }
    return {name: torch.cat([f(U), f(V)]) for name, f in makers.items()}


def capture_run(run_dir: Path) -> RunCapture:
    """Captures at every checkpoint of one run, plus its step to 99% test accuracy."""
    result: RunCapture = {
        "steps": [],
        "grok": find_grok_step(load_history(run_dir)),
        "captures": {b: [] for b in BASES},
    }
    for path in sorted(run_dir.glob("step_*.pt")):
        model, ckpt = load_model(path)
        assert isinstance(model, GrokMLP)
        U = model.embed_a.weight.detach().double()
        V = model.embed_b.weight.detach().double()
        result["steps"].append(ckpt["step"])
        for name, cap in capture_all(U, V).items():
            result["captures"][name].append(cap.numpy())
    return result


def gaussian_null(n: int, d: int, seed: int = 0) -> dict[str, float]:
    """Median capture per basis on Gaussian U, V of shape (n, d)."""
    g = torch.Generator().manual_seed(seed)
    U, V = (torch.randn(n, d, generator=g, dtype=torch.float64) for _ in "uv")
    return {k: float(np.nanmedian(v.numpy())) for k, v in capture_all(U, V).items()}


def at_step(run: RunCapture, basis: str, step: int) -> np.ndarray:
    """Per-column captures at the checkpoint nearest to `step`."""
    j = int(np.argmin(np.abs(np.array(run["steps"]) - step)))
    return run["captures"][basis][j]


def skill(capture: float, null: float) -> float:
    """(capture - null) / (1 - null)."""
    return (capture - null) / (1 - null)


def print_table(
    runs: dict[str, list[RunCapture]], nulls: dict[str, dict[str, float]]
) -> None:
    """Median capture (skill) per task and basis at step 0, 50% and 80% of grok, and the end.

    Each cell is the median over seeds of the median over columns.
    """
    print("| Task | Basis | Gaussian null | Step 0 | 50% of grok | 80% of grok | End |")
    print("|---|---|---|---|---|---|---|")
    for task, task_runs in runs.items():
        for basis in BASES:
            null = nulls[task][basis]
            cells = []
            for frac in (0.0, 0.5, 0.8, None):
                meds = [
                    float(
                        np.nanmedian(
                            r["captures"][basis][-1]
                            if frac is None
                            else at_step(r, basis, int(frac * (r["grok"] or 0)))
                        )
                    )
                    for r in task_runs
                ]
                med = float(np.median(meds))
                cells.append(f"{med:.2f} ({skill(med, null):+.2f})")
            print(f"| {task} | {basis} | {null:.2f} | " + " | ".join(cells) + " |")
    print("\nCells: raw capture (skill vs the Gaussian null).")


def plot(runs: dict[str, list[RunCapture]], nulls: dict[str, dict[str, float]]) -> None:
    """Rows: tasks. Columns: bases. Median and 75th percentile over columns, seeds pooled."""
    fig, axes = plt.subplots(
        len(runs),
        len(BASES),
        figsize=(3.2 * len(BASES), 2.7 * len(runs)),
        sharey=True,
        squeeze=False,
    )
    for row, (task, task_runs) in enumerate(runs.items()):
        for col, basis in enumerate(BASES):
            ax = axes[row, col]
            steps = sorted({s for r in task_runs for s in r["steps"]})
            for q, ls, label in [(50, "-", "median"), (75, "--", "75th pct")]:
                values = [
                    np.nanpercentile(
                        np.concatenate(
                            [
                                r["captures"][basis][r["steps"].index(s)]
                                for r in task_runs
                                if s in r["steps"]
                            ]
                        ),
                        q,
                    )
                    for s in steps
                ]
                ax.plot(steps, values, color=ARM_COLORS[0], lw=1.8, ls=ls, label=label)
            step0 = float(np.nanmedian(task_runs[0]["captures"][basis][0]))
            ax.axhline(step0, color=MUTED, lw=2.5, alpha=0.5, label="step-0 median")
            ax.axhline(
                nulls[task][basis],
                color=ARM_COLORS[1],
                lw=1.2,
                ls=":",
                label="Gaussian null (median)",
            )
            for r in task_runs:
                if r["grok"] is not None:
                    ax.axvline(r["grok"], color=MUTED, lw=0.8, ls="--")
            ax.set_ylim(0, 1)
            style_axes(ax)
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(basis, fontsize=10, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(f"{task}\ncaptured energy", color=MUTED, fontsize=9)
            if row == len(runs) - 1:
                ax.set_xlabel("training step", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        "Share of each input column captured per basis (columns of all seeds pooled; "
        "dashed verticals: each seed's step to 99% test acc)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)


def main() -> None:
    """Compute captures for every run (or load them with --cached), print the table, plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true", help=f"reuse {CACHE}")
    args = parser.parse_args()
    if args.cached:
        runs, nulls = torch.load(CACHE, weights_only=False)
    else:
        runs = {task: [capture_run(d) for d in dirs] for task, dirs in TASKS.items()}
        nulls: dict[str, dict[str, float]] = {}
        for task in runs:
            model, _ = load_model(next(TASKS[task][0].glob("step_*.pt")))
            assert isinstance(model, GrokMLP)
            n, d = model.embed_a.weight.shape
            nulls[task] = gaussian_null(n, d)
        torch.save((runs, nulls), CACHE)
    print_table(runs, nulls)
    print("\nGaussian nulls:", json.dumps(nulls, indent=1))
    plot(runs, nulls)


if __name__ == "__main__":
    main()
