"""Generic substitution bases (G1, G1b, self_expr look): does the method work without the Fourier prior?

Report: reports/generic_basis.md.
  uv run python experiments/generic_basis.py run       # 9 vanilla runs + 7 single-seed looks
  uv run python experiments/generic_basis.py figures   # -> reports/figures/g1/ (~4 min of CPU)
The report's GPU time, 10 in parallel on an RTX 5080: vanilla + G1 runs 4.6 min, levels ~4 min,
self_expr ~10 min (s5 is the slowest). The add references are runs of other experiments:
runs/accel_probe/seed100 (experiments/accel_probe.py) and runs/t6/vanilla/seed0-2
(experiments/substitution.py).

`figures` also runs the offline capture analysis (G1b). For every checkpoint of the vanilla MLP
runs of add, mul and s5, and every input column (u_i and v_i separately, mean removed), it
measures the captured share of its energy by:
  fourier     harmonics of the column's best base frequency, 8 directions (knows cyclicity)
  svd_global  the 8 left singular vectors of the mean-removed [U, V] that capture it most
  levels_L    optimal 1-D k-means of the column's values into L groups (L - 1 directions)
  self_expr   greedy orthogonal matching pursuit: 8 other columns of the same matrix
Nulls: the step-0 checkpoint and Gaussian matrices of the same shape. Skill = (capture - null) /
(1 - null) with the Gaussian null's median. Full-grid analysis of weights; no data rows are used.
It writes reports/figures/g1/basis_capture.png and prints the table.
"""

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
from quickgrok.launch import main, make_jobs
from quickgrok.models import GrokMLP, load_model
from quickgrok.plot_compare import ARM_COLORS, compare
from quickgrok.plot_pilot import INK, MUTED, find_grok_step, load_history, style_axes

COMMON = "--model mlp --probe-every 500 --stop-after-grok 3000"

# Vanilla regimes: runs/g1/<arm>/seed<s>.
VANILLA_SEEDS = [100, 101, 102]
VANILLA = {
    "mul_vanilla": "--op mul --train-frac 0.3 --steps 60000",
    "s5_vanilla": "--op s5 --train-frac 0.3 --steps 60000",
    "s5_vanilla_tf0.5": "--op s5 --train-frac 0.5 --steps 60000",
}

# Substitution looks, seed 100: runs/g1/<arm>/seed100.
SUB = "--substitute module"
GENERIC = "--tau-energy 0.8 --tau-r2 0.8"
LEVELS = "--basis levels_3 --tau-energy 0.975 --tau-r2 0.8"
SELF_EXPR = f"--basis self_expr {GENERIC}"
LOOKS = {
    "add_module_auto": f"--op add --steps 40000 {SUB} --basis auto {GENERIC}",
    "add_module_svd": f"--op add --steps 40000 {SUB} --basis svd {GENERIC}",
    "add_module_levels3": f"--op add --steps 40000 {SUB} {LEVELS}",
    "mul_module_levels3": f"--op mul --steps 60000 {SUB} {LEVELS}",
    "add_module_self_expr": f"--op add --steps 40000 {SUB} {SELF_EXPR}",
    "mul_module_self_expr": f"--op mul --steps 60000 {SUB} {SELF_EXPR}",
    "s5_module_self_expr": f"--op s5 --train-frac 0.5 --steps 60000 {SUB} {SELF_EXPR}",
}

JOBS = make_jobs("g1", VANILLA, VANILLA_SEEDS, COMMON) + make_jobs(
    "g1", LOOKS, [100], COMMON
)

TASKS: dict[str, list[Path]] = {
    "add": [Path(f"runs/t6/vanilla/seed{s}") for s in (0, 1, 2)],
    "mul": [Path(f"runs/g1/mul_vanilla/seed{s}") for s in (100, 101, 102)],
    "s5 (train 0.5)": [
        Path(f"runs/g1/s5_vanilla_tf0.5/seed{s}") for s in (100, 101, 102)
    ],
}
BASES = ["fourier", "svd_global", "levels_2", "levels_3", "self_expr"]
M = 9  # constant + 8 directions, as in the substitution hook
FIG = Path("reports/figures/g1")
OUT = FIG / "basis_capture.png"


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


def basis_capture() -> None:
    """Compute captures for every run, print the table and plot."""
    runs = {task: [capture_run(d) for d in dirs] for task, dirs in TASKS.items()}
    nulls: dict[str, dict[str, float]] = {}
    for task in runs:
        model, _ = load_model(next(TASKS[task][0].glob("step_*.pt")))
        assert isinstance(model, GrokMLP)
        n, d = model.embed_a.weight.shape
        nulls[task] = gaussian_null(n, d)
    print_table(runs, nulls)
    print("\nGaussian nulls:", json.dumps(nulls, indent=1))
    plot(runs, nulls)


def figures() -> None:
    """Vanilla regimes of mul and s5, the single-seed looks, and the capture analysis."""

    def g1(arm: str, seeds: list[int]) -> list[Path]:
        """Run directories of a g1 arm."""
        return [Path(f"runs/g1/{arm}/seed{s}") for s in seeds]

    compare({"vanilla_tf0.3": g1("mul_vanilla", VANILLA_SEEDS)}, FIG / "mul")
    compare(
        {
            "vanilla_tf0.3": g1("s5_vanilla", VANILLA_SEEDS),
            "vanilla_tf0.5": g1("s5_vanilla_tf0.5", VANILLA_SEEDS),
        },
        FIG / "s5",
    )
    compare(
        {
            "vanilla": [Path("runs/accel_probe/seed100")],
            "levels_3": g1("add_module_levels3", [100]),
            "self_expr": g1("add_module_self_expr", [100]),
        },
        FIG / "looks_add",
    )
    compare(
        {
            "vanilla": g1("mul_vanilla", [100]),
            "levels_3": g1("mul_module_levels3", [100]),
            "self_expr": g1("mul_module_self_expr", [100]),
        },
        FIG / "looks_mul",
    )
    compare(
        {
            "vanilla": g1("s5_vanilla_tf0.5", [100]),
            "self_expr": g1("s5_module_self_expr", [100]),
        },
        FIG / "looks_s5",
    )
    basis_capture()


if __name__ == "__main__":
    main({"run": JOBS}, {"figures": figures})
