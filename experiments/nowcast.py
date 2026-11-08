"""Online weight nowcasting: forecast the weights from their own history and jump there.

Report: reports/nowcast.md.
  uv run python experiments/nowcast.py run       # 15 tuning + 45 final runs
  uv run python experiments/nowcast.py figures   # -> reports/figures/nowcast/
The vanilla arm of the figures is runs/t6/vanilla (run it with experiments/substitution.py).
In the nowcast diagnostics figure, arms without nowcast.jsonl (vanilla) only appear in
the weight-norm panels.
"""

import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quickgrok.launch import main, make_jobs
from quickgrok.nowcast import NowcastRecord
from quickgrok.plot_compare import ARM_COLORS, compare, stack_metric
from quickgrok.plot_pilot import INK, MUTED, find_grok_step, load_history, style_axes

# Every run: runs/nowcast/<arm>/seed<s>. --nowcast-every 1000, --nowcast-k 8,
# --nowcast-max-loss-ratio 1.5 are the defaults.
COMMON = "--model mlp --steps 40000 --stop-after-grok 3000 --save-every 5000"
LINEAR = "--nowcast linear --nowcast-horizon 1000"
LSTM = "--nowcast lstm --nowcast-horizon 1000"

TUNE_SEEDS = [100, 101, 102, 103, 104]
TUNE = {
    "tune_linear_h1000": LINEAR,
    "tune_linear_h3000": "--nowcast linear --nowcast-horizon 3000",
    "tune_lstm_h1000": LSTM,
}

SEEDS = list(range(10))
ARMS = {
    "linear": LINEAR,
    "lstm": LSTM,
    "linear_norm_only": f"{LINEAR} --nowcast-norm-only",
    "shadow_linear": f"{LINEAR} --nowcast-max-loss-ratio 0",
}
SHADOW_LSTM_SEEDS = [0, 1, 2, 3, 4]
SHADOW_LSTM = {"shadow_lstm": f"{LSTM} --nowcast-max-loss-ratio 0"}

JOBS = (
    make_jobs("nowcast", TUNE, TUNE_SEEDS, COMMON)
    + make_jobs("nowcast", ARMS, SEEDS, COMMON)
    + make_jobs("nowcast", SHADOW_LSTM, SHADOW_LSTM_SEEDS, COMMON)
)


def load_nowcast(run_dir: Path) -> list[NowcastRecord]:
    """Read the jump log written by `NowcastHook`."""
    lines = (run_dir / "nowcast.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def by_step(
    runs: list[list[NowcastRecord]], key: Literal["accepted", "skill"]
) -> tuple[list[int], list[float]]:
    """Attempt steps and the mean of a record field over the runs that attempted at that step.

    For "skill" only clean backtests count (steps without one are dropped) and the mean is a
    median.
    """
    steps = sorted({r["step"] for run in runs for r in run})
    xs, ys = [], []
    for s in steps:
        rows = [r for run in runs for r in run if r["step"] == s]
        if key == "skill":
            vals = [
                r["skill"] for r in rows if r["skill"] is not None and r["skill_clean"]
            ]
            if vals:
                xs.append(s)
                ys.append(float(np.median(vals)))
        else:
            xs.append(s)
            ys.append(float(np.mean([r["accepted"] for r in rows])))
    return xs, ys


def plot_nowcast(arms: dict[str, list[Path]], out: Path) -> None:
    """Write the four-panel nowcast figure of `arms` (name -> run dirs) to `out`."""
    fig, (ax_acc, ax_skill, ax_norm, ax_vs) = plt.subplots(4, 1, figsize=(9, 13.5))
    ax_skill.sharex(ax_acc)
    ax_norm.sharex(ax_acc)
    for i, (name, dirs) in enumerate(arms.items()):
        color = ARM_COLORS[i % len(ARM_COLORS)]
        hists = [load_history(d) for d in dirs]
        groks = [g for g in (find_grok_step(h) for h in hists) if g is not None]
        steps, norms = stack_metric(hists, "weight_norm")
        ax_norm.plot(steps, np.median(norms, axis=0), color=color, lw=2, label=name)
        for run in norms:
            ax_norm.plot(steps, run, color=color, lw=0.6, alpha=0.25)
        for j, h in enumerate(hists):
            ax_vs.plot(
                [m["weight_norm"] for m in h],
                [m["val_acc"] for m in h],
                color=color,
                lw=0.8,
                alpha=0.6,
                label=name if j == 0 else None,
            )
        axes = [ax_norm]
        if all((d / "nowcast.jsonl").exists() for d in dirs):
            runs = [load_nowcast(d) for d in dirs]
            ax_acc.plot(
                *by_step(runs, "accepted"),
                color=color,
                lw=2,
                marker="o",
                ms=3,
                label=name,
            )
            for run in runs:
                pts = [(r["step"], r["skill"]) for r in run if r["skill"] is not None]
                clean = [r["skill_clean"] for r in run if r["skill"] is not None]
                for (s, v), ok in zip(pts, clean):
                    ax_skill.scatter(
                        s,
                        v,
                        s=10,
                        color=color if ok else "none",
                        edgecolor=color,
                        linewidth=0.6,
                        alpha=0.5,
                    )
            if not any(
                r["accepted"] for run in runs for r in run
            ):  # all backtests clean
                ax_skill.plot(*by_step(runs, "skill"), color=color, lw=2, label=name)
            axes += [ax_acc, ax_skill]
        if groks:
            for ax in axes:
                ax.axvline(float(np.median(groks)), color=color, ls="--", lw=1)
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_ylabel("fraction of seeds\nwith jump accepted", color=MUTED, fontsize=9)
    ax_acc.set_title(
        "Accepted jumps per attempt (dashed: median step to 99% test acc)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax_skill.axhline(1, color=MUTED, lw=1, ls=":")
    ax_skill.set_yscale("log")
    ax_skill.set_ylabel("forecast MSE / no-change MSE", color=MUTED, fontsize=9)
    ax_skill.set_title(
        "Forecast skill, backtest on the realised trajectory (< 1: beats no-change; "
        "filled: clean, hollow: window contains an accepted jump; line: median, for arms whose jumps are all rejected)",
        fontsize=9,
        color=INK,
        loc="left",
        wrap=True,
    )
    ax_norm.set_ylabel("L2 norm of all weights", color=MUTED, fontsize=9)
    ax_norm.set_xlabel("training step", color=MUTED, fontsize=9)
    ax_norm.set_ylim(top=180)
    ax_norm.set_title(
        "Weight norm (bold: median, thin: seeds)", fontsize=10, color=INK, loc="left"
    )
    ax_vs.set_xlim(
        160, ax_vs.get_xlim()[0]
    )  # norm falls during training: read left to right
    ax_vs.set_xlabel("L2 norm of all weights (decreasing →)", color=MUTED, fontsize=9)
    ax_vs.set_ylabel("test accuracy", color=MUTED, fontsize=9)
    ax_vs.set_title(
        "Test accuracy against weight norm (one line per seed)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    ax_vs.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in (ax_acc, ax_skill, ax_norm, ax_vs):
        style_axes(ax)
    ax_acc.legend(fontsize=8, loc="center right", framealpha=0.95, edgecolor="none")
    ax_skill.legend(frameon=False, fontsize=8, loc="upper right")
    ax_norm.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figures() -> None:
    """Compare the final arms with t6 vanilla and draw the nowcast diagnostics."""
    arms = {"vanilla": [Path(f"runs/t6/vanilla/seed{s}") for s in SEEDS]}
    for arm in ["shadow_linear", "linear", "lstm", "linear_norm_only"]:
        arms[arm] = [Path(f"runs/nowcast/{arm}/seed{s}") for s in SEEDS]
    out = Path("reports/figures/nowcast")
    compare(arms, out)
    arms["shadow_lstm"] = [
        Path(f"runs/nowcast/shadow_lstm/seed{s}") for s in SHADOW_LSTM_SEEDS
    ]
    plot_nowcast(arms, out / "nowcast.png")


if __name__ == "__main__":
    main({"run": JOBS}, {"figures": figures})
