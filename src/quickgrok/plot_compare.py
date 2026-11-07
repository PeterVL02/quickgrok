"""Compare training runs grouped into arms (e.g. vanilla vs substitution), several seeds per arm.

Run: uv run python -m quickgrok.plot_compare --arm vanilla runs/mlp_baseline/seed* \
         --out reports/figures/mlp_baseline
Writes curves.png (loss and accuracy per arm), grok_steps.png (steps to 99% test accuracy per
seed) and summary.md (median and IQR table).
"""

import argparse
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quickgrok.plot_pilot import (
    GRID,
    INK,
    MUTED,
    TRAIN,
    VAL,
    find_grok_step,
    load_history,
    style_axes,
)
from quickgrok.train import Metrics

# Reference categorical palette (dataviz skill), fixed order: arm i always gets slot i.
ARM_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]


class ArmSummary(TypedDict):
    arm: str
    n_runs: int
    memorised: list[int | None]  # first step with train acc >= 99%, per run
    grok95: list[int | None]  # first step with test acc >= 95%, per run
    grok99: list[int | None]  # first step with test acc >= 99%, per run


def find_memorise_step(history: list[Metrics], threshold: float = 0.99) -> int | None:
    """First logged step with train accuracy >= threshold, or None."""
    return next((m["step"] for m in history if m["train_acc"] >= threshold), None)


def summarise_arm(arm: str, histories: list[list[Metrics]]) -> ArmSummary:
    """Per-run milestone steps for one arm."""
    return {
        "arm": arm,
        "n_runs": len(histories),
        "memorised": [find_memorise_step(h) for h in histories],
        "grok95": [find_grok_step(h, 0.95) for h in histories],
        "grok99": [find_grok_step(h, 0.99) for h in histories],
    }


def describe(steps: list[int | None]) -> str:
    """'median [q25–q75] (k/n)' over runs that reached the milestone."""
    done = [s for s in steps if s is not None]
    if not done:
        return f"— (0/{len(steps)})"
    q25, med, q75 = np.quantile(done, [0.25, 0.5, 0.75])
    return f"{med:,.0f} [{q25:,.0f}–{q75:,.0f}] ({len(done)}/{len(steps)})"


def write_summary(summaries: list[ArmSummary], out: Path) -> str:
    """Markdown table of milestone steps per arm; also written to `out`."""
    lines = [
        "| Arm | Runs | Train acc ≥ 99% | Test acc ≥ 95% | Test acc ≥ 99% |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['arm']} | {s['n_runs']} | {describe(s['memorised'])} | "
            f"{describe(s['grok95'])} | {describe(s['grok99'])} |"
        )
    lines.append("")
    lines.append(
        "Cells: median [IQR] step over runs that reached the milestone (reached/total)."
    )
    table = "\n".join(lines)
    out.write_text(table + "\n", encoding="utf-8")
    return table


def stack_metric(
    histories: list[list[Metrics]], key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Steps (T,) and values (runs, T), truncated to the shortest run; step 0 dropped for log x."""
    n = min(len(h) for h in histories)
    steps = np.array([m["step"] for m in histories[0][:n]])
    values = np.array([[m[key] for m in h[:n]] for h in histories])  # type: ignore[literal-required]
    keep = steps > 0
    return steps[keep], values[:, keep]


def plot_curves(
    arms: list[str], histories: list[list[list[Metrics]]], out: Path
) -> None:
    """Rows: loss, accuracy. Columns: arms. Thin lines per seed, bold median of test."""
    fig, axes = plt.subplots(
        2, len(arms), figsize=(4.5 * len(arms), 6.5), sharex=True, sharey="row"
    )
    axes = axes.reshape(2, len(arms))
    for col, (arm, hists) in enumerate(zip(arms, histories)):
        for row, (train_key, test_key) in enumerate(
            [("train_loss", "val_loss"), ("train_acc", "val_acc")]
        ):
            ax = axes[row, col]
            for key, color, label in [
                (train_key, TRAIN, "train"),
                (test_key, VAL, "test"),
            ]:
                steps, vals = stack_metric(hists, key)
                for run in vals:
                    ax.plot(steps, run, color=color, lw=0.8, alpha=0.35)
                ax.plot(
                    steps,
                    np.median(vals, axis=0),
                    color=color,
                    lw=2,
                    label=f"{label} (median)",
                )
            ax.set_xscale("log")
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(
                    f"{arm} ({len(hists)} seeds)", fontsize=10, color=INK, loc="left"
                )
            style_axes(ax)
        axes[1, col].set_xlabel("training step (log)", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel("cross-entropy loss (log)", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("accuracy", color=MUTED, fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=8, loc="center left")
    fig.suptitle(
        "Train and test curves per arm (thin: individual seeds)",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grok_steps(summaries: list[ArmSummary], out: Path) -> None:
    """Dot plot: one row per arm, one dot per seed at its step to 99% test acc; bar = median."""
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.7 * len(summaries)))
    for i, s in enumerate(summaries):
        color = ARM_COLORS[i % len(ARM_COLORS)]
        done = [x for x in s["grok99"] if x is not None]
        y = len(summaries) - 1 - i
        ax.scatter(
            done,
            [y] * len(done),
            s=60,
            color=color,
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        if done:
            med = float(np.median(done))
            ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=2, zorder=4)
            ax.annotate(
                f"median {med:,.0f}",
                (med, y + 0.32),
                fontsize=8,
                color=INK,
                ha="center",
                va="bottom",
            )
        missing = s["n_runs"] - len(done)
        if missing:
            ax.annotate(
                f"{missing} run(s) never reached 99%",
                (1.0, y),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                color=MUTED,
                ha="right",
                va="center",
            )
    ax.set_yticks(range(len(summaries)), [s["arm"] for s in reversed(summaries)])
    ax.set_ylim(-0.6, len(summaries) - 0.2)
    ax.set_xlim(left=0)
    ax.set_xlabel("steps until test accuracy ≥ 99%", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_title(
        "Grokking step per seed (bar: median)", fontsize=11, color=INK, loc="left"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse arms, make the comparison figures and print the summary table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        action="append",
        required=True,
        metavar=("NAME", "RUN_DIR"),
        help="arm name followed by its run directories; repeat for more arms",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arms = [a[0] for a in args.arm]
    histories = [[load_history(Path(d)) for d in a[1:]] for a in args.arm]
    summaries = [summarise_arm(n, h) for n, h in zip(arms, histories)]
    plot_curves(arms, histories, args.out / "curves.png")
    plot_grok_steps(summaries, args.out / "grok_steps.png")
    print(write_summary(summaries, args.out / "summary.md"))


if __name__ == "__main__":
    main()
