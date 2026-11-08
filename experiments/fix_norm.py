"""Pin the total weight norm at its grok value (124) from step 1000: lever or readout?

Pre-registration: reports/preregistration_fix_norm.md. Report: reports/fix_norm.md.
  uv run python experiments/fix_norm.py run       # 20 runs, runs/fix_norm/<arm>/seed<s>
  uv run python experiments/fix_norm.py figures   # -> reports/figures/fix_norm/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from quickgrok.launch import main, make_jobs
from quickgrok.plot_compare import ARM_COLORS, compare, stack_metric
from quickgrok.plot_pilot import INK, MUTED, find_grok_step, load_history, style_axes

CAP = 20_000
COMMON = f"--model mlp --steps {CAP} --stop-after-grok 1000 --save-every 5000"
SEEDS = list(range(10))
ARMS = {"plain": "", "pinned": "--fix-norm 124 --fix-norm-from 1000"}
JOBS = make_jobs("fix_norm", ARMS, SEEDS, COMMON)
OUT = Path("reports/figures/fix_norm")


def plot_norm(runs: dict[str, list[Path]], out: Path) -> None:
    """Total weight norm (left) and test accuracy (right) over steps, one colour per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for i, (arm, dirs) in enumerate(runs.items()):
        hists = [load_history(d) for d in dirs]
        for ax, key in zip(axes, ["weight_norm", "val_acc"]):
            steps, vals = stack_metric(hists, key)
            for run in vals:
                ax.plot(steps, run, color=ARM_COLORS[i], lw=0.6, alpha=0.3)
            ax.plot(
                steps, np.median(vals, axis=0), color=ARM_COLORS[i], lw=2, label=arm
            )
    axes[0].axhline(124, color=MUTED, lw=1, ls="--")
    axes[0].text(110, 126, "plain grok norm ≈ 124", color=MUTED, fontsize=8)
    for ax, title in zip(axes, ["Total L2 weight norm", "Test accuracy"]):
        ax.set_xscale("log")
        ax.set_xlabel("training step (log)", color=MUTED, fontsize=9)
        ax.set_title(f"{title} (thin: seeds)", fontsize=10, color=INK, loc="left")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_decision(runs: dict[str, list[Path]], out: Path) -> str:
    """The pre-registered statistics: censored medians, Mann-Whitney p, memorisation check."""
    steps, held = {}, {}
    for arm, dirs in runs.items():
        hists = [load_history(d) for d in dirs]
        steps[arm] = [find_grok_step(h, 0.99) or CAP + 100 for h in hists]
        held[arm] = sum(
            all(m["train_acc"] >= 0.99 for m in h if m["step"] >= 2000) for h in hists
        )
    plain, pinned = np.median(steps["plain"]), np.median(steps["pinned"])
    p = mannwhitneyu(steps["plain"], steps["pinned"], alternative="two-sided").pvalue
    text = "\n".join(
        [
            "| Arm | Steps to 99% per seed | Median [IQR] | Seeds memorised from step 2000 |",
            "|---|---|---|---|",
            *(
                f"| {arm} | {', '.join(f'{s:,}' for s in v)} | {np.median(v):,.0f} "
                f"[{np.quantile(v, 0.25):,.0f}–{np.quantile(v, 0.75):,.0f}] | {held[arm]}/10 |"
                for arm, v in steps.items()
            ),
            "",
            (
                f"pinned / plain median = {pinned / plain:.3f}; two-sided Mann–Whitney "
                f"p = {p:.2g}. Non-grokking runs count as {CAP + 100:,}."
            ),
        ]
    )
    out.write_text(text + "\n", encoding="utf-8")
    return text


def figures() -> None:
    """Compare the two arms, draw the norm figure and write the pre-registered statistics."""
    runs = {arm: [Path(f"runs/fix_norm/{arm}/seed{s}") for s in SEEDS] for arm in ARMS}
    compare(runs, OUT)
    plot_norm(runs, OUT / "norm.png")
    print(write_decision(runs, OUT / "decision.md"))


if __name__ == "__main__":
    main({"run": JOBS}, {"figures": figures})
