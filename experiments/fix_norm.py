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
    """Left: total weight norm over steps per arm; right: test accuracy against weight norm."""
    fig, (ax_norm, ax_vs) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (arm, dirs) in enumerate(runs.items()):
        color = ARM_COLORS[i]
        hists = [load_history(d) for d in dirs]
        steps, norms = stack_metric(hists, "weight_norm")
        for run in norms:
            ax_norm.plot(steps, run, color=color, lw=0.6, alpha=0.3)
        ax_norm.plot(steps, np.median(norms, axis=0), color=color, lw=2, label=arm)
        for j, h in enumerate(hists):
            ax_vs.plot(
                [m["weight_norm"] for m in h],
                [m["val_acc"] for m in h],
                color=color,
                lw=0.8,
                alpha=0.6,
                label=arm if j == 0 else None,
            )
    ax_norm.set_xscale("log")
    ax_norm.set_xlabel("training step (log)", color=MUTED, fontsize=9)
    ax_norm.set_ylabel("total L2 weight norm", color=MUTED, fontsize=9)
    ax_norm.set_title("Weight norm (thin: seeds)", fontsize=10, color=INK, loc="left")
    ax_vs.set_xlabel("total L2 weight norm", color=MUTED, fontsize=9)
    ax_vs.set_ylabel("test accuracy", color=MUTED, fontsize=9)
    ax_vs.set_title("Test accuracy against norm", fontsize=10, color=INK, loc="left")
    ax_vs.invert_xaxis()  # training moves right as the norm shrinks
    for ax in (ax_norm, ax_vs):
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
