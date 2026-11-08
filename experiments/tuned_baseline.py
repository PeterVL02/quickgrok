"""Tuned baseline: how grokking time of the vanilla MLP depends on learning rate and weight decay.

Report: reports/tuned_baseline.md.
  uv run python experiments/tuned_baseline.py run       # 16 settings x 5 seeds = 80 runs
  uv run python experiments/tuned_baseline.py figures   # -> reports/figures/tuned_baseline/
The figure plots grok step against lr x weight decay (AdamW's per-step decay).
"""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quickgrok.launch import main, make_jobs
from quickgrok.plot_compare import ARM_COLORS
from quickgrok.plot_pilot import INK, MUTED, find_grok_step, style_axes

# Runs: runs/tuned_vanilla/lr<lr>_wd<wd>/seed<s>. Not every pair of the grid was run.
COMMON = "--model mlp --steps 40000 --stop-after-grok 3000"
SEEDS = [100, 101, 102, 103, 104]
GRID = {  # weight decay -> learning rates
    "0.5": ["4e-3", "8e-3", "1.6e-2"],
    "1.0": ["1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2"],
    "2.0": ["1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2"],
    "4.0": ["1e-3", "2e-3", "4e-3"],
}
ARMS = {
    f"lr{lr}_wd{wd}": f"--lr {lr} --weight-decay {wd}"
    for wd, lrs in GRID.items()
    for lr in lrs
}
JOBS = make_jobs("tuned_vanilla", ARMS, SEEDS, COMMON)

RUNS = Path("runs/tuned_vanilla")
OUT = Path("reports/figures/tuned_baseline")


def parse_config(name: str) -> tuple[float, float]:
    """'lr4e-3_wd2.0' -> (0.004, 2.0)."""
    lr, wd = name.removeprefix("lr").split("_wd")
    return float(lr), float(wd)


def collect() -> dict[float, list[tuple[float, int | None]]]:
    """Per weight decay: (lr, median grok step or None if under 5/5 seeds grokked), sorted by lr."""
    table: dict[float, list[tuple[float, int | None]]] = {}
    for cfg in sorted(RUNS.iterdir()):
        lr, wd = parse_config(cfg.name)
        steps = [
            find_grok_step(json.loads((run / "metrics.json").read_text()))
            for run in sorted(cfg.iterdir())
        ]
        done = [s for s in steps if s is not None]
        median = int(statistics.median(done)) if len(done) == len(steps) else None
        table.setdefault(wd, []).append((lr, median))
    return {wd: sorted(rows) for wd, rows in sorted(table.items())}


def figures() -> None:
    """Plot grok step vs lr*wd per weight decay, with the 1/(lr*wd) reference line."""
    OUT.mkdir(parents=True, exist_ok=True)
    table = collect()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (wd, rows) in enumerate(table.items()):
        pts = [(lr * wd, s) for lr, s in rows if s is not None]
        if not pts:
            continue  # e.g. wd 4: not every seed grokked at any lr
        ax.plot(
            [x for x, _ in pts],
            [y for _, y in pts],
            "o-",
            color=ARM_COLORS[i],
            lw=2,
            ms=8,
            mec="white",
            mew=2,
            label=f"weight decay {wd:g}",
        )
    xs = [1e-3 * 2**j for j in range(6)]
    ax.plot(
        xs,
        [17_900 * 1e-3 / x for x in xs],
        "--",
        color=MUTED,
        lw=1,
        label="∝ 1 / (lr · wd)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate × weight decay (per-step decay)", color=MUTED)
    ax.set_ylabel("steps to 99% test accuracy (median, 5 seeds)", color=MUTED)
    ax.annotate(
        "our baseline\n(lr 1e-3, wd 1)",
        (1e-3, 17_900),
        (1.05e-3, 2_500),
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        "Vanilla MLP: grokking time falls roughly like 1 / (lr · wd); lr matters most",
        fontsize=11,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "grok_vs_decay.png", dpi=150)
    plt.close(fig)
    lines = [
        "| lr | wd | lr·wd | Steps to 99% (median, 5 seeds) |",
        "| --- | --- | --- | --- |",
    ]
    for wd, rows in table.items():
        for lr, s in rows:
            lines.append(
                f"| {lr:g} | {wd:g} | {lr * wd:g} | {s if s else 'not all grokked'} |"
            )
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main({"run": JOBS}, {"figures": figures})
