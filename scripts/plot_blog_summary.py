"""Summary figure for the README: steps to 99% test accuracy per method and baseline setting.

The numbers are medians copied from the reports (no run data needed):
  lr 1e-3 / wd 1   reports/substitution.md, reports/nowcast.md (10 seeds)
  lr 2e-3 / wd 1   reports/tuned_baseline.md (5 seeds)
  lr 4e-3 / wd 2   reports/tuned_retest.md (10 seeds)
  lr 1.6e-2 / wd 2 reports/tuned_retest.md, secondary check (5 seeds)

Run: uv run python scripts/plot_blog_summary.py  (writes reports/figures/blog/summary.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "blog"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SETTINGS = [
    "lr 1e-3, wd 1\n(untuned)",
    "lr 2e-3, wd 1",
    "lr 4e-3, wd 2\n(tuned)",
    "lr 1.6e-2, wd 2\n(fastest)",
]
# method -> (colour, marker, median steps per setting; None = not run)
METHODS: dict[str, tuple[str, str, list[int | None]]] = {
    "plain training": ("#2a78d6", "o", [18400, 8700, 2920, 920]),
    "nowcasting": ("#eb6834", "s", [11150, None, 1840, 820]),
    "wave modules": ("#1baf7a", "D", [10800, None, 1620, 700]),
    "both": ("#eda100", "^", [None, None, 1420, None]),
}


def main() -> None:
    """Draw one row per baseline setting, one marker per method, on a log step axis."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rows = range(len(SETTINGS))
    plain = METHODS["plain training"][2]
    ax.plot(
        plain, rows, color=GRID, lw=2, zorder=1
    )  # how far tuning alone moves the baseline
    offsets = {
        "plain training": 0.0,
        "nowcasting": -0.14,
        "wave modules": 0.14,
        "both": 0.3,
    }
    # value label position per method (points): above, right or left of the marker
    label_at = {
        "plain training": (0, 9),
        "nowcasting": (0, 9),
        "wave modules": (24, 0),
        "both": (-22, 0),
    }
    for name, (colour, marker, steps) in METHODS.items():
        pts = [(s, r + offsets[name]) for r, s in zip(rows, steps) if s is not None]
        xs, ys = zip(*pts)
        ax.scatter(
            xs,
            ys,
            s=64,
            color=colour,
            marker=marker,
            label=name,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        for x, y in pts:
            ax.annotate(
                f"{x:,}",
                (x, y),
                xytext=label_at[name],
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color=MUTED,
            )
    ax.set_xscale("log")
    ax.set_xlim(500, 30000)
    ax.set_xticks(
        [500, 1000, 2000, 5000, 10000, 20000], ["500", "1k", "2k", "5k", "10k", "20k"]
    )
    ax.set_yticks(list(rows), SETTINGS)
    ax.set_ylim(len(SETTINGS) - 0.5, -0.6)
    ax.set_xlabel(
        "steps until test accuracy ≥ 99% (median over seeds, log scale)",
        color=MUTED,
        fontsize=9,
    )
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(
        "Tuning the baseline moves grokking more than either method",
        loc="left",
        fontsize=11,
        color=INK,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "summary.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT / 'summary.png'}")


if __name__ == "__main__":
    main()
