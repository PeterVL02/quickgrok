"""MLP baseline: the one-hidden-layer MLP as a cleaner test bed than the transformer.

Report: reports/mlp_baseline.md.
  uv run python experiments/mlp_baseline.py run       # 5 runs + 4 regime runs, ~2 min on one GPU
  uv run python experiments/mlp_baseline.py figures   # -> reports/figures/mlp_baseline/
The figures compare with the transformer runs of experiments/pilot.py (runs/pilot/).
"""

from pathlib import Path

from quickgrok.launch import main, make_jobs
from quickgrok.plot_compare import compare
from quickgrok.plot_pilot import make_figures

# Main arm: runs/mlp_baseline/seed<s>.
COMMON = "--model mlp --steps 40000"
SEEDS = [0, 1, 2, 3, 4]
# "Choosing the regime": single-seed exploration over 30k steps, runs/mlp_baseline/<arm>/seed0.
# The report's second row says train fraction 0.3; the original run directory (wd03_f40) and its
# metrics.json point to train fraction 0.4. The report's value is used here.
EXPLORE = {
    "explore_tf0.3_wd1": "--train-frac 0.3 --weight-decay 1.0",
    "explore_tf0.3_wd0.3": "--train-frac 0.3 --weight-decay 0.3",
    "explore_tf0.4_wd1": "--train-frac 0.4 --weight-decay 1.0",
    "explore_tf0.5_wd1": "--train-frac 0.5 --weight-decay 1.0",
}
JOBS = make_jobs("mlp_baseline", {"": ""}, SEEDS, COMMON) + make_jobs(
    "mlp_baseline", EXPLORE, [0], "--model mlp --steps 30000"
)

RUNS = Path("runs/mlp_baseline")
FIG = Path("reports/figures/mlp_baseline")


def figures() -> None:
    """Compare the MLP with the pilot transformer and probe the MLP's hidden layer."""
    compare(
        {
            "MLP vanilla": [RUNS / f"seed{s}" for s in SEEDS],
            "transformer (pilot)": [Path(f"runs/pilot/seed{s}") for s in (0, 1, 2)],
        },
        FIG,
    )
    make_figures([RUNS / f"seed{s}" for s in (0, 1, 2)], FIG / "probe", layer=0)


if __name__ == "__main__":
    main({"run": JOBS}, {"figures": figures})
