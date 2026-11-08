"""Neuron substitution (T4-T6): swap triggered neurons for wave modules, with controls.

Report: reports/substitution.md.
  uv run python experiments/substitution.py run       # 15 tuning + 60 final runs, ~10 min on one GPU
  uv run python experiments/substitution.py figures   # -> reports/figures/t6/
The report's (0.7, 0.8) seed-100 tuning run reused runs/accel_sub/module_seed100 (same flags);
here it is simply run again.
"""

from pathlib import Path

from quickgrok.launch import main, make_jobs
from quickgrok.plot_compare import compare

COMMON = "--model mlp --steps 40000 --probe-every 500 --stop-after-grok 3000"
MODULE = "--substitute module"

# Tuning, module mode: runs/t6/<arm>/seed<s>.
TUNE_SEEDS = [100, 101, 102, 103, 104]
TUNE = {
    "tune_0.5_0.8": f"{MODULE} --tau-share 0.5 --tau-r2 0.8",
    "tune_0.7_0.8": f"{MODULE} --tau-share 0.7 --tau-r2 0.8",
    "tune_0.85_0.9": f"{MODULE} --tau-share 0.85 --tau-r2 0.9",
}

# Final comparison: runs/t6/<arm>/seed<s>.
SEEDS = list(range(10))
ARMS = {
    "vanilla": "",
    "project": "--substitute project --tau-share 0.5 --tau-r2 0.8",
    "module": f"{MODULE} --tau-share 0.5 --tau-r2 0.8",
    "module_random_k": f"{MODULE} --tau-share 0.5 --tau-r2 0.8 --random-k",
    "module_all_step0": f"{MODULE} --sub-all-at 0",
    "module_all_step0_random_k": f"{MODULE} --sub-all-at 0 --random-k",
}

JOBS = make_jobs("t6", TUNE, TUNE_SEEDS, COMMON) + make_jobs("t6", ARMS, SEEDS, COMMON)


def figures() -> None:
    """Compare the six final arms."""
    arms = {arm: [Path(f"runs/t6/{arm}/seed{s}") for s in SEEDS] for arm in ARMS}
    compare(arms, Path("reports/figures/t6"))


if __name__ == "__main__":
    main({"run": JOBS}, {"figures": figures})
