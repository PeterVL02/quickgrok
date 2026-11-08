"""Online probe and substitution trigger (T3): does structure appear early enough to act on?

Report: reports/accel_probe.md.
  uv run python experiments/accel_probe.py run                    # 5 runs, ~70 s on one GPU
  uv run python experiments/accel_probe.py overhead --parallel 1  # probe overhead timing
  uv run python experiments/accel_probe.py figures                # -> reports/figures/accel_probe/
`overhead` runs seed 100 for 5,000 steps with and without the probe, 3 interleaved repeats, one
at a time; compare the medians of their wall.txt. Only the report's "final version" row can be
reproduced, the "first version" was older code. --save-every 5000 is taken from the original
run directories (runs/accel_overhead/), the report does not state it.
"""

from pathlib import Path

from quickgrok.launch import Job, main, make_jobs
from quickgrok.plot_compare import compare

# One arm, runs/accel_probe/seed<s>.
COMMON = "--model mlp --steps 40000 --probe-every 500 --stop-after-grok 3000"
SEEDS = [100, 101, 102, 103, 104]
JOBS = make_jobs("accel_probe", {"": ""}, SEEDS, COMMON)

# Overhead: runs/accel_overhead/<arm>/rep<r>.
OVERHEAD_COMMON = "--model mlp --steps 5000 --seed 100 --save-every 5000"
OVERHEAD = {"no_probe": "", "probe_every_500": "--probe-every 500"}
OVERHEAD_JOBS: list[Job] = [
    (Path("runs/accel_overhead", arm, f"rep{r}"), f"{OVERHEAD_COMMON} {flags}".split())
    for r in range(3)
    for arm, flags in OVERHEAD.items()
]


def figures() -> None:
    """Grokking curves and trigger lead times of the probed vanilla runs."""
    runs = [Path(f"runs/accel_probe/seed{s}") for s in SEEDS]
    compare({"MLP vanilla (probe)": runs}, Path("reports/figures/accel_probe"))


if __name__ == "__main__":
    main({"run": JOBS, "overhead": OVERHEAD_JOBS}, {"figures": figures})
