"""Retest on a tuned baseline: do substitution and nowcasting still add speed, and do they combine?

Report: reports/tuned_retest.md.
  uv run python experiments/tuned_retest.py run                  # 35 tuning + 65 final runs
  uv run python experiments/tuned_retest.py timing --parallel 1  # 27 solo runs for wall-clock
  uv run python experiments/tuned_retest.py figures              # -> reports/figures/tuned_retest{,_fast}/
The report ran 10 at a time: tuning 69 s, final + secondary 154 s, solo timing 157 s (RTX 5080).
Wall-clock per run is in each run's wall.txt. The overhead micro-benchmark is not included.
"""

from pathlib import Path

from quickgrok.launch import main, make_jobs
from quickgrok.plot_compare import compare

B = "--model mlp --lr 4e-3 --weight-decay 2.0 --steps 20000 --stop-after-grok 1000 --eval-every 20 --save-every 500"
NOWCAST = "--nowcast linear --nowcast-snapshot-every 20 --nowcast-k 8 --nowcast-max-loss-ratio 1.5"
NC = f"{NOWCAST} --nowcast-every 160 --nowcast-horizon 160"
MO = "--substitute module --probe-every 100 --tau-share 0.5 --tau-r2 0.8"

# Tuning: runs/tuned_retest/tune/<arm>/seed<s>.
TUNE_SEEDS = [100, 101, 102, 103, 104]
TUNE = {"vanilla": ""}
for j in (160, 300):
    for h in (160, 400):
        TUNE[f"nowcast_J{j}_H{h}"] = (
            f"{NOWCAST} --nowcast-every {j} --nowcast-horizon {h}"
        )
for p in (100, 200):
    TUNE[f"module_p{p}"] = (
        f"--substitute module --probe-every {p} --tau-share 0.5 --tau-r2 0.8"
    )

# Final: runs/tuned_retest/<arm>/seed<s>.
SEEDS = list(range(10))
ARMS = {
    "vanilla": "",
    "nowcast_linear": NC,
    "module": MO,
    "module_random_k": f"{MO} --random-k",
    "nowcast_plus_module": f"{NC} {MO}",
}

# Secondary at lr 1.6e-2, step constants x0.4: runs/tuned_retest_fast/<arm>/seed<s>.
BF = B.replace("--lr 4e-3", "--lr 1.6e-2")
FAST_SEEDS = [0, 1, 2, 3, 4]
FAST = {
    "vanilla": "",
    "nowcast_linear": "--nowcast linear --nowcast-snapshot-every 8 --nowcast-every 64 "
    "--nowcast-horizon 64 --nowcast-k 8 --nowcast-max-loss-ratio 1.5",
    "module": "--substitute module --probe-every 40 --tau-share 0.5 --tau-r2 0.8",
}

JOBS = (
    make_jobs("tuned_retest/tune", TUNE, TUNE_SEEDS, B)
    + make_jobs("tuned_retest", ARMS, SEEDS, B)
    + make_jobs("tuned_retest_fast", FAST, FAST_SEEDS, BF)
)

# Wall-clock: seeds 0-2 of every final and secondary arm with --stop-after-grok 0 (which
# overrides B's 1000), run alone one after another, plus 3 empty runs to measure start-up.
# Out: runs/tuned_retest_timing{,_fast}/<arm>/seed<s>.
TIMING_SEEDS = [0, 1, 2]
TIMING_JOBS = (
    make_jobs("tuned_retest_timing", ARMS, TIMING_SEEDS, B + " --stop-after-grok 0")
    + make_jobs(
        "tuned_retest_timing_fast", FAST, TIMING_SEEDS, BF + " --stop-after-grok 0"
    )
    + make_jobs(
        "tuned_retest_timing", {"startup": ""}, TIMING_SEEDS, "--model mlp --steps 0"
    )
)


def figures() -> None:
    """Compare the final arms at lr 4e-3 and the secondary arms at lr 1.6e-2."""
    for root, arms, seeds in [
        ("tuned_retest", ARMS, SEEDS),
        ("tuned_retest_fast", FAST, FAST_SEEDS),
    ]:
        runs = {
            arm: [Path(f"runs/{root}/{arm}/seed{s}") for s in seeds] for arm in arms
        }
        compare(runs, Path("reports/figures") / root)


if __name__ == "__main__":
    main({"run": JOBS, "timing": TIMING_JOBS}, {"figures": figures})
