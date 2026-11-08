"""Does linear nowcasting still save steps and wall-clock time on the 1-layer transformer?

Pre-registration: reports/preregistration_nowcast_transformer.md. Report: reports/nowcast_transformer.md.
  uv run python experiments/nowcast_transformer.py tune                 # baseline lr x wd grid
  uv run python experiments/nowcast_transformer.py tune_nowcast         # 3 nowcast configs
  uv run python experiments/nowcast_transformer.py run                  # final, seeds 0-9
  uv run python experiments/nowcast_transformer.py timing --parallel 1  # wall-clock, solo
  uv run python experiments/nowcast_transformer.py figures  # -> reports/figures/nowcast_transformer/
Wall-clock per run is in each run's wall.txt.
"""

from pathlib import Path

from quickgrok.launch import main, make_jobs
from quickgrok.plot_compare import compare

B = "--model transformer --steps 20000 --eval-every 20 --save-every 500"
TUNE_B = f"{B} --stop-after-grok 500"

# Baseline tuning (pre-registration, section 1): runs/nowcast_transformer/tune/<arm>/seed<s>.
TUNE_SEEDS = [100, 101, 102]
# lr 1.6e-2 is the pre-registered one-step extension (8e-3 was the fastest, at the grid edge).
TUNE = {
    f"lr{lr}_wd{wd}": f"--lr {lr} --weight-decay {wd}"
    for lr, wd in [
        ("2e-3", 1),
        ("4e-3", 1),
        ("8e-3", 1),
        ("4e-3", 2),
        ("8e-3", 2),
        ("1.6e-2", 2),
    ]
}

# Frozen baseline choice and its median steps to 99% test G on seeds 100-102.
LR, WD, G = "8e-3", 2, 880
PLAIN = f"--lr {LR} --weight-decay {WD}"

# Nowcast tuning (section 2): runs/nowcast_transformer/tune_nowcast/<arm>/seed<s>.
S = max(1, round(0.06 * G / 8))  # snapshot interval
NOWCAST = (
    f"--nowcast linear --nowcast-snapshot-every {S} --nowcast-k 8 "
    "--nowcast-max-loss-ratio 1.5"
)
NC_SEEDS = [100, 101, 102, 103, 104]
TUNE_NC = {
    f"nowcast_{name}": f"{NOWCAST} --nowcast-every {j} --nowcast-horizon {h}"
    for name, j, h in [("A", 8 * S, 8 * S), ("B", 16 * S, 8 * S), ("C", 8 * S, 4 * S)]
}

# Final (section 3): runs/nowcast_transformer/<arm>/seed<s>. Frozen nowcast choice:
ARMS = {"plain": "", "nowcast": TUNE_NC["nowcast_C"]}
SEEDS = list(range(10))

# Wall-clock: seeds 0-2 of both arms, stopped at the 99% step, plus 3 empty start-up runs.
# Out: runs/nowcast_transformer_timing/<arm>/seed<s>.
TIMING_SEEDS = [0, 1, 2]

JOB_SETS = {
    "tune": make_jobs("nowcast_transformer/tune", TUNE, TUNE_SEEDS, TUNE_B),
    "tune_nowcast": make_jobs(
        "nowcast_transformer/tune_nowcast",
        {"plain": ""},
        [103, 104],
        f"{TUNE_B} {PLAIN}",
    )
    + make_jobs(
        "nowcast_transformer/tune_nowcast", TUNE_NC, NC_SEEDS, f"{TUNE_B} {PLAIN}"
    ),
    "run": make_jobs(
        "nowcast_transformer", ARMS, SEEDS, f"{B} {PLAIN} --stop-after-grok 1000"
    ),
    "timing": make_jobs(
        "nowcast_transformer_timing",
        ARMS,
        TIMING_SEEDS,
        f"{B} {PLAIN} --stop-after-grok 0",
    )
    + make_jobs(
        "nowcast_transformer_timing",
        {"startup": ""},
        TIMING_SEEDS,
        "--model transformer --steps 0",
    ),
}


def figures() -> None:
    """Compare the tuning arms and the final arms; tables go next to the figures."""
    out = Path("reports/figures/nowcast_transformer")
    for sub, arms, seeds in [
        ("tune", TUNE, TUNE_SEEDS),
        ("tune_nowcast", {"plain": ""} | TUNE_NC, NC_SEEDS),
    ]:
        runs = {
            arm: [Path(f"runs/nowcast_transformer/{sub}/{arm}/seed{s}") for s in seeds]
            for arm in arms
        }
        # Plain was only run on seeds 103-104 here; 100-102 are the baseline-tuning runs.
        if sub == "tune_nowcast":
            runs["plain"] = [
                Path(f"runs/nowcast_transformer/tune/lr{LR}_wd{WD}/seed{s}")
                for s in TUNE_SEEDS
            ] + runs["plain"][3:]
        compare(runs, out / sub)
    runs = {
        arm: [Path(f"runs/nowcast_transformer/{arm}/seed{s}") for s in SEEDS]
        for arm in ARMS
    }
    compare(runs, out)


if __name__ == "__main__":
    main(JOB_SETS, {"figures": figures})
