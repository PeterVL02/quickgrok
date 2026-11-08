"""Command line shared by the experiment files in `experiments/`.

Each experiment lists its training jobs (output directory and `quickgrok.train` flags) and calls
`main`, which gives it the subcommands `run [--dry-run] [--parallel N]` (one per job set) plus its
own commands such as `figures`. Paths are relative, so run from the repo root.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# (output directory, quickgrok.train flags without --out-dir)
Job = tuple[Path, list[str]]


def make_jobs(
    root: str, arms: dict[str, str], seeds: Sequence[int], common: str = ""
) -> list[Job]:
    """One job per arm and seed: `common`, the arm's flags and `--seed s`, into runs/<root>/<arm>/seed<s>."""
    return [
        (
            Path("runs", root, arm, f"seed{s}"),
            shlex.split(f"{common} {flags} --seed {s}"),
        )
        for arm, flags in arms.items()
        for s in seeds
    ]


def run_job(job: Job, env: dict[str, str]) -> tuple[Path, float, int]:
    """Run one training process; return (output dir, wall-clock seconds, return code).

    Its output goes to <out>/log.txt and its wall-clock time to <out>/wall.txt.
    """
    out, flags = job
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "quickgrok.train", *flags, "--out-dir", str(out)]
    start = time.perf_counter()
    with open(out / "log.txt", "w") as log:
        rc = subprocess.run(
            cmd,
            check=False,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
        ).returncode
    seconds = time.perf_counter() - start
    (out / "wall.txt").write_text(f"{seconds:.2f}\n")
    return out, seconds, rc


def launch(jobs: list[Job], parallel: int, dry_run: bool, env: dict[str, str]) -> int:
    """Run the jobs `parallel` at a time (or print their commands); return the number that failed."""
    if dry_run:
        prefix = "".join(f"{k}={v} " for k, v in env.items())
        for out, flags in jobs:
            cmd = ["uv", "run", "python", "-m", "quickgrok.train", *flags]
            print(prefix + shlex.join([*cmd, "--out-dir", out.as_posix()]))
        return 0
    start = time.perf_counter()
    failed = 0
    with ThreadPoolExecutor(parallel) as pool:
        for out, seconds, rc in pool.map(lambda job: run_job(job, env), jobs):
            print(f"{out.as_posix()}  {seconds:.1f}s  rc={rc}", flush=True)
            failed += rc != 0
    print(f"{len(jobs)} runs, {failed} failed, {time.perf_counter() - start:.1f}s")
    return failed


def main(
    job_sets: dict[str, list[Job]],
    commands: dict[str, Callable[[], None]],
    env: dict[str, str] | None = None,
) -> None:
    """Parse the experiment's command line and launch a job set or run a command.

    `env` is added to the environment of every training process.
    """
    parser = argparse.ArgumentParser(
        description=sys.modules["__main__"].__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, jobs in job_sets.items():
        p = sub.add_parser(name, help=f"launch {len(jobs)} training runs")
        p.add_argument("--dry-run", action="store_true", help="print the commands")
        p.add_argument("--parallel", type=int, default=10, help="runs at a time")
    for name, command in commands.items():
        sub.add_parser(name, help=(command.__doc__ or "").split("\n")[0])
    args = parser.parse_args()
    if args.command in job_sets:
        jobs = job_sets[args.command]
        sys.exit(launch(jobs, args.parallel, args.dry_run, env or {}) > 0)
    commands[args.command]()
