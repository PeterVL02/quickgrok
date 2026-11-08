"""The shared experiment launcher builds the expected jobs and runs a real (tiny) training run."""

import json
from pathlib import Path

from quickgrok.launch import launch, make_jobs


def test_launch_runs_a_tiny_training_job(tmp_path: Path) -> None:
    """make_jobs gives runs/<root>/<arm>/seed<s>; launch writes metrics, log and wall-clock."""
    jobs = make_jobs("demo", {"a": "--lr 2e-3", "b": ""}, [0, 1], "--model mlp")
    assert [out.as_posix() for out, _ in jobs] == [
        "runs/demo/a/seed0",
        "runs/demo/a/seed1",
        "runs/demo/b/seed0",
        "runs/demo/b/seed1",
    ]
    assert jobs[1][1] == ["--model", "mlp", "--lr", "2e-3", "--seed", "1"]

    out = tmp_path / "run"
    flags = ["--model", "mlp", "--steps", "20", "--eval-every", "10"]
    assert launch([(out, flags)], parallel=1, dry_run=False, env={}) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert [m["step"] for m in metrics] == [0, 10, 20]
    assert float((out / "wall.txt").read_text()) > 0
    assert "step     20" in (out / "log.txt").read_text()
