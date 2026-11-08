"""Check the training loop's norm pinning."""

import json
import tempfile
from pathlib import Path

import pytest

from quickgrok.train import train


def test_fix_norm() -> None:
    """With --fix-norm X from step 1 the logged total weight norm is X after that step only."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        train(
            "mlp",
            1,
            97,
            "add",
            0.3,
            3,
            1e-3,
            1.0,
            0,
            1,
            1,
            out,
            fix_norm=50.0,
            fix_norm_from=1,
        )
        norms = [
            m["weight_norm"] for m in json.loads((out / "metrics.json").read_text())
        ]
    assert norms[1] != pytest.approx(50.0, rel=1e-3)  # step 0 not pinned
    assert norms[2:] == pytest.approx([50.0, 50.0], rel=1e-5)
