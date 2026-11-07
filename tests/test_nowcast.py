"""Check the nowcast forecasters, jump verification and the Grokfast hook.

Run: uv run python tests/test_nowcast.py
"""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from quickgrok.data import make_dataset
from quickgrok.models import GrokMLP
from quickgrok.nowcast import (
    GrokfastHook,
    LinearForecaster,
    LSTMForecaster,
    NowcastHook,
    get_flat,
)
from quickgrok.train import to_tensors


def test_linear_exact() -> None:
    """A trajectory that is exactly linear in time is extrapolated exactly."""
    k, every, horizon = 8, 100, 3000
    w0, v = torch.randn(50, dtype=torch.float64), torch.randn(50, dtype=torch.float64)
    window = torch.stack([w0 + v * i * every for i in range(k)])
    forecast = LinearForecaster(k, every, horizon).forecast(window)
    assert torch.allclose(forecast, w0 + v * ((k - 1) * every + horizon), atol=1e-9)


def test_rejected_jump_restores() -> None:
    """With max_loss_ratio 0 every jump is rejected and the parameters are restored bit-for-bit."""
    torch.manual_seed(0)
    train, _ = make_dataset(97, "add", 0.3, seed=0)
    x, y = to_tensors(train, "cpu")
    model = GrokMLP(97, d_hidden=16)
    opt = torch.optim.SGD(model.parameters())
    with tempfile.TemporaryDirectory() as tmp:
        hook = NowcastHook("linear", x, y, Path(tmp) / "n.jsonl", max_loss_ratio=0.0)
        for step in range(0, 1001, 100):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.01 * torch.randn_like(p))
            before = get_flat(model)
            hook.after_step(step, model, opt)
            assert torch.equal(get_flat(model), before)
        records = [json.loads(line) for line in hook.out_path.read_text().splitlines()]
    assert len(records) == 1 and records[0]["step"] == 1000
    assert not records[0]["accepted"] and records[0]["rel_change"] > 0


def test_lstm_learns() -> None:
    """The LSTM forecaster's training loss falls on smooth synthetic trajectories."""
    torch.manual_seed(0)
    k, n = 8, 2000
    t = torch.arange(k + 1 + 10, dtype=torch.float32)[:, None]
    amp, freq, drift = torch.rand(n), 0.05 + 0.1 * torch.rand(n), torch.randn(n)
    traj = amp * torch.sin(freq * t) + 0.02 * drift * t  # (k + 11, n)
    lstm = LSTMForecaster(k, torch.device("cpu"), n_fit=n)
    stats = lstm.fit(traj[: k + 1], traj[-1])
    assert stats["loss_last"] < 0.5 * stats["loss_first"], stats
    assert lstm.forecast(traj[-(k + 1) :]).shape == (n,)


def test_grokfast_lambda_zero() -> None:
    """Grokfast with lambda 0 leaves gradients unchanged."""
    torch.manual_seed(0)
    model = GrokMLP(97, d_hidden=16)
    hook = GrokfastHook(alpha=0.98, lamb=0.0)
    x = torch.randint(0, 97, (64, 3))
    for step in range(3):
        model.zero_grad()
        F.cross_entropy(model(x), x[:, 0]).backward()
        grads = [p.grad.clone() for p in model.parameters()]
        hook.after_backward(step, model)
        assert all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters()))


if __name__ == "__main__":
    test_linear_exact()
    test_rejected_jump_restores()
    test_lstm_learns()
    test_grokfast_lambda_zero()
    print("ok")
