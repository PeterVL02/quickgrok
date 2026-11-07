"""Online weight nowcasting and a Grokfast-EMA baseline, as `Hook`s for `quickgrok.accel`.

`NowcastHook` forecasts every weight `horizon` steps ahead from the run's own recent snapshots,
jumps there, and reverts the jump if the full-batch train loss rises by more than
`max_loss_ratio`. It never sees test data or another run.
"""

import json
import time
from pathlib import Path
from typing import Literal, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quickgrok.accel import Hook

ForecasterName = Literal["linear", "lstm"]


class NowcastRecord(TypedDict):
    step: int
    accepted: bool
    loss_before: float  # full-batch train loss before the jump
    loss_after: float  # ... and with the forecast weights
    rel_change: float  # ||forecast - w|| / ||w||
    # Backtest: forecast from the window ending at step - horizon vs the realised weights now,
    # as MSE / MSE of the no-change forecast. None until enough history exists.
    skill: float | None
    skill_clean: bool  # no accepted jump in [window start, step): purely realised
    fit_seconds: float  # forecaster training time (0 for linear)
    fit_loss_first: float | None  # LSTM training loss at the first and last Adam step
    fit_loss_last: float | None


def get_flat(model: nn.Module) -> Tensor:
    """All parameters of the model concatenated into one new vector."""
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_flat(model: nn.Module, flat: Tensor) -> None:
    """Copy a vector from `get_flat` back into the model's parameters, in place."""
    sizes = [p.numel() for p in model.parameters()]
    with torch.no_grad():
        for p, chunk in zip(model.parameters(), flat.split(sizes)):
            p.copy_(chunk.view_as(p))


class LinearForecaster:
    def __init__(self, k: int, snapshot_every: int, horizon: int) -> None:
        self.window = k  # snapshots per forecast
        self.snapshot_every = snapshot_every
        self.horizon = horizon

    def forecast(self, window: Tensor) -> Tensor:
        """Least-squares line through the snapshots `window` (L, N), extended `horizon` steps
        past the last one."""
        t = torch.arange(len(window), dtype=window.dtype, device=window.device)
        t = (t - t.mean()) * self.snapshot_every
        mean = window.mean(0)
        slope = (t[:, None] * (window - mean)).sum(0) / t.pow(2).sum()
        return mean + slope * (t[-1] + self.horizon)


class FitStats(TypedDict):
    seconds: float
    loss_first: float
    loss_last: float


class LSTMForecaster:
    def __init__(
        self,
        k: int,
        device: torch.device,
        hidden: int = 16,
        n_fit: int = 50_000,
        fit_steps: int = 200,
        lr: float = 1e-2,
        seed: int = 0,
    ) -> None:
        self.window = k + 1  # snapshots per forecast: k deltas
        self.lstm = nn.LSTM(1, hidden, batch_first=True).to(device)
        self.head = nn.Linear(hidden, 1).to(device)
        self.opt = torch.optim.Adam(
            [*self.lstm.parameters(), *self.head.parameters()], lr=lr
        )
        self.n_fit = n_fit
        self.fit_steps = fit_steps
        self.gen = torch.Generator().manual_seed(seed)

    @staticmethod
    def normalise(window: Tensor) -> tuple[Tensor, Tensor]:
        """Deltas (N, k) of the snapshots `window` (k + 1, N) over their RMS, and that RMS."""
        deltas = window.diff(dim=0).T
        scale = deltas.pow(2).mean().sqrt().clamp_min(1e-12)
        return deltas / scale, scale

    def predict(self, deltas: Tensor) -> Tensor:
        """Normalised total change over the horizon (N,) from normalised deltas (N, k)."""
        out, _ = self.lstm(deltas[..., None])
        return self.head(out[:, -1])[:, 0]

    def fit(self, window: Tensor, target: Tensor) -> FitStats:
        """Train on a random subsample of weights: `window` (k + 1, N) of snapshots ending
        `horizon` steps ago, `target` (N,) the realised weights now."""
        start = time.perf_counter()
        idx = torch.randperm(window.shape[1], generator=self.gen)[: self.n_fit]
        idx = idx.to(window.device)
        deltas, scale = self.normalise(window[:, idx])
        y = (target[idx] - window[-1, idx]) / scale
        losses: list[Tensor] = []
        for _ in range(self.fit_steps):
            loss = F.mse_loss(self.predict(deltas), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.detach())
        first, last = losses[0].item(), losses[-1].item()  # .item() syncs the GPU
        return {
            "seconds": time.perf_counter() - start,
            "loss_first": first,
            "loss_last": last,
        }

    def forecast(self, window: Tensor) -> Tensor:
        """Forecast weights (N,) `horizon` steps past the last snapshot of `window` (k + 1, N)."""
        deltas, scale = self.normalise(window)
        with torch.no_grad():
            return window[-1] + scale * self.predict(deltas)


class NowcastHook(Hook):
    """Snapshot every `snapshot_every` steps; every `jump_every` steps jump to the forecast.

    Snapshots hold the weights before any jump at their step. A jump is kept only if the
    full-batch train loss after it is at most `max_loss_ratio` times the loss before;
    otherwise the parameters are restored exactly. The LSTM forecaster is
    trained at each attempt on the window ending `horizon` steps ago. Optimizer state is left
    as is. Appends one `NowcastRecord` per attempt to `out_path` (truncated on construction).
    Plain parameters only (no parametrizations).
    """

    def __init__(
        self,
        forecaster: ForecasterName,
        x_train: Tensor,
        y_train: Tensor,
        out_path: Path,
        horizon: int = 1000,
        jump_every: int = 1000,
        snapshot_every: int = 100,
        k: int = 8,
        max_loss_ratio: float = 1.5,
        seed: int = 0,
    ) -> None:
        if horizon % snapshot_every or jump_every % snapshot_every:
            raise ValueError(
                "horizon and jump_every must be multiples of snapshot_every"
            )
        self.forecaster = (
            LinearForecaster(k, snapshot_every, horizon)
            if forecaster == "linear"
            else LSTMForecaster(k, x_train.device, seed=seed)
        )
        self.x_train = x_train
        self.y_train = y_train
        self.out_path = out_path
        self.horizon = horizon
        self.jump_every = jump_every
        self.snapshot_every = snapshot_every
        self.max_loss_ratio = max_loss_ratio
        self.snaps: dict[int, Tensor] = {}
        self.accepted_steps: list[int] = []
        out_path.write_text("")

    def window(self, end: int) -> Tensor | None:
        """Snapshots (L, N) ending at step `end`, or None if any is missing."""
        n = self.forecaster.window
        steps = [end - i * self.snapshot_every for i in reversed(range(n))]
        if any(s not in self.snaps for s in steps):
            return None
        return torch.stack([self.snaps[s] for s in steps])

    def train_loss(self, model: nn.Module) -> float:
        """Full-batch cross-entropy on the training set."""
        with torch.no_grad():
            return F.cross_entropy(model(self.x_train), self.y_train).item()

    def backtest(self, step: int, now: Tensor) -> tuple[float | None, bool]:
        """Skill of the forecast from the window ending `horizon` steps ago against `now`.

        Returns (forecast MSE / no-change MSE, whether no accepted jump lies in that window).
        """
        past = self.window(step - self.horizon)
        if past is None:
            return None, True
        start = step - self.horizon - (len(past) - 1) * self.snapshot_every
        clean = not any(start <= s < step for s in self.accepted_steps)
        err = (self.forecaster.forecast(past) - now).pow(2).mean()
        return (err / (past[-1] - now).pow(2).mean()).item(), clean

    def jump(self, step: int, model: nn.Module) -> NowcastRecord | None:
        """Try one jump at `step`; None if there is not enough history yet."""
        now = get_flat(model)
        current = self.window(step)
        train_past = self.window(step - self.horizon)
        is_lstm = isinstance(self.forecaster, LSTMForecaster)
        if current is None or (is_lstm and train_past is None):
            return None
        skill, clean = self.backtest(step, now)
        fit: FitStats | None = None
        if isinstance(self.forecaster, LSTMForecaster) and train_past is not None:
            fit = self.forecaster.fit(train_past, now)
        forecast = self.forecaster.forecast(current)
        loss_before = self.train_loss(model)
        set_flat(model, forecast)
        loss_after = self.train_loss(model)
        accepted = loss_after <= self.max_loss_ratio * loss_before
        if accepted:
            self.accepted_steps.append(step)
        else:
            set_flat(model, now)
        return {
            "step": step,
            "accepted": accepted,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "rel_change": ((forecast - now).norm() / now.norm()).item(),
            "skill": skill,
            "skill_clean": clean,
            "fit_seconds": fit["seconds"] if fit else 0.0,
            "fit_loss_first": fit["loss_first"] if fit else None,
            "fit_loss_last": fit["loss_last"] if fit else None,
        }

    def after_step(
        self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer | None:
        """Snapshot (pre-jump weights); every `jump_every` steps from `jump_every` on, try a jump."""
        if step % self.snapshot_every:
            return None
        self.snaps[step] = get_flat(model)
        if step >= self.jump_every and step % self.jump_every == 0:
            record = self.jump(step, model)
            if record is not None:
                with self.out_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
        keep = (self.forecaster.window - 1) * self.snapshot_every + self.horizon
        for s in [s for s in self.snaps if s < step - keep]:
            del self.snaps[s]
        return None


class GrokfastHook(Hook):
    """Grokfast-EMA: ema = alpha * ema + (1 - alpha) * grad, then grad += lamb * ema."""

    def __init__(self, alpha: float = 0.98, lamb: float = 2.0) -> None:
        self.alpha = alpha
        self.lamb = lamb
        self.ema: dict[nn.Parameter, Tensor] = {}

    def after_backward(self, step: int, model: nn.Module) -> None:
        """Update each gradient's EMA and add `lamb` times it to the gradient."""
        for p in model.parameters():
            if p.grad is None:
                continue
            ema = self.ema.setdefault(p, torch.zeros_like(p.grad))
            ema.mul_(self.alpha).add_(p.grad, alpha=1 - self.alpha)
            p.grad.add_(ema, alpha=self.lamb)
