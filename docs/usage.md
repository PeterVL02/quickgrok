# Using quickgrok

quickgrok is research code for studying **grokking**: small networks on modular arithmetic (a + b mod 97) fit their training data quickly and only generalise much later. It trains those networks, probes hidden neurons with regressions during training, and can swap well-fit neurons for trainable parametric modules to make grokking happen sooner. Results are in `reports/`; a student-level explanation is in `intuition/README.md`.

> Status: research code, not a published package. Interfaces may change between branches. This page describes what exists on the current branch; update it when an entry point changes.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.15.

```bash
uv sync
```

`pyproject.toml` pins torch to the CUDA 13.2 wheel index (`[tool.uv.sources]`), which suits recent NVIDIA GPUs and drivers. On a machine without a suitable GPU, delete the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks and run `uv sync` again to get the default build; everything also runs on the CPU, just slower.

## Quick start (command line)

```bash
# Vanilla MLP on a + b mod 97: memorises by step ~200, groks (99% test acc) at ~18k steps
uv run python -m quickgrok.train --model mlp --steps 40000 --seed 0 --out-dir runs/vanilla/seed0

# Same, with a probe that fits every hidden neuron on Fourier features every 500 steps
uv run python -m quickgrok.train --model mlp --steps 40000 --seed 0 --probe-every 500 \
    --stop-after-grok 3000 --out-dir runs/probe/seed0

# Swap neurons for trainable wave modules once their fit is good enough (~41% sooner grokking)
uv run python -m quickgrok.train --model mlp --steps 40000 --seed 0 --probe-every 500 \
    --stop-after-grok 3000 --substitute module --tau-share 0.5 --tau-r2 0.8 --out-dir runs/module/seed0

# Compare groups of runs: curves, grokking step per seed, median/IQR table, Mann-Whitney test
uv run python -m quickgrok.plot_compare --arm vanilla runs/vanilla/seed0 \
    --arm module runs/module/seed0 --out reports/figures/my_comparison
```

`uv run python -m quickgrok.train --help` lists every flag. The most used ones:

| Flag | Meaning |
| --- | --- |
| `--model mlp\|transformer` | One-hidden-layer MLP on one-hot (a, b), or a 1-layer transformer |
| `--op add\|sub\|mul\|div\|s5`, `--p`, `--train-frac` | Task: a op b mod p (or composition in S5, which fixes p = 120), fraction of all pairs used for training |
| `--steps`, `--lr`, `--weight-decay`, `--seed` | Full-batch AdamW training |
| `--probe-every N` | Fit all hidden neurons every N steps (training inputs only) |
| `--substitute none\|project\|module` | Swap triggered neurons: snap weights once, or constrain them to a trainable wave module |
| `--tau-share`, `--tau-r2` | Trigger thresholds (fraction of variance in one frequency and its harmonics; R² on held-out training rows) |
| `--basis probe\|fourier\|poly\|svd\|random\|auto\|levels_2\|levels_3\|self_expr`, `--tau-energy` | Substitution basis: `probe` uses the activation fit's frequency; the others are chosen from the weights and trigger on captured energy ≥ `--tau-energy` |
| `--random-k`, `--sub-all-at N` | Controls: wrong frequency; swap every neuron at step N |
| `--nowcast linear\|lstm`, `--nowcast-every`, `--nowcast-horizon`, `--nowcast-snapshot-every`, `--nowcast-k` | Forecast every weight `--nowcast-horizon` steps ahead from the run's own snapshots and jump there every `--nowcast-every` steps; revert if train loss rises more than `--nowcast-max-loss-ratio` |
| `--grokfast-lambda`, `--grokfast-alpha` | Grokfast-EMA gradient filter (baseline; 0 = off) |
| `--monitor-every N` | Log the train-only grokking monitors every N steps (MLP only) |
| `--random-labels` | Control: a fixed seeded permutation of the training targets |
| `--stop-after-grok N` | Stop N steps after test accuracy first reaches 99% |

Other entry points: `uv run python -m quickgrok.plot_nowcast` (nowcast diagnostics per arm), `uv run python -m quickgrok.regress` (fit neurons of saved checkpoints), and the one-off analyses in `scripts/` (each lists its command in its docstring).

## What a run writes

Every run directory (`--out-dir`) contains:

| File | Content |
| --- | --- |
| `metrics.json` | Every `--eval-every` steps: train/test loss and accuracy, parameter norm |
| `step_XXXXXX.pt` | Checkpoint every `--save-every` steps (load with `quickgrok.models.load_model`) |
| `probe.jsonl` | One line per probe: each neuron's frequency, harmonic share and held-out R² |
| `substitutions.jsonl` | One line per swap event: step, neurons, frequencies, train loss before/after |
| `nowcast.jsonl` | With `--nowcast`: one line per jump attempt, accepted or reverted |
| `monitor.jsonl` | With `--monitor-every`: one `MonitorRecord` of train-only monitors per log step |

## Python API

| Module | Main entry points |
| --- | --- |
| `quickgrok.data` | `make_dataset(p, op, train_frac, seed) -> (train, test)`, each a list of `((a, b, eq_token), answer)` |
| `quickgrok.models` | `build_model("mlp" \| "transformer", p)`, `GrokMLP`, `GrokTransformer`, `load_model(path) -> (model, checkpoint)`, `get_activation_layers(model)` |
| `quickgrok.regress` | `record_neurons(model, neurons=None, layer=0)` (activations on all p² inputs), `record_activations(model, x, layer)`, `fit_neurons(...)` (least-squares fits on Fourier or polynomial features) |
| `quickgrok.accel` | `Hook`, `Accelerator`, `ProbeHook`, `SubstitutionHook`, `effective_state_dict(model)` |
| `quickgrok.nowcast` | `NowcastHook`, `GrokfastHook`, `LinearForecaster`, `LSTMForecaster` |
| `quickgrok.monitor` | `MonitorHook` (train-only grokking monitors, `GrokMLP` only) |
| `quickgrok.plot_compare`, `quickgrok.plot_pilot`, `quickgrok.plot_nowcast` | Figures; mostly used from the command line |

### Hooks in your own training loop

`Accelerator` runs a list of hooks at two points of an ordinary PyTorch loop: after `backward()` (a hook may edit gradients) and after `optimizer.step()` (a hook may edit weights or modules and may hand back a new optimizer, e.g. when it added parameters). Always keep the returned optimizer.

```python
from pathlib import Path

import torch
import torch.nn.functional as F

from quickgrok.accel import Accelerator, SubstitutionHook
from quickgrok.data import make_dataset
from quickgrok.models import build_model

p, device = 97, "cuda" if torch.cuda.is_available() else "cpu"
train_set, _ = make_dataset(p, "add", train_frac=0.3, seed=0)
x = torch.tensor([tokens for tokens, _ in train_set], device=device)
y = torch.tensor([c for _, c in train_set], device=device)

model = build_model("mlp", p).to(device)
opt = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98)
)
out = Path("runs/my_loop")
out.mkdir(parents=True, exist_ok=True)
hook = SubstitutionHook(
    x,
    p,
    out / "probe.jsonl",
    every=500,
    mode="module",
    tau_share=0.5,
    tau_r2=0.8,
    y_train=y,
    log_path=out / "substitutions.jsonl",
)
accel = Accelerator(model, [hook])

for step in range(20_000):
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    accel.after_backward(step)
    opt.step()
    opt = accel.after_step(step, opt)  # may be a rebuilt optimizer after a swap
```

To save a model that has swapped neurons, use `effective_state_dict(model)` instead of `model.state_dict()`: it stores plain weights, so `load_model` and every analysis script can read the checkpoint.

### Writing your own hook

Subclass `Hook` and override one or both methods. This one doubles the gradient of every parameter for the first 1,000 steps:

```python
from torch import nn

from quickgrok.accel import Hook


class WarmBoost(Hook):
    def after_backward(self, step: int, model: nn.Module) -> None:
        """Scale gradients by 2 during the first 1,000 steps."""
        if step < 1000:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(2.0)
```

`Hook` and `Accelerator` work with any `nn.Module`. `ProbeHook` and `SubstitutionHook` are specific to this project: they expect token inputs `(a, b, "=")`, and substitution currently supports only the first hidden layer of `GrokMLP`.

## Good to know

- **Compare over seeds, not single runs.** GPU training is not bit-for-bit reproducible; the same seed can grok ~10% earlier or later. The reports use 10 seeds and a Mann–Whitney test.
- **No test data in the fits.** Probes and swaps only ever use training inputs (tested in `tests/test_accel.py`).
- **Tests:** `uv run pytest`.
- **Lint:** `uv run ruff check . && uv run ruff format .`
