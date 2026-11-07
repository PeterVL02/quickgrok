"""Train a 1-layer transformer or an MLP on modular arithmetic, saving checkpoints along the way.

Run: uv run python -m quickgrok.train --model mlp --op add --steps 30000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Literal, TypedDict, get_args

# Idle OpenMP threads sleep instead of spinning. Spinning after the probe's CPU fits slowed the
# GPU training loop by ~10%. Only takes effect if torch is not imported yet.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quickgrok.accel import (
    Accelerator,
    Hook,
    ProbeHook,
    SubstitutionHook,
    SubstitutionMode,
    effective_state_dict,
)
from quickgrok.data import S5_ORDER, Example, Operation, make_dataset
from quickgrok.models import (  # noqa: F401  (re-exported for `from quickgrok.train import ...`)
    Checkpoint,
    GrokMLP,
    GrokModel,
    GrokTransformer,
    ModelName,
    build_model,
    get_activation_layers,
    load_model,
)

GROK_ACC = 0.99  # test accuracy that counts as grokked


class Metrics(TypedDict):
    step: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    weight_norm: float  # L2 norm of all (effective) weights


def to_tensors(examples: list[Example], device: str) -> tuple[Tensor, Tensor]:
    """Stack examples into an input tensor (N, 3) and a target tensor (N,)."""
    x = torch.tensor([tokens for tokens, _ in examples], device=device)
    y = torch.tensor([c for _, c in examples], device=device)
    return x, y


def evaluate(model: nn.Module, x: Tensor, y: Tensor) -> tuple[float, float]:
    """Return (cross-entropy loss, accuracy) of the model on the given examples."""
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y).item()
        return loss, (logits.argmax(-1) == y).float().mean().item()


def train(
    model_name: ModelName,
    n_hidden: int,
    p: int,
    op: Operation,
    train_frac: float,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
    eval_every: int,
    save_every: int,
    out_dir: Path,
    probe_every: int = 0,
    probe_layer: int = 0,
    probe_harmonics: int = 4,
    stop_after_grok: int | None = None,
    substitute: Literal["none"] | SubstitutionMode = "none",
    tau_share: float = 0.7,
    tau_r2: float = 0.8,
    random_k: bool = False,
    sub_all_at: int | None = None,
) -> None:
    """Full-batch AdamW training.

    Every `eval_every` steps, losses and accuracies go to `out_dir/metrics.json`; every
    `save_every` steps (a multiple of `eval_every`) a checkpoint is saved and progress printed.
    `probe_every > 0` adds a `ProbeHook` writing `out_dir/probe.jsonl`; with `substitute` other
    than "none" it is a `SubstitutionHook` (thresholds `tau_share`, `tau_r2`) that also writes
    `out_dir/substitutions.jsonl` (see `SubstitutionHook` for `random_k` and `sub_all_at`).
    Checkpoints and `weight_norm` use the effective weights. With `stop_after_grok`, training
    ends that many steps after test accuracy first reaches GROK_ACC. For op "s5", `p` is
    replaced by the number of symbols, `S5_ORDER`.
    """
    assert save_every % eval_every == 0, "save_every must be a multiple of eval_every"
    p = S5_ORDER if op == "s5" else p
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    train_set, val_set = make_dataset(p, op, train_frac, seed)
    x_train, y_train = to_tensors(train_set, device)
    x_val, y_val = to_tensors(val_set, device)

    model = build_model(model_name, p, n_hidden).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    hooks: list[Hook] = []
    probe_args = (
        x_train,
        p,
        out_dir / "probe.jsonl",
        probe_layer,
        probe_every,
        probe_harmonics,
        seed,
    )
    if substitute != "none":
        assert probe_every > 0, "substitution needs probe_every > 0"
        hooks.append(
            SubstitutionHook(
                *probe_args,
                mode=substitute,
                tau_share=tau_share,
                tau_r2=tau_r2,
                y_train=y_train,
                log_path=out_dir / "substitutions.jsonl",
                random_k=random_k,
                sub_all_at=sub_all_at,
            )
        )
    elif probe_every:
        hooks.append(ProbeHook(*probe_args))
    accel = Accelerator(model, hooks)
    history: list[Metrics] = []
    end = steps

    for step in range(steps + 1):
        if step % eval_every == 0 or step == end:
            train_loss, train_acc = evaluate(model, x_train, y_train)
            val_loss, val_acc = evaluate(model, x_val, y_val)
            weights = effective_state_dict(model).values()
            norm = torch.cat([w.flatten() for w in weights]).norm()
            history.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "weight_norm": norm.item(),
                }
            )
            if stop_after_grok is not None and end == steps and val_acc >= GROK_ACC:
                end = min(steps, step + stop_after_grok)
        if step % save_every == 0 or step == end:
            print(
                f"step {step:6d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}"
                f"  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}"
            )
            ckpt: Checkpoint = {
                "model": model_name,
                "n_hidden": n_hidden,
                "p": p,
                "op": op,
                "train_frac": train_frac,
                "seed": seed,
                "step": step,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "state_dict": effective_state_dict(model),
            }
            torch.save(ckpt, out_dir / f"step_{step:06d}.pt")
            (out_dir / "metrics.json").write_text(json.dumps(history))
        if step == end:
            break
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        accel.after_backward(step)
        opt.step()
        opt = accel.after_step(step, opt)


def main() -> None:
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=get_args(ModelName), default="transformer")
    parser.add_argument("--n-hidden", type=int, default=1, help="MLP hidden layers")
    parser.add_argument("--p", type=int, default=97, help="ignored for --op s5")
    parser.add_argument("--op", choices=get_args(Operation), default="add")
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--probe-every", type=int, default=0, help="0 = no probe")
    parser.add_argument("--probe-layer", type=int, default=0)
    parser.add_argument("--probe-harmonics", type=int, default=4)
    parser.add_argument(
        "--stop-after-grok",
        type=int,
        default=None,
        help="end training this many steps after test acc first reaches 99%%",
    )
    parser.add_argument(
        "--substitute",
        choices=["none", *get_args(SubstitutionMode)],
        default="none",
        help="substitute neurons that pass the probe trigger (needs --probe-every)",
    )
    parser.add_argument("--tau-share", type=float, default=0.7)
    parser.add_argument("--tau-r2", type=float, default=0.8)
    parser.add_argument(
        "--random-k",
        action="store_true",
        help="control: substitute with a random frequency instead of the fitted k*",
    )
    parser.add_argument(
        "--sub-all-at",
        type=int,
        default=None,
        help="substitute every live neuron at this probe step, ignoring the thresholds",
    )
    args = parser.parse_args()
    if args.substitute != "none" and args.probe_every <= 0:
        parser.error("--substitute needs --probe-every > 0")
    if args.substitute == "none" and (args.random_k or args.sub_all_at is not None):
        parser.error("--random-k and --sub-all-at need --substitute")
    out_dir = args.out_dir or Path("runs") / f"{args.model}_p{args.p}_{args.op}"
    train(
        args.model,
        args.n_hidden,
        args.p,
        args.op,
        args.train_frac,
        args.steps,
        args.lr,
        args.weight_decay,
        args.seed,
        args.eval_every,
        args.save_every,
        out_dir,
        args.probe_every,
        args.probe_layer,
        args.probe_harmonics,
        args.stop_after_grok,
        args.substitute,
        args.tau_share,
        args.tau_r2,
        args.random_k,
        args.sub_all_at,
    )


if __name__ == "__main__":
    main()
