"""Train a 1-layer transformer on modular arithmetic and save checkpoints along the way.

Run: uv run python -m quickgrok.train --op add --steps 30000
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quickgrok.data import Example, Operation, make_dataset


class Checkpoint(TypedDict):
    p: int
    op: Operation
    train_frac: float
    seed: int
    step: int
    train_acc: float
    val_acc: float
    state_dict: dict[str, Tensor]


class Metrics(TypedDict):
    step: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    weight_norm: float  # L2 norm of all parameters


class GrokTransformer(nn.Module):
    def __init__(
        self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(p + 1, d_model)  # p numbers + "=" token
        self.pos = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # mlp[1] is the ReLU whose outputs are the "neurons" used in regress.py
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d_model)
        )
        self.unembed = nn.Linear(d_model, p)

    def forward(self, x: Tensor) -> Tensor:
        """Map token sequences (N, 3) to logits (N, p) read at the "=" position."""
        h = self.embed(x) + self.pos
        h = h + self.attn(h, h, h, need_weights=False)[0]
        h = h + self.mlp(h)
        return self.unembed(h[:, -1])


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


def load_model(path: Path) -> tuple[GrokTransformer, Checkpoint]:
    """Load a checkpoint onto the CPU and return the model in eval mode and the checkpoint."""
    ckpt: Checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GrokTransformer(ckpt["p"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def train(
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
) -> None:
    """Full-batch AdamW training.

    Every `eval_every` steps, losses and accuracies go to `out_dir/metrics.json`; every
    `save_every` steps (a multiple of `eval_every`) a checkpoint is saved and progress printed.
    """
    assert save_every % eval_every == 0, "save_every must be a multiple of eval_every"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    train_set, val_set = make_dataset(p, op, train_frac, seed)
    x_train, y_train = to_tensors(train_set, device)
    x_val, y_val = to_tensors(val_set, device)

    model = GrokTransformer(p).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[Metrics] = []

    for step in range(steps + 1):
        if step % eval_every == 0:
            train_loss, train_acc = evaluate(model, x_train, y_train)
            val_loss, val_acc = evaluate(model, x_val, y_val)
            norm = torch.cat([q.detach().flatten() for q in model.parameters()]).norm()
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
        if step % save_every == 0:
            print(
                f"step {step:6d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}"
                f"  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}"
            )
            ckpt: Checkpoint = {
                "p": p,
                "op": op,
                "train_frac": train_frac,
                "seed": seed,
                "step": step,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "state_dict": model.state_dict(),
            }
            torch.save(ckpt, out_dir / f"step_{step:06d}.pt")
            (out_dir / "metrics.json").write_text(json.dumps(history))
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()


def main() -> None:
    """Parse CLI arguments and run training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--op", choices=get_args(Operation), default="add")
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or Path("runs") / f"p{args.p}_{args.op}"
    train(
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
    )


if __name__ == "__main__":
    main()
