"""One-off PCA / pruning analysis of the pilot checkpoints (2026-11-07). Produces
reports/figures/pilot/pca_rank.png. Standalone on purpose: it copies the transformer class
instead of importing quickgrok, so it keeps working on the pilot checkpoints.

Run: uv run python scripts/pca_probe_pilot.py
"""

"""PCA / prunability probe of MLP hidden activations across grokking checkpoints."""

import json
import random
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "pilot"
FIG = ROOT / "reports" / "figures" / "pilot" / "pca_rank.png"
SEEDS = (0, 1, 2)


class GrokTransformer(nn.Module):
    def __init__(
        self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(p + 1, d_model)
        self.pos = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.ReLU(), nn.Linear(d_mlp, d_model)
        )
        self.unembed = nn.Linear(d_model, p)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits for the last position."""
        h = self.embed(x) + self.pos
        h = h + self.attn(h, h, h, need_weights=False)[0]
        h = h + self.mlp(h)
        return self.unembed(h[:, -1])


def train_pairs(p: int, train_frac: float, seed: int) -> list[tuple[int, int]]:
    """Reproduce the training split exactly."""
    pairs = [(a, b) for a in range(p) for b in range(p)]
    examples = [((a, b, p), (a + b) % p) for a, b in pairs]
    random.Random(seed).shuffle(examples)
    n = int(train_frac * len(examples))
    return [(a, b) for (a, b, _), _ in examples[:n]]


def make_batch(pairs: list[tuple[int, int]], p: int) -> tuple[Tensor, Tensor]:
    """Build (N, 3) token inputs and (N,) labels."""
    x = torch.tensor([[a, b, p] for a, b in pairs])
    return x, (x[:, 0] + x[:, 1]) % p


def load_model(path: Path) -> tuple[GrokTransformer, dict[str, object]]:
    """Load a checkpoint into a model in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    m = GrokTransformer(ckpt["p"])
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, ckpt


@torch.no_grad()
def run(
    m: GrokTransformer, x: Tensor, edit: Callable[[Tensor], Tensor] | None = None
) -> tuple[Tensor, Tensor]:
    """Forward pass; returns (logits, last-position ReLU activations after optional edit)."""
    store: dict[str, Tensor] = {}

    def hook(_mod: nn.Module, _inp: tuple[Tensor, ...], out: Tensor) -> Tensor:
        """Forward hook: apply `edit` to the last position, store it, return the output."""
        if edit is not None:
            out = out.clone()
            out[:, -1] = edit(out[:, -1])
        store["act"] = out[:, -1]
        return out

    h = m.mlp[1].register_forward_hook(hook)
    try:
        logits = m(x)
    finally:
        h.remove()
    return logits, store["act"]


def acc(logits: Tensor, y: Tensor) -> float:
    """Classification accuracy."""
    return (logits.argmax(-1) == y).double().mean().item()


def pca_stats(act: Tensor) -> tuple[float, int, int, Tensor, Tensor, Tensor]:
    """Return participation ratio, #PCs for 90/99% variance, eigenvalues, mean and right singular vectors."""
    a = act.double()
    mu = a.mean(0)
    _, s, vh = torch.linalg.svd(a - mu, full_matrices=False)
    lam = s**2
    pr = (lam.sum() ** 2 / (lam**2).sum()).item()
    cum = lam.cumsum(0) / lam.sum()
    k90 = int((cum < 0.90).sum().item()) + 1
    k99 = int((cum < 0.99).sum().item()) + 1
    return pr, k90, k99, lam, mu, vh


def grok_step(seed: int) -> int:
    """First logged step with val_acc >= 0.99."""
    metrics = json.loads((RUNS / f"seed{seed}" / "metrics.json").read_text())
    return next(r["step"] for r in metrics if r["val_acc"] >= 0.99)


def main() -> None:
    """Compute per-checkpoint PCA/pruning stats, pruning sanity checks, and the figure."""
    results: dict[int, list[dict[str, float]]] = {}
    table: list[str] = []
    for seed in SEEDS:
        ckpts = sorted((RUNS / f"seed{seed}").glob("step_*.pt"))
        _, c0 = load_model(ckpts[0])
        p = int(c0["p"])
        tr = train_pairs(p, float(c0["train_frac"]), seed)
        tr_set = set(tr)
        te = [(a, b) for a in range(p) for b in range(p) if (a, b) not in tr_set]
        xtr, ytr = make_batch(tr, p)
        xte, yte = make_batch(te, p)
        g = grok_step(seed)
        check_steps = {5000, (g - 1) // 500 * 500, int(ckpts[-1].stem.split("_")[1])}
        rows: list[dict[str, float]] = []
        for path in ckpts:
            m, ck = load_model(path)
            step = int(ck["step"])
            logits_tr, act = run(m, xtr)
            pr, k90, k99, _, mu, vh = pca_stats(act)
            var = act.double().var(0)
            dead = int((var == 0).sum())
            low = int((var < 0.01 * var.median()).sum())
            logits_te, _ = run(m, xte)
            rows.append(
                {
                    "step": step,
                    "pr": pr,
                    "k90": k90,
                    "k99": k99,
                    "dead": dead,
                    "low": low,
                    "train_acc": acc(logits_tr, ytr),
                    "test_acc": acc(logits_te, yte),
                }
            )
            if step in check_steps:
                keep = torch.ones_like(var, dtype=torch.float32)
                keep[var.argsort()[: var.numel() // 2]] = 0.0
                basis = vh[:k99]

                def prune(o: Tensor, keep: Tensor = keep) -> Tensor:
                    """Zero the lower-variance half of the neurons."""
                    return o * keep

                def project(
                    o: Tensor, mu: Tensor = mu, basis: Tensor = basis
                ) -> Tensor:
                    """Project activations onto the top-k99 principal directions."""
                    c = o.double() - mu
                    return (mu + c @ basis.T @ basis).float()

                line = f"seed{seed} step{step:>6} k99={k99:>3} | base tr/te {rows[-1]['train_acc']:.3f}/{rows[-1]['test_acc']:.3f}"
                for name, fn in (("prune50", prune), ("projk99", project)):
                    a_tr = acc(run(m, xtr, fn)[0], ytr)
                    a_te = acc(run(m, xte, fn)[0], yte)
                    line += f" | {name} {a_tr:.3f}/{a_te:.3f}"
                table.append(line)
        results[seed] = rows
        print(f"seed{seed}: grok step {g}")
        for r in rows:
            if (
                r["step"]
                in (
                    0,
                    500,
                    1000,
                    2500,
                    5000,
                    10000,
                    15000,
                    (g - 1) // 500 * 500,
                    g // 500 * 500 + 500,
                    35000,
                )
                or r["step"] % 2500 == 0
            ):
                print(
                    f"  step {r['step']:>6} PR {r['pr']:6.1f} k90 {r['k90']:>3} k99 {r['k99']:>3} "
                    f"dead {r['dead']:>3} low {r['low']:>3} tr {r['train_acc']:.3f} te {r['test_acc']:.3f}"
                )
    print("\n".join(table))
    (FIG.parent / "pca_rank.json").write_text(json.dumps(results))
    plot(results)


def plot(results: dict[int, list[dict[str, float]]]) -> None:
    """Draw the 3x3 figure (rows: test acc, dimensionality, prunable neurons; cols: seeds)."""
    blue, orange, muted = "#2a78d6", "#eb6834", "#52514e"
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.edgecolor": "#b5b3ad",
            "axes.labelcolor": muted,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.5), sharex=True)
    for j, seed in enumerate(SEEDS):
        rows = results[seed]
        s = [r["step"] for r in rows]
        g = grok_step(seed)
        series = [
            [("test accuracy", [r["test_acc"] for r in rows])],
            [
                ("participation ratio", [r["pr"] for r in rows]),
                ("#PCs for 99% var", [r["k99"] for r in rows]),
            ],
            [
                ("<1% median var", [r["low"] for r in rows]),
                ("dead", [r["dead"] for r in rows]),
            ],
        ]
        titles = ["Test accuracy", "Effective dimensionality", "Prunable neurons"]
        for i, ss in enumerate(series):
            ax = axes[i, j]
            for (lab, v), c in zip(ss, (blue, orange)):
                ax.plot(s, v, color=c, lw=2, label=lab)
            ax.axvline(g, color=muted, ls="--", lw=1)
            ax.set_title(f"{titles[i]} - seed {seed}", loc="left", fontsize=10)
            if len(ss) == 2:
                ax.legend(frameon=False, fontsize=8)
            if i == 2:
                ax.set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)


if __name__ == "__main__":
    main()
