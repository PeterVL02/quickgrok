"""Models for modular arithmetic: a 1-layer transformer and an MLP, plus checkpoint loading."""

from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import torch
from torch import Tensor, nn

from quickgrok.data import Operation

ModelName = Literal["transformer", "mlp"]


class Checkpoint(TypedDict):
    model: NotRequired[
        ModelName
    ]  # missing in pilot checkpoints, which are all "transformer"
    n_hidden: NotRequired[int]  # MLP only
    p: int
    op: Operation
    train_frac: float
    seed: int
    step: int
    train_acc: float
    val_acc: float
    state_dict: dict[str, Tensor]


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


class GrokMLP(nn.Module):
    def __init__(self, p: int, d_hidden: int = 512, n_hidden: int = 1) -> None:
        super().__init__()
        # One-hot a and b straight into the first hidden layer: pre-activation U[a] + V[b] + bias.
        self.embed_a = nn.Embedding(p, d_hidden)  # U
        self.embed_b = nn.Embedding(p, d_hidden)  # V
        self.bias = nn.Parameter(torch.zeros(d_hidden))
        layers: list[nn.Module] = [nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU()]
        self.hidden = nn.Sequential(
            *layers
        )  # every ReLU here is a selectable neuron layer
        self.unembed = nn.Linear(d_hidden, p)

    def forward(self, x: Tensor) -> Tensor:
        """Map token sequences (N, 3) to logits (N, p); the "=" token is ignored."""
        h = self.embed_a(x[:, 0]) + self.embed_b(x[:, 1]) + self.bias
        return self.unembed(self.hidden(h))


GrokModel = GrokTransformer | GrokMLP


def build_model(name: ModelName, p: int, n_hidden: int = 1) -> GrokModel:
    """Construct a fresh model by name; `n_hidden` only applies to the MLP."""
    return (
        GrokTransformer(p) if name == "transformer" else GrokMLP(p, n_hidden=n_hidden)
    )


def get_activation_layers(model: GrokModel) -> list[nn.Module]:
    """The activation modules whose outputs count as neurons, ordered by depth."""
    if isinstance(model, GrokTransformer):
        return [model.mlp[1]]
    return [m for m in model.hidden if isinstance(m, nn.ReLU)]


def load_model(path: Path) -> tuple[GrokModel, Checkpoint]:
    """Load a checkpoint onto the CPU and return the model in eval mode and the checkpoint."""
    ckpt: Checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(
        ckpt.get("model", "transformer"), ckpt["p"], ckpt.get("n_hidden", 1)
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt
