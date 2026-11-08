"""Check model shapes and the layer selector."""

import torch

from quickgrok.models import GrokMLP, GrokTransformer, build_model
from quickgrok.regress import make_grid, record_neurons


def test_record_neurons() -> None:
    """Every model/layer returns (p*p, width) activations; MLP layer 0 is ReLU(U[a] + V[b] + bias)."""
    p = 11
    torch.manual_seed(0)
    mlp = build_model("mlp", p, n_hidden=2)
    assert isinstance(mlp, GrokMLP)
    for layer in (0, 1):
        assert record_neurons(mlp, layer=layer).shape == (p * p, 512)
    assert record_neurons(mlp, [3, 7], layer=1).shape == (p * p, 2)

    a, b = make_grid(p)
    expected = torch.relu(mlp.embed_a.weight[a] + mlp.embed_b.weight[b] + mlp.bias)
    assert torch.allclose(record_neurons(mlp).float(), expected.detach(), atol=1e-6)

    transformer = build_model("transformer", p)
    assert isinstance(transformer, GrokTransformer)
    assert record_neurons(transformer).shape == (p * p, 512)
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    assert mlp(x).shape == transformer(x).shape == (p * p, p)
