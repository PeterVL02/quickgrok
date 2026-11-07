"""Check the probe split, the harmonic share, the hook API and substitution.

Run: uv run python tests/test_accel.py
"""

import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from quickgrok.accel import (
    Accelerator,
    Hook,
    ProbeHook,
    SubstitutionHook,
    SubstitutionMode,
    SubstitutionRecord,
    effective_state_dict,
    expand,
    fit_coef,
    harmonic_basis,
)
from quickgrok.data import make_dataset
from quickgrok.models import GrokMLP, build_model
from quickgrok.regress import fit_neurons, harmonic_share, make_grid
from quickgrok.train import to_tensors


def test_probe_split() -> None:
    """Fit and check rows partition the training rows and contain no test pair; probe runs."""
    p = 97
    train, val = make_dataset(p, "add", 0.3, seed=100)
    x_train, _ = to_tensors(train, "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.jsonl"
        hook = ProbeHook(x_train, p, path, every=2, seed=100)
        pairs = [tuple(r) for r in x_train[:, :2].tolist()]
        fit = {q for q, m in zip(pairs, hook.fit_rows.tolist()) if m}
        check = {q for q, m in zip(pairs, hook.fit_rows.tolist()) if not m}
        assert not fit & check and fit | check == {(a, b) for (a, b, _), _ in train}
        assert not (fit | check) & {(a, b) for (a, b, _), _ in val}
        assert abs(len(fit) / len(train) - 0.8) < 0.01

        model = build_model("mlp", p)
        opt = torch.optim.SGD(model.parameters())
        accel = Accelerator(model, [hook])
        for step in range(3):
            accel.after_step(step, opt)
        records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in records] == [0, 2]
    assert all(len(r["harmonic_share"]) == len(r["r2_check"]) == 512 for r in records)


def test_harmonic_share() -> None:
    """A square wave in k*a mod p is ~fully explained by harmonics of k, not by k alone."""
    p, k = 97, 7
    a, b = make_grid(p)
    wave = torch.sign(torch.cos(2 * math.pi * k * a.double() / p))
    fit = fit_neurons(wave[:, None], a, b, p, "trig", 0, [0])[0]
    base, share = harmonic_share(torch.tensor(fit["freq_shares"]), p)
    assert fit["top_freq"] == k and base.item() == k
    # Ideal square wave: harmonics k and 3k hold (1 + 1/9) / (pi^2 / 8) = 0.90 of the variance.
    assert share.item() > 0.89 and fit["top_freq_share"] < 0.82
    assert harmonic_share(torch.tensor(fit["freq_shares"]), p, 20)[1] > 0.98


def test_optimizer_replacement() -> None:
    """A hook returning an optimizer replaces it; a no-op hook keeps it."""
    model = nn.Linear(2, 2)
    old, new = torch.optim.SGD(model.parameters()), torch.optim.Adam(model.parameters())

    class SwapHook(Hook):
        def after_step(
            self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer
        ) -> torch.optim.Optimizer | None:
            """Swap in `new` at step 1."""
            return new if step == 1 else None

    accel = Accelerator(model, [Hook(), SwapHook(), Hook()])
    assert accel.after_step(0, old) is old
    assert accel.after_step(1, old) is new


P = 97
NEURONS, KS = [0, 5, 9], [3, 7, 20]


def make_hook(
    mode: SubstitutionMode,
    tmp: Path,
    tau_share: float = 0.7,
    tau_r2: float = 0.8,
    random_k: bool = False,
    sub_all_at: int | None = None,
) -> SubstitutionHook:
    """A substitution hook (probe every 2 steps) on the seed-100 split, logging into `tmp`."""
    x_train, y_train = to_tensors(make_dataset(P, "add", 0.3, seed=100)[0], "cpu")
    return SubstitutionHook(
        x_train,
        P,
        tmp / f"{mode}_probe.jsonl",
        every=2,
        seed=100,
        mode=mode,
        tau_share=tau_share,
        tau_r2=tau_r2,
        y_train=y_train,
        log_path=tmp / f"{mode}_subs.jsonl",
        random_k=random_k,
        sub_all_at=sub_all_at,
    )


def run_hook(hook: SubstitutionHook, steps: int) -> list[SubstitutionRecord]:
    """Call the hook's `after_step` on a fresh MLP for `steps` steps; return the logged events."""
    model, opt = make_mlp()
    for step in range(steps):
        opt = hook.after_step(step, model, opt) or opt
    return [json.loads(line) for line in hook.log_path.read_text().splitlines()]


def make_mlp() -> tuple[GrokMLP, torch.optim.Optimizer]:
    """A fixed-seed GrokMLP and its AdamW optimizer."""
    torch.manual_seed(0)
    model = GrokMLP(P)
    return model, torch.optim.AdamW(model.parameters(), weight_decay=1.0)


def full_grid() -> torch.Tensor:
    """All p*p token inputs (N, 3)."""
    a, b = make_grid(P)
    return torch.stack([a, b, torch.full_like(a, P)], dim=1)


def test_projection() -> None:
    """A column in the harmonic span is a fixed point; projecting noise shrinks it."""
    basis = harmonic_basis(KS, P, 4)
    in_span = expand(basis, torch.randn(len(KS), basis.shape[-1]))
    assert torch.allclose(expand(basis, fit_coef(basis, in_span)), in_span, atol=1e-5)
    noise = torch.randn(len(KS), P)
    projected = expand(basis, fit_coef(basis, noise))
    assert (projected.norm(dim=1) < 0.6 * noise.norm(dim=1)).all()  # 9 of 97 dims kept


def test_module_matches_project() -> None:
    """Right after the same swap both modes give the same outputs; module grads go to coef only."""
    x = full_grid()
    with tempfile.TemporaryDirectory() as tmp:
        outputs = []
        for mode in ("project", "module"):
            model, opt = make_mlp()
            before = model(x)
            make_hook(mode, Path(tmp)).substitute(model, NEURONS, KS, opt)
            outputs.append(model(x))
        assert (outputs[0] - before).abs().max() > 1e-3  # the swap changed something
        assert (outputs[0] - outputs[1]).abs().max() < 1e-5

    outputs[1].logsumexp(-1).sum().backward()
    for embed in (model.embed_a, model.embed_b):
        coef = embed.parametrizations.weight[0].coef
        original = embed.parametrizations.weight.original
        assert coef.grad is not None and coef.grad.abs().min() > 0
        assert original.grad is not None
        assert (original.grad[:, NEURONS] == 0).all()
        assert original.grad[:, 1].abs().max() > 0


def test_grow_optimizer() -> None:
    """Growing rebuilds the optimizer over every parameter once, carrying over state."""
    model, opt = make_mlp()
    x, y = full_grid(), torch.zeros(P * P, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmp:
        hook = make_hook("module", Path(tmp))

        def train_step(opt: torch.optim.Optimizer) -> None:
            """One AdamW step on a dummy objective."""
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

        train_step(opt)
        opt = hook.substitute(model, NEURONS[:2], KS[:2], opt)
        train_step(opt)
        coef = model.embed_a.parametrizations.weight[0].coef
        old_coef_state = dict(opt.state[coef])
        old_unembed_state = opt.state[model.unembed.weight]
        new_opt = hook.substitute(model, NEURONS[2:], KS[2:], opt)

    params = new_opt.param_groups[0]["params"]
    assert len(params) == len({id(q) for q in params}) == len(list(model.parameters()))
    assert {id(q) for q in params} == {id(q) for q in model.parameters()}
    assert (
        new_opt.state[model.unembed.weight]["exp_avg"] is old_unembed_state["exp_avg"]
    )
    new_coef = model.embed_a.parametrizations.weight[0].coef
    state = new_opt.state[new_coef]
    assert new_coef.shape[0] == 3 and state["exp_avg"].shape == new_coef.shape
    assert torch.equal(state["exp_avg"][:2], old_coef_state["exp_avg"])
    assert (state["exp_avg_sq"][2:] == 0).all() and state["step"] == old_coef_state[
        "step"
    ]
    assert new_opt.param_groups[0]["weight_decay"] == 1.0
    train_step(new_opt)  # the rebuilt optimizer steps without error


def test_effective_state_dict() -> None:
    """A module-mode model's effective weights load into a plain GrokMLP with equal outputs."""
    model, opt = make_mlp()
    with tempfile.TemporaryDirectory() as tmp:
        make_hook("module", Path(tmp)).substitute(model, NEURONS, KS, opt)
    fresh = GrokMLP(P)
    fresh.load_state_dict(effective_state_dict(model))
    x = full_grid()
    assert torch.allclose(fresh(x), model(x), atol=1e-6)


def test_random_k() -> None:
    """Random k never equals the fitted k*, and the same seed gives the same draws."""
    with tempfile.TemporaryDirectory() as tmp:
        runs = [
            run_hook(make_hook("project", Path(tmp), 0, -math.inf, random_k=True), 1)
            for _ in range(2)
        ]
    assert runs[0][0]["k"] == runs[1][0]["k"]
    event = runs[0][0]
    assert len(event["k"]) == 512
    assert all(k != f for k, f in zip(event["k"], event["k_fitted"]))


def test_sub_all_at() -> None:
    """With sub_all_at=2 every live neuron is substituted once at step 2, despite tau_r2 > 1."""
    with tempfile.TemporaryDirectory() as tmp:
        hook = make_hook("module", Path(tmp), 1, 2, sub_all_at=2)
        events = run_hook(hook, 7)
        probes = [json.loads(line) for line in hook.out_path.read_text().splitlines()]
    assert [e["step"] for e in events] == [2] and [r["step"] for r in probes] == [
        0,
        2,
        4,
        6,
    ]
    live = [i for i, k in enumerate(probes[1]["top_freq"]) if k is not None]
    assert events[0]["neurons"] == live and events[0]["n_substituted_total"] == len(
        live
    )


if __name__ == "__main__":
    test_probe_split()
    test_harmonic_share()
    test_optimizer_replacement()
    test_projection()
    test_module_matches_project()
    test_grow_optimizer()
    test_effective_state_dict()
    test_random_k()
    test_sub_all_at()
    print("ok")
