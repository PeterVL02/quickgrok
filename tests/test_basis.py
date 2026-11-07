"""Check the weight-based substitution bases, the energy trigger and the S5 task.

Run: uv run python tests/test_basis.py
"""

import math
import random
import tempfile
from pathlib import Path

import torch

from quickgrok.accel import (
    BasisName,
    SubstitutionHook,
    captured_energy,
    make_basis,
    project,
    random_basis,
)
from quickgrok.data import S5, S5_ORDER, compose_s5, make_dataset
from quickgrok.models import GrokMLP
from quickgrok.regress import score_r2
from quickgrok.train import to_tensors

N, D, M = 97, 512, 9
X = torch.linspace(-1, 1, N, dtype=torch.float64)
GRID = torch.arange(N, dtype=torch.float64)


def noisy_weights(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random U, V (N, D) in float64."""
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(N, D, generator=g, dtype=torch.float64)
    return U, torch.randn(N, D, generator=g, dtype=torch.float64)


def energy_of(
    name: BasisName, U: torch.Tensor, V: torch.Tensor, neurons: list[int]
) -> torch.Tensor:
    """Captured energy of `neurons` in basis `name`."""
    bases, _, _ = make_basis(name, U, V, neurons, M, seed=0)
    return captured_energy(bases, U[:, neurons].T, V[:, neurons].T)


def test_shapes() -> None:
    """Every basis is (k, n, m), full column rank, with a constant first column."""
    U, V = noisy_weights()
    neurons = [0, 3, 7, 100]
    names_all: list[BasisName] = ["fourier", "poly", "svd", "random", "auto"]
    for name in names_all:
        (basis, basis_v), names, ks = make_basis(name, U, V, neurons, M, seed=0)
        assert basis is basis_v and basis.shape == (len(neurons), N, M), name
        assert len(names) == len(ks) == len(neurons)
        assert (torch.linalg.matrix_rank(basis) == M).all(), name
        first = basis[:, :, 0]
        assert (first.std(-1) < 1e-9).all() and (first.abs() > 0).all(), name


def test_capture() -> None:
    """fourier: a pure harmonic; poly: a cubic; svd: columns of a low-rank matrix."""
    U, V = noisy_weights()
    U[:, 0] = torch.cos(2 * math.pi * 5 * GRID / N + 0.3) + 2
    V[:, 0] = torch.sin(2 * math.pi * 15 * GRID / N)  # 15 = 3 * 5, a harmonic of 5
    bases, _, ks = make_basis("fourier", U, V, [0], M, seed=0)
    assert ks == [5]
    assert captured_energy(bases, U[:, :1].T, V[:, :1].T).item() > 1 - 1e-9

    U[:, 1], V[:, 1] = X**3 - X, 2 * X**2 + 1
    assert energy_of("poly", U, V, [1]).item() > 1 - 1e-9

    g = torch.Generator().manual_seed(1)
    A = torch.randn(N, 5, generator=g, dtype=torch.float64)
    low_U, low_V = (
        A @ torch.randn(5, D, generator=g, dtype=torch.float64) for _ in "uv"
    )
    assert (energy_of("svd", low_U, low_V, [0, 1, 2]) > 1 - 1e-9).all()
    assert (energy_of("svd", U, V, [2, 3]) < 0.5).all()  # noise is not captured


def test_auto() -> None:
    """auto picks fourier for a cosine column and poly for a quadratic one."""
    U, V = noisy_weights()
    U[:, 0] = V[:, 0] = torch.cos(2 * math.pi * 7 * GRID / N)
    U[:, 1] = V[:, 1] = 3 * X**2 + X
    _, names, ks = make_basis("auto", U, V, [0, 1], M, seed=0)
    assert names == ["fourier", "poly"] and ks == [7, None]


def test_levels() -> None:
    """levels_L fits u and v separately: L-valued columns are captured fully, the bases differ."""
    U, V = noisy_weights()
    U[:, 0] = torch.where(torch.cos(2 * math.pi * 3 * GRID / N) > 0, 2.0, -1.0)
    V[:, 0] = (GRID % 3) * 1.5  # three levels, a different partition
    (bu, bv), names, _ = make_basis("levels_3", U, V, [0, 1], M, seed=0)
    assert bu.shape == bv.shape == (2, N, 3) and names == ["levels_3"] * 2
    assert torch.allclose(
        (bu > 0).sum(-1), torch.ones(2, N, dtype=torch.long)
    )  # a partition
    assert not torch.equal(bu[0], bv[0])
    energy = captured_energy((bu, bv), U[:, [0, 1]].T, V[:, [0, 1]].T)
    assert energy[0] > 1 - 1e-9 and energy[1] < 0.95  # Gaussian noise: ~0.82 expected
    assert energy_of("levels_2", U, V, [0]) < 1 - 1e-3  # v has three levels


def test_random_seeded() -> None:
    """random is reproducible for a fixed seed and differs across seeds and neurons."""
    a, b = random_basis([4, 9], N, M, seed=3), random_basis([4, 9], N, M, seed=3)
    assert torch.equal(a, b)
    assert not torch.allclose(a, random_basis([4, 9], N, M, seed=4))
    assert not torch.allclose(a[0], a[1])
    assert torch.equal(random_basis([9], N, M, seed=3)[0], a[1])


def test_s5() -> None:
    """14,400 entries, identity, inverses, associativity and the a∘b = a(b(x)) convention."""
    train, val = make_dataset(S5_ORDER, "s5", 0.3, seed=0)
    table = {(a, b): c for (a, b, _), c in train + val}
    assert len(train) + len(val) == len(table) == 14_400
    assert all(eq == S5_ORDER for (_, _, eq), _ in train)
    e = S5.index((0, 1, 2, 3, 4))
    assert all(table[e, a] == table[a, e] == a for a in range(S5_ORDER))
    assert all(any(table[a, b] == e for b in range(S5_ORDER)) for a in range(S5_ORDER))
    rng = random.Random(0)
    for _ in range(2000):
        a, b, c = (rng.randrange(S5_ORDER) for _ in range(3))
        assert table[table[a, b], c] == table[a, table[b, c]]
    a, b = 17, 64
    assert S5[compose_s5(a, b)] == tuple(S5[a][S5[b][x]] for x in range(5))
    assert (
        len({table[0, b] for b in range(S5_ORDER)}) == S5_ORDER
    )  # a group: rows are permutations
    assert any(table[a, b] != table[b, a] for a, b in table)  # not abelian


def test_check_r2_uses_training_check_rows() -> None:
    """The trigger's R^2 is measured exactly on the check split of the training rows."""
    train, val = make_dataset(N, "add", 0.3, seed=100)
    x_train, y_train = to_tensors(train, "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        hook = SubstitutionHook(
            x_train,
            N,
            Path(tmp) / "probe.jsonl",
            every=2,
            seed=100,
            mode="module",
            tau_share=0.7,
            tau_r2=0.8,
            y_train=y_train,
            log_path=Path(tmp) / "subs.jsonl",
            basis="svd",
        )
    check = set(zip(hook.check_a.tolist(), hook.check_b.tolist()))
    pairs = [tuple(r) for r in x_train[:, :2].tolist()]
    fit = {q for q, m in zip(pairs, hook.fit_rows.tolist()) if m}
    assert check == set(pairs) - fit and len(check) == len(train) - len(fit)
    assert not check & {(a, b) for (a, b, _), _ in val}

    torch.manual_seed(0)
    model = GrokMLP(N)
    neurons = [0, 1, 2]
    U, V, bias = hook.input_weights(model)
    bases, _, _ = make_basis("levels_3", U, V, neurons, M, seed=0)
    u, v = U[:, neurons].T, V[:, neurons].T
    pu, pv = project(bases[0], u), project(bases[1], v)

    def r2_on(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Activation R^2 of the projected neurons on the rows (a, b)."""
        c = bias[neurons, None]
        act = torch.relu(u[:, a] + v[:, b] + c).T
        return score_r2(act, torch.relu(pu[:, a] + pv[:, b] + c).T)

    r2 = hook.check_r2(model, neurons, bases)
    assert torch.allclose(r2, r2_on(hook.check_a, hook.check_b))
    x_val, _ = to_tensors(val, "cpu")
    assert not torch.allclose(r2, r2_on(x_val[:, 0], x_val[:, 1]))


if __name__ == "__main__":
    test_shapes()
    test_capture()
    test_auto()
    test_levels()
    test_random_seeded()
    test_s5()
    test_check_r2_uses_training_check_rows()
    print("ok")
