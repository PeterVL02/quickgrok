"""Make the teaching figures for intuition/README.md.

Run from the repo root: uv run python intuition/make_figures.py
Reads (CPU only) runs/mlp_baseline/seed0/ and runs/t6/ and writes PNGs to intuition/figures/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from quickgrok.accel import harmonic_basis
from quickgrok.data import make_dataset
from quickgrok.models import load_model

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "mlp_baseline" / "seed0"
T6 = ROOT / "runs" / "t6"
ARMS = {  # run folder -> plain-language label, in plotting order
    "vanilla": "normal training",
    "project": "snap once",
    "module": "wave module",
    "module_random_k": "wave module, wrong frequency",
    "module_all_step0": "all waves from the start",
    "module_all_step0_random_k": "all waves from the start, wrong frequencies",
}
SWAP_NEURON = (
    15,
    2,
    3_000,
)  # neuron, fitted frequency, swap step: first event of seed 0
OUT = Path(__file__).resolve().parent / "figures"
P = 97
NEURON = (
    444  # a strong neuron of MLP seed 0, also shown in reports/figures/mlp_baseline
)

BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
COLOURS = (BLUE, ORANGE, GREEN, YELLOW, "#e87ba4", "#008300")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TEST_TINT = (
    "#f7d5c6"  # light orange for the many held-out cells, so the train cells stand out
)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    """Recessive axes: no top/right spines, light grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if grid:
        ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to intuition/figures/ at dpi 150 and close it."""
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def load_input_weights(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the MLP's input weights U (p, n), V (p, n) and hidden bias (n,) at a checkpoint."""
    model, _ = load_model(RUN / f"step_{step:06d}.pt")
    sd = model.state_dict()
    return (
        sd["embed_a.weight"].double().numpy(),
        sd["embed_b.weight"].double().numpy(),
        sd["bias"].double().numpy(),
    )


def find_frequency(y: np.ndarray) -> int:
    """Return the frequency k in 1..(p-1)/2 with the largest Fourier amplitude in `y` (length p)."""
    return int(np.abs(np.fft.rfft(y - y.mean()))[1:].argmax()) + 1


def fit_r2(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares fit of `y` on the columns of `X`; return the fitted values and R²."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    return y_hat, float(1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Columns 1, x, ..., x^degree with x rescaled to [-1, 1] for a stable solve."""
    z = 2 * x / x.max() - 1
    return np.stack([z**d for d in range(degree + 1)], axis=1)


def wave_features(x: np.ndarray, k: int, harmonics: tuple[int, ...]) -> np.ndarray:
    """Columns 1, cos(2πmkx/p), sin(2πmkx/p) for each harmonic m."""
    cols = [np.ones_like(x, dtype=float)]
    for m in harmonics:
        w = 2 * np.pi * m * k * x / P
        cols += [np.cos(w), np.sin(w)]
    return np.stack(cols, axis=1)


def plot_clock() -> None:
    """Clock arithmetic mod 12 as adding angles, and a number as a point (cos, sin)."""
    n, a, b = 12, 5, 9
    c = (a + b) % n
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"wspace": 0.3})

    def angle(x: float, k: int = 1) -> float:
        """Angle in radians of number x on the mod-n clock (frequency k)."""
        return 2 * np.pi * k * x / n

    # Panel 1: the clock, a then b as two turns.
    ax = axes[0]
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1)
    for x in range(n):
        ax.plot(
            *(0.97 * np.array([np.cos(angle(x)), np.sin(angle(x))])),
            "o",
            ms=3,
            color=MUTED,
        )
        ax.text(
            1.14 * np.cos(angle(x)),
            1.14 * np.sin(angle(x)),
            str(x),
            ha="center",
            va="center",
        )
    arc_a = np.linspace(0, angle(a), 100)
    arc_b = np.linspace(angle(a), angle(a + b), 200)
    ax.plot(
        0.55 * np.cos(arc_a),
        0.55 * np.sin(arc_a),
        color=BLUE,
        lw=3,
        label=f"turn by a = {a}",
    )
    ax.plot(
        0.72 * np.cos(arc_b),
        0.72 * np.sin(arc_b),
        color=ORANGE,
        lw=3,
        label=f"then by b = {b}",
    )
    ax.annotate(
        "",
        xy=(0.95 * np.cos(angle(c)), 0.95 * np.sin(angle(c))),
        xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 2.5},
    )
    ax.plot([], [], color=GREEN, lw=2.5, label=f"land on (a + b) mod {n} = {c}")
    ax.set_title(f"Clock with {n} hours: {a} + {b} = {a + b} → {c}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=9)

    # Panel 2: a number as a point on the unit circle.
    ax = axes[1]
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1)
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    x0 = 2
    px, py = np.cos(angle(x0)), np.sin(angle(x0))
    ax.plot([0, px], [0, py], color=BLUE, lw=2)
    ax.plot(
        [px, px],
        [0, py],
        color=ORANGE,
        lw=2,
        ls="--",
        label=f"sin(2π·{x0}/{n}) = {py:.2f}",
    )
    ax.plot([0, px], [0, 0], color=GREEN, lw=3, label=f"cos(2π·{x0}/{n}) = {px:.2f}")
    ax.plot(px, py, "o", color=BLUE, ms=8)
    ax.text(px + 0.06, py + 0.06, f"number {x0}", color=BLUE)
    ax.set_title("A number x becomes the point (cos, sin)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=9)

    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis("off")

    # Panel 3: the same cos as a wave over the number line; it repeats every n.
    ax = axes[2]
    xs = np.arange(0, 2 * n + 1)
    fine = np.linspace(0, 2 * n, 400)
    ax.plot(fine, np.cos(angle(fine)), color=BLUE, lw=1.5, label="k = 1: cos(2π·x/12)")
    ax.plot(
        fine, np.cos(angle(fine, 3)), color=ORANGE, lw=1.5, label="k = 3: cos(2π·3x/12)"
    )
    ax.plot(xs, np.cos(angle(xs)), "o", color=BLUE, ms=4)
    ax.axvline(n, color=MUTED, ls=":", lw=1)
    ax.text(n + 0.3, 1.12, "12 = 0 again", color=MUTED, fontsize=9)
    ax.set_xticks(range(0, 2 * n + 1, 3))
    ax.set_ylim(-1.25, 1.35)
    ax.set_xlabel("number x")
    ax.set_title("The same idea as waves: frequency k")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9)
    style_axes(ax)
    ax.set_box_aspect(1)
    save(fig, "clock.png")


def plot_table_split() -> None:
    """The (a + b) mod 97 table, the real 30% train split, and a zoomed corner with numbers."""
    a, b = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    table = (a + b) % P
    train, _ = make_dataset(P, "add", 0.3, seed=0)
    is_train = np.zeros((P, P), dtype=bool)
    for (x, y, _), _c in train:
        is_train[x, y] = True

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    im = ax.imshow(table, cmap="Blues", origin="upper")
    ax.set_title("Answer table: (a + b) mod 97")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    ax = axes[1]
    ax.imshow(is_train, cmap=ListedColormap([TEST_TINT, BLUE]), origin="upper")
    ax.set_title(f"Which cells the network may see ({is_train.mean():.0%})")
    ax.legend(
        handles=[
            Patch(color=BLUE, label=f"train: {is_train.sum():,} pairs"),
            Patch(color=TEST_TINT, label=f"test (hidden): {(~is_train).sum():,} pairs"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        fontsize=9,
    )
    for ax in axes[:2]:
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        ax.tick_params(colors=MUTED, labelsize=8)

    z = 12
    ax = axes[2]
    ax.imshow(is_train[:z, :z], cmap=ListedColormap([TEST_TINT, BLUE]), origin="upper")
    for x in range(z):
        for y in range(z):
            ax.text(
                y,
                x,
                str(table[x, y]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if is_train[x, y] else INK,
            )
    ax.set_xticks(range(z))
    ax.set_yticks(range(z))
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"Zoom on the top-left {z}×{z} corner")
    save(fig, "table_split.png")


def plot_story_curve() -> None:
    """Train vs test accuracy of MLP seed 0 on a log step axis, with the two phases marked."""
    hist = json.loads((RUN / "metrics.json").read_text())
    hist = [h for h in hist if h["step"] > 0]
    steps = np.array([h["step"] for h in hist])
    train = np.array([h["train_acc"] for h in hist])
    test = np.array([h["val_acc"] for h in hist])
    memo = int(steps[np.argmax(train >= 0.99)])
    grok = int(steps[np.argmax(test >= 0.99)])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        steps, train, color=BLUE, lw=2.5, label="train accuracy (pairs it has seen)"
    )
    ax.plot(
        steps,
        test,
        color=ORANGE,
        lw=2.5,
        label="test accuracy (pairs it has never seen)",
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("training step (log scale: each grid line is 10× more steps)")
    ax.set_ylabel("accuracy")
    ax.annotate(
        f"memorised\n(step {memo:,})",
        xy=(memo, 1.0),
        xytext=(memo * 2.2, 0.62),
        arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.5},
        color=BLUE,
        ha="left",
    )
    ax.annotate(
        f"understood: grokked\n(step {grok:,})",
        xy=(grok, 0.99),
        xytext=(grok / 9, 0.62),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.5},
        color=ORANGE,
        ha="left",
    )
    ax.annotate(
        "",
        xy=(grok, 0.08),
        xytext=(memo, 0.08),
        arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 1},
    )
    ax.text(
        np.sqrt(memo * grok), 0.11, f"{grok // memo}× longer", color=MUTED, ha="center"
    )
    ax.set_title("A one-hidden-layer network learning a + b mod 97 (one training run)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.35), fontsize=9)
    style_axes(ax)
    save(fig, "story_curve.png")


def neuron_map(step: int) -> np.ndarray:
    """Activation of NEURON on all (a, b): ReLU(U[a] + V[b] + bias), shape (p, p)."""
    U, V, bias = load_input_weights(step)
    pre = U[:, NEURON, None] + V[None, :, NEURON] + bias[NEURON]
    return np.maximum(pre, 0)


def plot_neuron_lens() -> None:
    """One neuron before and after grokking, in natural order and re-ordered by k·x mod p."""
    U, _, _ = load_input_weights(40_000)
    k = find_frequency(U[:, NEURON])
    order = np.argsort((k * np.arange(P)) % P)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4), gridspec_kw={"hspace": 0.3})
    for col, (step, label) in enumerate(
        [(1_000, "memorising"), (40_000, "after grokking")]
    ):
        act = neuron_map(step)
        for row, (img, how) in enumerate(
            [
                (act, "a, b in normal order"),
                (act[order][:, order], f"sorted by {k}·x mod 97"),
            ]
        ):
            ax = axes[row, col]
            ax.imshow(img, cmap="Blues", origin="upper")
            ax.set_title(f"step {step:,} ({label})\n{how}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("b")
            ax.set_ylabel("a")
    fig.suptitle(
        f"Neuron {NEURON}: dark = active. Same numbers, two orderings",
        x=0.125,
        ha="left",
        fontsize=12,
    )
    save(fig, "neuron_lens.png")


def plot_poly_vs_wave() -> None:
    """A grokked neuron's input weight U[:, i] vs a, fit with a polynomial and with waves."""
    U, _, _ = load_input_weights(40_000)
    y = U[:, NEURON]
    a = np.arange(P, dtype=float)
    k = find_frequency(y)
    poly_hat, poly_r2 = fit_r2(poly_features(a, 6), y)
    wave_hat, wave_r2 = fit_r2(wave_features(a, k, (1,)), y)
    harm_hat, harm_r2 = fit_r2(wave_features(a, k, (1, 2, 3)), y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"wspace": 0.2})
    ax = axes[0]
    ax.plot(a, y, "o", color=INK, ms=3.5, label="weight from input a to the neuron")
    ax.plot(
        a,
        poly_hat,
        color=ORANGE,
        lw=2,
        label=f"best polynomial, degree 6 (R² = {poly_r2:.2f})",
    )
    ax.plot(
        a,
        wave_hat,
        "o",
        mfc="none",
        color=BLUE,
        ms=5,
        label=f"best wave, frequency {k} (R² = {wave_r2:.2f})",
    )
    ax.set_xlabel("a (normal order 0, 1, 2, ..., 96)")
    ax.set_ylabel("input weight U[a]")
    ax.set_title("After grokking, a in normal order: looks like noise")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)

    order = np.argsort((k * a.astype(int)) % P)
    pos = np.arange(P)
    ax = axes[1]
    ax.plot(pos, y[order], "o", color=INK, ms=3.5, label="same weights, re-ordered")
    ax.plot(
        pos,
        wave_hat[order],
        color=BLUE,
        lw=2,
        label=f"wave, frequency {k} (R² = {wave_r2:.2f})",
    )
    ax.plot(
        pos,
        harm_hat[order],
        color=GREEN,
        lw=1.5,
        ls="--",
        label=f"wave + 2 harmonics (R² = {harm_r2:.2f})",
    )
    ax.set_xlabel(f"a sorted by {k}·a mod 97 (where a sits on the frequency-{k} clock)")
    ax.set_title("Same neuron, a re-ordered: a square wave, which waves can fit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    for ax in axes:
        style_axes(ax)
    fig.suptitle(
        f"Neuron {NEURON}: polynomial R² = {poly_r2:.2f}, wave R² = {wave_r2:.2f}"
        f" ({harm_r2:.2f} with harmonics)",
        x=0.1,
        y=1.02,
        ha="left",
        fontsize=12,
    )
    save(fig, "poly_vs_wave.png")


def load_history(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return logged steps and test accuracy of one run from its metrics.json."""
    hist = json.loads((run / "metrics.json").read_text())
    return np.array([h["step"] for h in hist]), np.array([h["val_acc"] for h in hist])


def grok_step(run: Path) -> int:
    """First logged step at which test accuracy reaches 99%."""
    steps, test = load_history(run)
    return int(steps[np.argmax(test >= 0.99)])


def plot_jump_idea() -> None:
    """Measured test accuracy (median of 10 seeds) for normal training and the three jump arms."""
    end = 21_000
    grid = np.arange(0, end + 1, 100)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (arm, label), color in zip(list(ARMS.items())[:4], COLOURS):
        curves = []
        for run in sorted((T6 / arm).glob("seed*")):
            steps, test = load_history(run)
            # A run stops 3,000 steps after grokking; hold its last value after that.
            curves.append(np.interp(grid, steps, test))
        ax.plot(grid, np.median(curves, axis=0), color=color, lw=2.5, label=label)
    ax.axvline(2_250, color=MUTED, ls=":", lw=1)
    ax.text(
        2_400,
        0.9,
        "first neurons swapped\n(step ~2,250)",
        color=MUTED,
        fontsize=9,
        va="top",
    )
    ax.set_xlim(0, end)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("training step")
    ax.set_ylabel("test accuracy (median of 10 seeds)")
    ax.set_title("The jump, measured: swapping wave-shaped neurons as they appear")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    style_axes(ax)
    save(fig, "jump_idea.png")


def plot_results_dots() -> None:
    """Steps to 99% test accuracy per seed for the six arms, median marked."""
    fig, ax = plt.subplots(figsize=(10, 5.4))
    rng = np.random.default_rng(0)
    base = None
    for row, ((arm, label), color) in enumerate(zip(ARMS.items(), COLOURS)):
        steps = np.array([grok_step(r) for r in sorted((T6 / arm).glob("seed*"))])
        y = -row + rng.uniform(-0.15, 0.15, len(steps))  # jitter so equal steps show
        ax.plot(steps, y, "o", color=color, ms=6, mec="white", mew=0.8)
        med = float(np.median(steps))
        base = base or med
        ax.plot([med, med], [-row - 0.3, -row + 0.3], color=INK, lw=2)
        note = f"median {med:,.0f}" + (
            "" if row == 0 else f"  ({1 - med / base:.0%} fewer)"
        )
        ax.text(med + 250, -row + 0.3, note, fontsize=9, va="bottom")
    ax.plot([], [], "o", color=MUTED, ms=6, label="one seed (one training run)")
    ax.plot([], [], color=INK, lw=2, label="median of 10 seeds")
    ax.set_yticks(-np.arange(len(ARMS)), list(ARMS.values()))
    ax.tick_params(axis="y", labelsize=10, colors=INK)
    ax.set_ylim(-len(ARMS) + 0.4, 0.8)
    ax.set_xlim(0, 21_000)
    ax.set_xlabel("training steps until test accuracy ≥ 99% (fewer is faster)")
    ax.set_title("How long each version takes to grok (10 seeds each)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    save(fig, "results_dots.png")


def outside_share(run: Path, neuron: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Share of a neuron's input-weight energy (column of U) outside its fitted waves, per checkpoint.

    The waves are the 9 basis functions used by the substitution: a constant plus cos and sin
    of frequency k and its harmonics 2k, 3k, 4k.
    """
    B = harmonic_basis([k], P, 4)[0].double().numpy()
    proj = B @ np.linalg.pinv(B)
    steps, shares = [], []
    for ckpt in sorted(run.glob("step_*.pt")):
        u = (
            load_model(ckpt)[0]
            .state_dict()["embed_a.weight"]
            .double()
            .numpy()[:, neuron]
        )
        r = u - proj @ u
        steps.append(int(ckpt.stem.removeprefix("step_")))
        shares.append(float(r @ r / (u @ u)))
    return np.array(steps), np.array(shares)


def plot_snap_vs_mould() -> None:
    """One neuron (seed 0) swapped at the same moment in a 'snap once' and a 'module' run."""
    neuron, k, swap = SWAP_NEURON
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"wspace": 0.2})
    for arm, color in zip(list(ARMS)[:3], COLOURS):
        steps, share = outside_share(T6 / arm / "seed0", neuron, k)
        if (
            arm != "vanilla"
        ):  # checkpoints are saved just before the swap; right after it is 0
            i = int(np.searchsorted(steps, swap)) + 1
            steps, share = np.insert(steps, i, swap), np.insert(share, i, 0.0)
        for ax in axes:
            ax.plot(steps, 100 * share, color=color, lw=2.2, label=ARMS[arm])
    for ax in axes:
        ax.axvline(swap, color=MUTED, ls=":", lw=1)
        ax.set_xlabel("training step")
        style_axes(ax)
    axes[0].set_ylabel("% of weight energy outside its waves")
    axes[0].text(swap + 300, 70, f"swap at step {swap:,}", color=MUTED, fontsize=9)
    axes[0].set_title(f"Neuron {neuron} (frequency {k}), seed 0: whole run")
    axes[0].legend(loc="upper right", fontsize=9)
    ax = axes[1]
    ax.set_xlim(swap - 500, 21_500)
    ax.set_ylim(-1, 20)
    ax.set_title("Zoom after the swap: snapped drifts back, moulded stays at 0")
    ax.text(12_000, 9, "snap once: back to ~7%,\nlike normal training", color=ORANGE)
    ax.text(12_000, 1.5, "wave module: held at 0%", color=GREEN)
    save(fig, "snap_vs_mould.png")


def main() -> None:
    """Make every figure."""
    plot_clock()
    plot_table_split()
    plot_story_curve()
    plot_neuron_lens()
    plot_poly_vs_wave()
    plot_jump_idea()
    plot_results_dots()
    plot_snap_vs_mould()


if __name__ == "__main__":
    main()
