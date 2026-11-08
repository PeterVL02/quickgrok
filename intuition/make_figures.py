"""Make the figures for intuition/README.md.

Run from the repo root: uv run python intuition/make_figures.py
Reads (CPU only) runs under the run roots and writes PNGs to intuition/figures/:
clock.png needs no runs; story_curve, neuron_lens and nowcast_idea need mlp_baseline/seed0;
results_dots needs t6; baseline_lesson needs tuned_vanilla; tuned_results needs
tuned_retest. The run roots default to ./runs. To read runs kept elsewhere (for example in
the main checkout), set QUICKGROK_RUN_ROOTS to folders separated by os.pathsep (";" on
Windows, ":" elsewhere). A figure whose runs are missing is skipped with a warning and its
committed PNG is left as it is.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quickgrok.models import load_model

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOTS = [
    Path(p)
    for p in os.environ.get("QUICKGROK_RUN_ROOTS", str(ROOT / "runs")).split(os.pathsep)
    if p
]
RUN = "mlp_baseline/seed0"  # run folders, relative to a run root
T6 = "t6"
TUNED_ARMS = {  # runs/tuned_retest folder -> label, wall-clock to 99% × vanilla
    "vanilla": ("normal training", 1.00),
    "nowcast_linear": ("nowcast", 0.80),
    "module": ("wave module", 1.53),
    "module_random_k": ("wave module, wrong frequency", 1.90),
    "nowcast_plus_module": ("nowcast + wave module", 1.50),
}  # wall-clock: reports/tuned_retest.md, seeds 0-2 run alone, start-up removed
OLD_SETTING_METHODS = {"nowcast": 11_150, "wave module": 10_800}  # lr 1e-3, wd 1
ARMS = {  # run folder -> plain-language label, in plotting order
    "vanilla": "normal training",
    "project": "snap once",
    "module": "wave module",
    "module_random_k": "wave module, wrong frequency",
    "module_all_step0": "all waves from the start",
    "module_all_step0_random_k": "all waves from the start, wrong frequencies",
}
OUT = Path(__file__).resolve().parent / "figures"
P = 97
NEURON = (
    444  # a strong neuron of MLP seed 0, also shown in reports/figures/mlp_baseline
)

BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
COLOURS = (BLUE, ORANGE, GREEN, YELLOW, "#e87ba4", "#008300")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

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


def find_run(rel: str) -> Path:
    """Return the folder `rel` under the first run root that has it.

    Raises:
        FileNotFoundError: if no run root contains `rel`.
    """
    for root in RUN_ROOTS:
        if (root / rel).exists():
            return root / rel
    raise FileNotFoundError(f"{rel} not found under {[str(r) for r in RUN_ROOTS]}")


def load_input_weights(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the MLP's input weights U (p, n), V (p, n) and hidden bias (n,) at a checkpoint."""
    model, _ = load_model(find_run(RUN) / f"step_{step:06d}.pt")
    sd = model.state_dict()
    return (
        sd["embed_a.weight"].double().numpy(),
        sd["embed_b.weight"].double().numpy(),
        sd["bias"].double().numpy(),
    )


def find_frequency(y: np.ndarray) -> int:
    """Return the frequency k in 1..(p-1)/2 with the largest Fourier amplitude in `y` (length p)."""
    return int(np.abs(np.fft.rfft(y - y.mean()))[1:].argmax()) + 1


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


def plot_story_curve() -> None:
    """Train vs test accuracy of MLP seed 0 on a log step axis, with the two phases marked."""
    hist = json.loads((find_run(RUN) / "metrics.json").read_text())
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


def load_history(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return logged steps and test accuracy of one run from its metrics.json."""
    hist = json.loads((run / "metrics.json").read_text())
    return np.array([h["step"] for h in hist]), np.array([h["val_acc"] for h in hist])


def grok_step(run: Path) -> int:
    """First logged step at which test accuracy reaches 99%."""
    steps, test = load_history(run)
    return int(steps[np.argmax(test >= 0.99)])


def draw_seed_dots(ax: plt.Axes, rows: list[tuple[str, list[Path]]], dx: float) -> None:
    """Draw one row per arm: steps to 99% per seed as dots, the median as a black bar.

    Each median is labelled with its % fewer steps than the first row; `dx` is the label's
    x offset from the bar, in steps.
    """
    rng = np.random.default_rng(0)
    base = None
    for row, ((label, runs), color) in enumerate(zip(rows, COLOURS)):
        steps = np.array([grok_step(r) for r in runs])
        y = -row + rng.uniform(-0.15, 0.15, len(steps))  # jitter so equal steps show
        ax.plot(steps, y, "o", color=color, ms=6, mec="white", mew=0.8)
        med = float(np.median(steps))
        base = base or med
        ax.plot([med, med], [-row - 0.3, -row + 0.3], color=INK, lw=2)
        note = f"median {med:,.0f}" + (
            "" if row == 0 else f"  ({1 - med / base:.0%} fewer)"
        )
        ax.text(med + dx, -row + 0.3, note, fontsize=9, va="bottom")
    ax.plot([], [], "o", color=MUTED, ms=6, label="one seed (one training run)")
    ax.plot([], [], color=INK, lw=2, label=f"median of {len(rows[0][1])} seeds")
    ax.set_yticks(-np.arange(len(rows)), [label for label, _ in rows])
    ax.tick_params(axis="y", labelsize=10, colors=INK)
    ax.set_ylim(-len(rows) + 0.4, 0.8)


def plot_results_dots() -> None:
    """Steps to 99% test accuracy per seed for the six arms, median marked."""
    fig, ax = plt.subplots(figsize=(10, 5.4))
    rows = [
        (label, sorted(find_run(f"{T6}/{arm}").glob("seed*")))
        for arm, label in ARMS.items()
    ]
    draw_seed_dots(ax, rows, dx=250)
    ax.set_xlim(0, 21_000)
    ax.set_xlabel("training steps until test accuracy ≥ 99% (fewer is faster)")
    ax.set_title("How long each version takes to grok (10 seeds each)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    save(fig, "results_dots.png")


def plot_baseline_lesson() -> None:
    """Plain training: median steps to 99% test accuracy vs lr × wd (log-log), one line per wd."""
    root = find_run("tuned_vanilla")
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for wd, color in zip((0.5, 1.0, 2.0), COLOURS):
        xs, meds = [], []
        for folder in sorted(root.glob(f"lr*_wd{wd}")):
            lr = float(folder.name.removeprefix("lr").split("_wd")[0])
            hists = [load_history(r) for r in sorted(folder.glob("seed*"))]
            if not all((test >= 0.99).any() for _, test in hists):
                print(f"warning: not every seed grokked at lr {lr:g}, wd {wd:g}")
                continue
            steps = [int(s[np.argmax(t >= 0.99)]) for s, t in hists]
            ax.plot([lr * wd] * len(steps), steps, "o", color=color, ms=4, alpha=0.3)
            xs.append(lr * wd)
            meds.append(float(np.median(steps)))
        order = np.argsort(xs)
        ax.plot(
            np.array(xs)[order],
            np.array(meds)[order],
            "o-",
            color=color,
            lw=2,
            ms=7,
            label=f"weight decay {wd:g} (lr 1e-3 … 1.6e-2)",
        )
    x = np.array([0.0008, 0.04])
    ax.plot(x, 17.9 / x, color=MUTED, ls="--", lw=1, label="∝ 1 / (lr × wd)")
    for (name, steps), color, va in zip(
        OLD_SETTING_METHODS.items(), COLOURS[3:], ("bottom", "top")
    ):
        ax.axhline(steps, color=color, lw=1.5, ls=":")
        ax.text(
            0.033,
            steps * (1.04 if va == "bottom" else 0.96),
            f"{name} at the old setting: {steps:,}",
            color=color,
            ha="right",
            va=va,
            fontsize=9,
        )
    ax.annotate(
        "our original baseline\nlr 1e-3, wd 1: 17,900",
        xy=(0.001, 17_900),
        xytext=(0.00125, 30_000),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1},
        fontsize=9,
    )
    ax.annotate(
        "just double the learning rate:\n8,700",
        xy=(0.002, 8_700),
        xytext=(0.00085, 2_600),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1},
        fontsize=9,
    )
    ax.text(
        0.0042,
        24_000,
        "best setting (lr 1.6e-2, wd 2): ~1,000 steps\n"
        "weight decay 4 (not shown): most seeds never grok within 40k steps",
        color=MUTED,
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    yticks = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000]
    ax.set_yticks(yticks, [f"{t:,}" for t in yticks])
    ax.minorticks_off()
    ax.set_xlim(0.0008, 0.04)
    ax.set_ylim(500, 50_000)
    ax.set_xlabel("learning rate × weight decay (log scale)")
    ax.set_ylabel("steps until test accuracy ≥ 99% (log scale)")
    ax.set_title(
        "Plain training, 5 seeds per setting (faint dots), median connected: "
        "the old baseline was slow"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    style_axes(ax)
    save(fig, "baseline_lesson.png")


def plot_tuned_results() -> None:
    """Tuned setting (lr 4e-3, wd 2): steps to 99% per seed, and wall-clock vs normal training."""
    rows = [
        (label, sorted(find_run(f"tuned_retest/{arm}").glob("seed*")))
        for arm, (label, _) in TUNED_ARMS.items()
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.08},
    )
    ax = axes[0]
    draw_seed_dots(ax, rows, dx=30)
    ax.set_xlim(0, 3_600)
    ax.set_xlabel("training steps until test accuracy ≥ 99% (fewer is better)")
    ax.set_title("Steps: every method still helps on the tuned baseline")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)

    ax = axes[1]
    for row, ((label, ratio), color) in enumerate(zip(TUNED_ARMS.values(), COLOURS)):
        ax.barh(-row, ratio, height=0.5, color=color)
        note = "" if row == 0 else ("  faster" if ratio < 1 else "  slower")
        ax.text(
            max(ratio, 1) + 0.03, -row, f"×{ratio:.2f}{note}", va="center", fontsize=9
        )
    ax.axvline(1, color=INK, lw=1)
    ax.set_xlim(0, 2.4)
    ax.set_xlabel("wall-clock seconds to 99% (× normal training)")
    ax.set_title("Seconds (3 seeds, run alone): only nowcast is faster")
    for ax in axes:
        style_axes(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Tuned baseline (lr 4e-3, weight decay 2), 10 seeds per method",
        x=0.125,
        y=1.0,
        ha="left",
        fontsize=12,
    )
    save(fig, "tuned_results.png")


def plot_nowcast_idea() -> None:
    """Three fast-moving weights of MLP seed 0: a line through recent snapshots, extended ahead."""
    ckpts = sorted(find_run(RUN).glob("step_*.pt"))
    steps = np.array([int(c.stem.removeprefix("step_")) for c in ckpts])
    keep = (steps >= 3_000) & (steps <= 12_000)
    steps = steps[keep]
    weights = np.stack(
        [
            load_model(c)[0].state_dict()["embed_a.weight"].double().numpy().ravel()
            for c, k in zip(ckpts, keep)
            if k
        ]
    )
    now, horizon, k = 9_000, 1_000, 4  # line through 4 snapshots, 1,000 steps ahead
    window = (steps > now - k * 500) & (steps <= now)
    t = steps[window]
    slope, icept = np.polyfit(t, weights[window], 1)
    order = np.argsort(np.abs(slope))
    # typical movers, not the extremes: the weights at the 90th, 95th and 98th percentile of |slope|
    picks = [order[int(q * (len(order) - 1))] for q in (0.98, 0.95, 0.90)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, color in zip(picks, COLOURS):
        ax.plot(steps, weights[:, i], color=color, lw=1, alpha=0.45)
        ax.plot(t, weights[window, i], "o", color=color, ms=6)
        line_t = np.array([t[0], now + horizon])
        ax.plot(line_t, slope[i] * line_t + icept[i], color=color, ls="--", lw=1.8)
        ax.plot(
            now + horizon,
            slope[i] * (now + horizon) + icept[i],
            "*",
            color=color,
            ms=15,
            mec="white",
        )
    ax.axvline(now, color=MUTED, ls=":", lw=1)
    xform = ax.get_xaxis_transform()  # x in data units, y in axes fraction
    ax.text(now - 80, 0.97, "now", color=MUTED, ha="right", va="top", transform=xform)
    ax.axvspan(now, now + horizon, color=GRID, alpha=0.5, lw=0)
    ax.text(now + horizon / 2, 0.03, "jump", color=MUTED, ha="center", transform=xform)
    handles = [
        plt.Line2D([], [], color=MUTED, lw=1, alpha=0.6, label="what training did"),
        plt.Line2D([], [], color=MUTED, marker="o", ls="", label="recent snapshots"),
        plt.Line2D([], [], color=MUTED, ls="--", label="straight-line forecast"),
        plt.Line2D([], [], color=MUTED, marker="*", ms=12, ls="", label="jump target"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    ax.set_xlabel("training step")
    ax.set_ylabel("value of the weight")
    ax.set_title(
        "Nowcasting: draw a line through each weight's recent past, jump ahead along it\n"
        "(3 weights of one plain run that are moving, but not the extremes; colour = one weight)"
    )
    style_axes(ax)
    save(fig, "nowcast_idea.png")


def main() -> None:
    """Make every figure, skipping (with a warning) those whose runs are missing."""
    for make in (
        plot_clock,
        plot_story_curve,
        plot_neuron_lens,
        plot_results_dots,
        plot_baseline_lesson,
        plot_tuned_results,
        plot_nowcast_idea,
    ):
        try:
            make()
        except FileNotFoundError as err:
            plt.close("all")
            print(f"warning: skipped {make.__name__}: {err}")


if __name__ == "__main__":
    main()
