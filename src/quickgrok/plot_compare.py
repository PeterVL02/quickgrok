"""Compare training runs grouped into arms (e.g. vanilla vs substitution), several seeds per arm.

Run: uv run python -m quickgrok.plot_compare --arm vanilla runs/mlp_baseline/seed* \
         --out reports/figures/mlp_baseline
Writes curves.png (loss and accuracy per arm), grok_steps.png (steps to 99% test accuracy per
seed) and summary.md (median and IQR table). If run directories contain probe.jsonl (written by
`train --probe-every`), also trigger.png and lead_time.md: when neurons pass the substitution
trigger (harmonic share >= tau_share and check-split R^2 >= tau_r2) relative to grokking.
If they contain substitutions.jsonl (`train --substitute`), also substitutions.png. With two or
more arms, stats.md tests every arm against the first (the baseline).
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from quickgrok.accel import ProbeRecord, SubstitutionRecord
from quickgrok.plot_pilot import (
    GRID,
    INK,
    MUTED,
    TRAIN,
    VAL,
    find_grok_step,
    load_history,
    style_axes,
)
from quickgrok.train import Metrics

# Reference categorical palette (dataviz skill), fixed order: arm i always gets slot i.
ARM_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]


TAU_SHARES = [0.5, 0.7, 0.85]
TAU_R2S = [0.8, 0.9]
LEAD_FRAC = 0.1  # fraction of neurons that must pass for the trigger to count as fired
N_BOOT = 10_000  # bootstrap resamples for the CI of the median difference


class ArmSummary(TypedDict):
    arm: str
    n_runs: int
    memorised: list[int | None]  # first step with train acc >= 99%, per run
    grok95: list[int | None]  # first step with test acc >= 95%, per run
    grok99: list[int | None]  # first step with test acc >= 99%, per run


def find_memorise_step(history: list[Metrics], threshold: float = 0.99) -> int | None:
    """First logged step with train accuracy >= threshold, or None."""
    return next((m["step"] for m in history if m["train_acc"] >= threshold), None)


def summarise_arm(arm: str, histories: list[list[Metrics]]) -> ArmSummary:
    """Per-run milestone steps for one arm."""
    return {
        "arm": arm,
        "n_runs": len(histories),
        "memorised": [find_memorise_step(h) for h in histories],
        "grok95": [find_grok_step(h, 0.95) for h in histories],
        "grok99": [find_grok_step(h, 0.99) for h in histories],
    }


def describe(steps: list[int | None]) -> str:
    """'median [q25–q75] (k/n)' over runs that reached the milestone."""
    done = [s for s in steps if s is not None]
    if not done:
        return f"— (0/{len(steps)})"
    q25, med, q75 = np.quantile(done, [0.25, 0.5, 0.75])
    return f"{med:,.0f} [{q25:,.0f}–{q75:,.0f}] ({len(done)}/{len(steps)})"


def write_summary(summaries: list[ArmSummary], out: Path) -> str:
    """Markdown table of milestone steps per arm; also written to `out`."""
    lines = [
        "| Arm | Runs | Train acc ≥ 99% | Test acc ≥ 95% | Test acc ≥ 99% |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['arm']} | {s['n_runs']} | {describe(s['memorised'])} | "
            f"{describe(s['grok95'])} | {describe(s['grok99'])} |"
        )
    lines.append("")
    lines.append(
        "Cells: median [IQR] step over runs that reached the milestone (reached/total)."
    )
    table = "\n".join(lines)
    out.write_text(table + "\n", encoding="utf-8")
    return table


def bootstrap_median_diff(
    base: list[int], arm: list[int], seed: int = 0
) -> tuple[float, float]:
    """95% percentile-bootstrap CI of median(base) - median(arm), resampling each group."""
    rng = np.random.default_rng(seed)
    b = rng.choice(base, (N_BOOT, len(base))).astype(float)
    a = rng.choice(arm, (N_BOOT, len(arm))).astype(float)
    diffs = np.median(b, axis=1) - np.median(a, axis=1)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high)


def write_stats(summaries: list[ArmSummary], out: Path) -> str:
    """Markdown table: each arm's steps to 99% test acc against the first arm; also to `out`.

    Speed-up = (baseline median - arm median) / baseline median, with a bootstrap 95% CI of the
    median difference on the same scale; p from a two-sided Mann-Whitney U test. Runs that never
    reached 99% are left out of both (the table counts them).
    """
    base = [x for x in summaries[0]["grok99"] if x is not None]
    lines = [
        f"Steps to 99% test accuracy vs **{summaries[0]['arm']}** (median {np.median(base):,.0f}).",
        "",
        "| Arm | Grokked | Median | Speed-up | 95% CI | Mann–Whitney p |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries[1:]:
        arm = [x for x in s["grok99"] if x is not None]
        if not arm:
            lines.append(f"| {s['arm']} | 0/{s['n_runs']} | — | — | — | — |")
            continue
        scale = 100 / np.median(base)
        low, high = bootstrap_median_diff(base, arm)
        speedup = (np.median(base) - np.median(arm)) * scale
        p = mannwhitneyu(base, arm, alternative="two-sided").pvalue
        lines.append(
            f"| {s['arm']} | {len(arm)}/{s['n_runs']} | {np.median(arm):,.0f} | "
            f"{speedup:+.0f}% | [{low * scale:+.0f}%, {high * scale:+.0f}%] | {p:.2g} |"
        )
    table = "\n".join(lines)
    out.write_text(table + "\n", encoding="utf-8")
    return table


def stack_metric(
    histories: list[list[Metrics]], key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Steps (T,) over all runs and values (runs, T); step 0 dropped for log x.

    A run that ended early keeps its last logged value (forward fill) up to the longest run.
    """
    steps = np.array(sorted({m["step"] for h in histories for m in h}))
    values = np.array(
        [
            np.array([m[key] for m in h])[  # type: ignore[literal-required]
                np.searchsorted([m["step"] for m in h], steps, side="right") - 1
            ]
            for h in histories
        ]
    )
    keep = steps > 0
    return steps[keep], values[:, keep]


def plot_curves(
    arms: list[str], histories: list[list[list[Metrics]]], out: Path
) -> None:
    """Rows: loss, accuracy. Columns: arms. Thin lines per seed, bold median of test."""
    fig, axes = plt.subplots(
        2, len(arms), figsize=(4.5 * len(arms), 6.5), sharex=True, sharey="row"
    )
    axes = axes.reshape(2, len(arms))
    for col, (arm, hists) in enumerate(zip(arms, histories)):
        for row, (train_key, test_key) in enumerate(
            [("train_loss", "val_loss"), ("train_acc", "val_acc")]
        ):
            ax = axes[row, col]
            for key, color, label in [
                (train_key, TRAIN, "train"),
                (test_key, VAL, "test"),
            ]:
                steps, vals = stack_metric(hists, key)
                for run in vals:
                    ax.plot(steps, run, color=color, lw=0.8, alpha=0.35)
                ax.plot(
                    steps,
                    np.median(vals, axis=0),
                    color=color,
                    lw=2,
                    label=f"{label} (median)",
                )
            ax.set_xscale("log")
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(
                    f"{arm} ({len(hists)} seeds)", fontsize=10, color=INK, loc="left"
                )
            style_axes(ax)
        axes[1, col].set_xlabel("training step (log)", color=MUTED, fontsize=9)
    axes[0, 0].set_ylabel("cross-entropy loss (log)", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("accuracy", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")  # empty corner
    fig.suptitle(
        "Train and test curves per arm (thin: individual seeds)",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grok_steps(summaries: list[ArmSummary], out: Path) -> None:
    """Dot plot: one row per arm, one dot per seed at its step to 99% test acc; bar = median."""
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.7 * len(summaries)))
    for i, s in enumerate(summaries):
        color = ARM_COLORS[i % len(ARM_COLORS)]
        done = [x for x in s["grok99"] if x is not None]
        y = len(summaries) - 1 - i
        ax.scatter(
            done,
            [y] * len(done),
            s=60,
            color=color,
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        if done:
            med = float(np.median(done))
            ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=2, zorder=4)
            ax.annotate(
                f"median {med:,.0f}",
                (med, y + 0.32),
                fontsize=8,
                color=INK,
                ha="center",
                va="bottom",
            )
        missing = s["n_runs"] - len(done)
        if missing:
            ax.annotate(
                f"{missing} run(s) never reached 99%",
                (1.0, y),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                color=MUTED,
                ha="right",
                va="center",
            )
    ax.set_yticks(range(len(summaries)), [s["arm"] for s in reversed(summaries)])
    ax.set_ylim(-0.6, len(summaries) - 0.2)
    ax.set_xlim(left=0)
    ax.set_xlabel("steps until test accuracy ≥ 99%", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID)
    ax.set_title(
        "Grokking step per seed (bar: median)", fontsize=11, color=INK, loc="left"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load_probe(run_dir: Path) -> list[ProbeRecord]:
    """Read the probe log written by `ProbeHook`."""
    lines = (run_dir / "probe.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def trigger_fraction(
    records: list[ProbeRecord], tau_share: float, tau_r2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Probe steps and the fraction of neurons with harmonic share and check R^2 above both taus."""
    steps = np.array([r["step"] for r in records])
    frac = np.array(
        [
            np.mean(
                [
                    s is not None and q is not None and s >= tau_share and q >= tau_r2
                    for s, q in zip(r["harmonic_share"], r["r2_check"])
                ]
            )
            for r in records
        ]
    )
    return steps, frac


def first_pass(
    records: list[ProbeRecord], tau_share: float, tau_r2: float
) -> int | None:
    """First probe step where at least LEAD_FRAC of neurons pass the trigger, or None."""
    steps, frac = trigger_fraction(records, tau_share, tau_r2)
    return next((int(s) for s, f in zip(steps, frac) if f >= LEAD_FRAC), None)


def plot_trigger(
    arms: list[str],
    probes: list[list[list[ProbeRecord]]],
    groks: list[list[int | None]],
    out: Path,
) -> None:
    """Rows: arms with probes. Columns: tau_r2. One line per tau_share (median, thin per run)."""
    fig, axes = plt.subplots(
        len(arms),
        len(TAU_R2S),
        figsize=(5 * len(TAU_R2S), 3.6 * len(arms)),
        sharex=True,
        squeeze=False,
    )
    for row, (arm, runs, arm_groks) in enumerate(zip(arms, probes, groks)):
        longest = [r["step"] for r in max(runs, key=len)]
        for col, tau_r2 in enumerate(TAU_R2S):
            ax = axes[row, col]
            for i, tau_share in enumerate(TAU_SHARES):
                color = ARM_COLORS[i]
                fracs = np.full(
                    (len(runs), len(longest)), np.nan
                )  # nan after a run ended
                for j, records in enumerate(runs):
                    steps, frac = trigger_fraction(records, tau_share, tau_r2)
                    fracs[j, : len(frac)] = frac
                    ax.plot(steps, frac, color=color, lw=0.8, alpha=0.35)
                ax.plot(
                    longest,
                    np.nanmedian(fracs, axis=0),
                    color=color,
                    lw=2,
                    ls=["-", "--", ":"][i],  # dashes keep coinciding lines visible
                    label=f"harmonic share ≥ {tau_share}",
                )
            done = [g for g in arm_groks if g is not None]
            if done:
                ax.axvline(float(np.median(done)), color=MUTED, ls="--", lw=1)
            ax.axhline(LEAD_FRAC, color=MUTED, lw=0.8, ls=":")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(
                f"{arm}: check R² ≥ {tau_r2}", fontsize=10, color=INK, loc="left"
            )
            style_axes(ax)
            if col == 0:
                ax.set_ylabel("fraction of neurons passing", color=MUTED, fontsize=9)
            if row == len(arms) - 1:
                ax.set_xlabel("training step", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Substitution trigger over training (thin: seeds, bold: median, dashed: median "
        f"grok step, dotted: {LEAD_FRAC:.0%})",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()  # also makes room for the suptitle
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_lead_times(
    arms: list[str],
    probes: list[list[list[ProbeRecord]]],
    groks: list[list[int | None]],
    names: list[list[str]],
    out: Path,
) -> str:
    """Markdown table per arm: first probe step with >= LEAD_FRAC passing, and step / grok step."""
    lines = []
    for arm, runs, arm_groks, arm_names in zip(arms, probes, groks, names):
        lines += [
            (
                f"**{arm}**: first probe step with ≥ {LEAD_FRAC:.0%} of neurons passing "
                "(in brackets: that step / the run's grok step)"
            ),
            "",
            "| τ_share | τ_r2 | "
            + " | ".join(arm_names)
            + " | median ratio | ratio < 0.8 |",
            "|---" * (len(arm_names) + 4) + "|",
        ]
        for tau_share in TAU_SHARES:
            for tau_r2 in TAU_R2S:
                cells, ratios = [], []
                for records, grok in zip(runs, arm_groks):
                    first = first_pass(records, tau_share, tau_r2)
                    ratio = first / grok if first is not None and grok else None
                    ratios.append(ratio)
                    cells.append(
                        "—"
                        if first is None
                        else f"{first:,}" + ("" if ratio is None else f" ({ratio:.2f})")
                    )
                done = [r for r in ratios if r is not None]
                med = f"{np.median(done):.2f}" if done else "—"
                early = sum(r < 0.8 for r in done)
                lines.append(
                    f"| {tau_share} | {tau_r2} | "
                    + " | ".join(cells)
                    + f" | {med} | {early}/{len(runs)} |"
                )
        lines.append("")
    table = "\n".join(lines)
    out.write_text(table, encoding="utf-8")
    return table


def count_substituted(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Probe steps of a run and the number of neurons substituted so far at each of them."""
    steps = np.array([r["step"] for r in load_probe(run_dir)])
    lines = (run_dir / "substitutions.jsonl").read_text().splitlines()
    events: list[SubstitutionRecord] = [json.loads(line) for line in lines]
    counts = np.zeros(len(steps))
    for e in events:
        counts[steps >= e["step"]] = e["n_substituted_total"]
    return steps, counts


def plot_substitutions(
    arms: list[str],
    colors: list[str],
    run_dirs: list[list[Path]],
    groks: list[list[int | None]],
    out: Path,
) -> None:
    """Substituted neurons over training per arm: thin per seed, bold median, dashed median grok."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (arm, color, dirs, arm_groks) in enumerate(
        zip(arms, colors, run_dirs, groks)
    ):
        runs = [count_substituted(d) for d in dirs]
        longest = max((steps for steps, _ in runs), key=len)
        counts = np.full((len(runs), len(longest)), np.nan)  # nan after a run ended
        for j, (steps, count) in enumerate(runs):
            counts[j, : len(count)] = count
            ax.plot(steps, count, color=color, lw=0.6, alpha=0.3)
        ax.plot(
            longest,
            np.nanmedian(counts, axis=0),
            color=color,
            lw=2,
            label=arm,
            ls=["-", "-."][i % 2],
        )  # -. keeps overlaps visible
        done = [g for g in arm_groks if g is not None]
        if done:
            ax.axvline(float(np.median(done)), color=color, ls="--", lw=1.2)
    ax.set_xlabel("training step", color=MUTED, fontsize=9)
    ax.set_ylabel("neurons substituted (of 512)", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title(
        "Substituted neurons over training (thin: seeds, bold: median, "
        "dashed: median step to 99% test acc)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def compare(arms: dict[str, list[Path]], out: Path) -> None:
    """Write the comparison figures and tables of `arms` (name -> run dirs) to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    names = list(arms)
    histories = [[load_history(d) for d in dirs] for dirs in arms.values()]
    summaries = [summarise_arm(n, h) for n, h in zip(names, histories)]
    plot_curves(names, histories, out / "curves.png")
    plot_grok_steps(summaries, out / "grok_steps.png")
    print(write_summary(summaries, out / "summary.md"))
    if len(summaries) > 1:
        print(write_stats(summaries, out / "stats.md"))

    subbed = [
        (arm, ARM_COLORS[i % len(ARM_COLORS)], dirs, s["grok99"])
        for i, ((arm, dirs), s) in enumerate(zip(arms.items(), summaries))
        if all((d / "substitutions.jsonl").exists() for d in dirs)
    ]
    if subbed:
        s_arms, s_colors, s_dirs, s_groks = map(list, zip(*subbed))
        plot_substitutions(s_arms, s_colors, s_dirs, s_groks, out / "substitutions.png")

    probed = [
        (arm, [load_probe(d) for d in dirs], s["grok99"], [d.name for d in dirs])
        for (arm, dirs), s in zip(arms.items(), summaries)
        if all((d / "probe.jsonl").exists() for d in dirs)
    ]
    if probed:
        p_arms, p_runs, p_groks, p_names = map(list, zip(*probed))
        plot_trigger(p_arms, p_runs, p_groks, out / "trigger.png")
        print(write_lead_times(p_arms, p_runs, p_groks, p_names, out / "lead_time.md"))


def main() -> None:
    """Parse arms from the command line, make the comparison figures and print the tables."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        nargs="+",
        action="append",
        required=True,
        metavar=("NAME", "RUN_DIR"),
        help="arm name followed by its run directories; repeat for more arms",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare({name: [Path(d) for d in dirs] for name, *dirs in args.arm}, args.out)


if __name__ == "__main__":
    main()
