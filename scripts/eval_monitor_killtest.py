"""Evaluate the pre-registered monitor kill test (reports/preregistration_monitor.md).

Reads runs/killtest/<arm>/seed<s>/{metrics.json,monitor.jsonl}, written by train.py with
--monitor-every 50. Calibrates c (B1), the clock (B2), the B3 norm threshold and the ratio r of
every firing method on seeds 0-9 of each grokking arm, forecasts the grok step of seeds 10-19,
and applies the frozen pass criteria to M1 and M2. Controls use the calibration of G2, the arm
they are derived from. Test accuracy is read only here, for scoring, never by a monitor.

Run: uv run python scripts/eval_monitor_killtest.py
Writes reports/figures/monitor_killtest/{tables.md,scores.png,forecast_error.png}.
"""

import json
import math
from pathlib import Path
from statistics import median
from typing import Literal, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from quickgrok.monitor import fire_step, log_mae
from quickgrok.plot_compare import ARM_COLORS
from quickgrok.plot_pilot import INK, MUTED, find_grok_step, load_history, style_axes

RUNS = Path("runs/killtest")
OUT = Path("reports/figures/monitor_killtest")
CALIB, TEST = list(range(10)), list(range(10, 20))

Method = Literal["M1", "M2", "B1", "B2", "B3", "B4", "B5"]
FIRING: list[Method] = ["M1", "M2", "B3", "B4", "B5"]
MONITORS: list[Method] = ["M1", "M2"]
BASELINES: list[Method] = ["B1", "B2", "B3", "B4", "B5"]
SCORE_KEYS: dict[Method, str] = {
    "M1": "m1",
    "M2": "m2",
    "B3": "norm",
    "B4": "b4",
    "B5": "b5",
}
THRESHOLDS: dict[Method, float] = {"M1": 0.10, "M2": 0.15, "B4": 0.10}
B5_RATIO = 2.0  # fires at >= 2x its step-0 value
LEAD_FRAC, LEAD_MIN = 0.8, 8  # t_fire <= 0.8 x grok on >= 8 of 10 test seeds
MAX_ALARMS = 1
SPEARMAN_MARGIN = 0.1
ADD_ONLY: set[Method] = {"M1", "B4", "B5"}


class ArmSpec(TypedDict):
    lr: float
    wd: float
    budget: int
    add: bool
    calib: str  # arm whose calibration seeds fit c, r and the B3 threshold


ARMS: dict[str, ArmSpec] = {
    "G1": {"lr": 1e-3, "wd": 1.0, "budget": 40_000, "add": True, "calib": "G1"},
    "G2": {"lr": 4e-3, "wd": 2.0, "budget": 20_000, "add": True, "calib": "G2"},
    "G3": {"lr": 4e-3, "wd": 2.0, "budget": 20_000, "add": False, "calib": "G3"},
    "G4": {"lr": 4e-3, "wd": 2.0, "budget": 20_000, "add": False, "calib": "G4"},
    "C1": {"lr": 4e-3, "wd": 0.0, "budget": 20_000, "add": True, "calib": "G2"},
    "C2": {"lr": 4e-3, "wd": 2.0, "budget": 20_000, "add": True, "calib": "G2"},
    "C3": {"lr": 4e-3, "wd": 2.0, "budget": 20_000, "add": True, "calib": "G2"},
}
GROK_ARMS = ["G1", "G2", "G3", "G4"]
TUNED_ARMS = ["G2", "G3", "G4"]
CONTROL_ARMS = ["C1", "C2", "C3"]


class Run(TypedDict):
    seed: int
    grok: int | None  # first eval step with test accuracy >= 99%
    norm_at_grok: float | None  # metrics.json weight norm at that step
    eval_steps: list[int]
    val_acc: list[float]
    steps: list[int]  # monitor probe steps
    scores: dict[Method, list[float | None]]


class Calibration(TypedDict):
    # None when no calibration seed of the arm groks (then nothing can be calibrated)
    c: float | None  # B1: median grok * lr * wd
    clock: float | None  # B2: median grok step
    b3: float | None  # B3 threshold: median norm at grok
    r: dict[Method, float | None]  # median grok / t_fire; None if never fired
    n_ratio: dict[Method, int]  # calibration seeds behind r


def applies(method: Method, arm: str) -> bool:
    """M1, B4 and B5 only apply to the a + b arms (grokking and controls)."""
    return method not in ADD_ONLY or ARMS[arm]["add"]


def load_run(arm: str, seed: int) -> Run:
    """Read one run's metrics and monitor log."""
    run_dir = RUNS / arm / f"seed{seed}"
    history = load_history(run_dir)
    records = [
        json.loads(x) for x in (run_dir / "monitor.jsonl").read_text().splitlines()
    ]
    grok = find_grok_step(history)
    return {
        "seed": seed,
        "grok": grok,
        "norm_at_grok": next(
            (m["weight_norm"] for m in history if m["step"] == grok), None
        ),
        "eval_steps": [m["step"] for m in history],
        "val_acc": [m["val_acc"] for m in history],
        "steps": [r["step"] for r in records],
        "scores": {m: [r[k] for r in records] for m, k in SCORE_KEYS.items()},
    }


def fire(method: Method, run: Run, b3_threshold: float | None) -> int | None:
    """Firing step of a monitor or firing baseline on one run, by its frozen rule.

    B3 never fires without a calibrated threshold.
    """
    values = run["scores"][method]
    if method == "B3":
        if b3_threshold is None:
            return None
        return fire_step(run["steps"], values, b3_threshold, below=True)
    if method == "B5":
        base = values[run["steps"].index(0)]
        assert base is not None
        return fire_step(run["steps"], values, B5_RATIO * base)
    return fire_step(run["steps"], values, THRESHOLDS[method])


def calibrate(arm: str, runs: list[Run]) -> Calibration:
    """Fit c, the clock, the B3 threshold and r on the calibration runs that grok."""
    spec = ARMS[arm]
    grokked = [r for r in runs if r["grok"] is not None]
    groks = [float(r["grok"]) for r in grokked if r["grok"] is not None]
    if not grokked:
        return {
            "c": None,
            "clock": None,
            "b3": None,
            "r": {m: None for m in FIRING},
            "n_ratio": {m: 0 for m in FIRING},
        }
    b3 = median(r["norm_at_grok"] for r in grokked if r["norm_at_grok"] is not None)
    ratios: dict[Method, list[float]] = {m: [] for m in FIRING}
    for run in grokked:
        assert run["grok"] is not None
        for m in FIRING:
            if applies(m, arm) and (t := fire(m, run, b3)):  # t = 0 has no ratio
                ratios[m].append(run["grok"] / t)
    return {
        "c": median(g * spec["lr"] * spec["wd"] for g in groks),
        "clock": median(groks),
        "b3": b3,
        "r": {m: median(v) if v else None for m, v in ratios.items()},
        "n_ratio": {m: len(v) for m, v in ratios.items()},
    }


def forecast(method: Method, run: Run, arm: str, cal: Calibration) -> float:
    """grok_hat for one test run; a firing method that never fires gives the step budget.

    Anything left uncalibrated (no calibration seed grokked or fired) also gives the budget.
    """
    spec = ARMS[arm]
    budget = float(spec["budget"])
    if method == "B1":
        c = cal["c"]
        return budget if c is None else c / (spec["lr"] * spec["wd"])
    if method == "B2":
        return budget if cal["clock"] is None else cal["clock"]
    t, r = fire(method, run, cal["b3"]), cal["r"][method]
    return budget if t is None or r is None else t * r


def spearman_pair(run: Run) -> tuple[float, float]:
    """Spearman rho of M2 and of -norm against test accuracy, steps 0..grok + 1000.

    Test accuracy (logged every 20 steps) is linearly interpolated at the probe steps.
    """
    end = run["grok"] + 1000 if run["grok"] is not None else max(run["steps"])
    keep = [i for i, s in enumerate(run["steps"]) if s <= end]
    steps = [run["steps"][i] for i in keep]
    acc = np.interp(steps, run["eval_steps"], run["val_acc"])
    m2 = [run["scores"]["M2"][i] for i in keep]
    norm = [-(run["scores"]["B3"][i] or 0.0) for i in keep]
    return float(spearmanr(m2, acc).statistic), float(spearmanr(norm, acc).statistic)


def fmt(x: float | None, spec: str = ".3f") -> str:
    """Format a number, or a dash for None/nan."""
    return (
        "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{spec}}"
    )


def evaluate(
    runs: dict[str, dict[str, list[Run]]],
) -> tuple[str, dict[str, Calibration], dict[str, dict[Method, list[float]]]]:
    """Every criterion as markdown tables, the calibrations, and per-seed |ln error|."""
    cals = {a: calibrate(a, runs[a]["calib"]) for a in GROK_ARMS}
    lines: list[str] = []
    passed: dict[Method, dict[str, bool]] = {m: {} for m in MONITORS}

    lines += ["## Grok steps (first eval step with test acc >= 99%)", ""]
    lines += ["| Arm | Split | Grokked | Median | Min–max |", "|---|---|---|---|---|"]
    for arm in ARMS:
        for split, rs in runs[arm].items():
            g = [r["grok"] for r in rs if r["grok"] is not None]
            span = f"{min(g):,}–{max(g):,}" if g else "—"
            med = f"{median(g):,.0f}" if g else "—"
            lines.append(f"| {arm} | {split} | {len(g)}/{len(rs)} | {med} | {span} |")

    lines += ["", "## Calibration (seeds 0–9; controls use G2's)", ""]
    lines += [
        "| Arm | c (B1) | clock (B2) | B3 norm threshold | "
        + " | ".join(f"r {m} (n)" for m in FIRING)
        + " |",
        "|---|---|---|---|" + "---|" * len(FIRING),
    ]
    for arm, cal in cals.items():
        rs = [
            f"{fmt(cal['r'][m], '.2f')} ({cal['n_ratio'][m]})"
            if applies(m, arm)
            else "n/a"
            for m in FIRING
        ]
        lines.append(
            f"| {arm} | {fmt(cal['c'])} | {fmt(cal['clock'], ',.0f')} | "
            f"{fmt(cal['b3'], '.2f')} | " + " | ".join(rs) + " |"
        )

    lines += [
        "",
        "## Criterion 1: false alarms (test seeds; fires at any probe step)",
        "",
    ]
    lines += [
        "| Control | Grokked (excluded) | " + " | ".join(FIRING) + " |",
        "|---|---|" + "---|" * len(FIRING),
    ]
    for arm in CONTROL_ARMS:
        clean = [r for r in runs[arm]["test"] if r["grok"] is None]
        b3 = cals[ARMS[arm]["calib"]]["b3"]
        cells = []
        for m in FIRING:
            if not applies(m, arm):
                cells.append("n/a")
                continue
            n = sum(fire(m, r, b3) is not None for r in clean)
            cells.append(f"{n}/{len(clean)}")
            if m in MONITORS:
                passed[m][f"false alarms {arm}"] = n <= MAX_ALARMS
        lines.append(
            f"| {arm} | {len(runs[arm]['test']) - len(clean)} | "
            + " | ".join(cells)
            + " |"
        )

    lines += ["", "## Criterion 2: lead, t_fire <= 0.8 x grok (test seeds)", ""]
    lines += [
        "| Arm | " + " | ".join(FIRING) + " |",
        "|---|" + "---|" * len(FIRING),
    ]
    for arm in GROK_ARMS:
        cells = []
        for m in FIRING:
            if not applies(m, arm):
                cells.append("n/a")
                continue
            ok, fracs, n_fired = 0, [], 0
            for r in runs[arm]["test"]:
                t = fire(m, r, cals[arm]["b3"])
                n_fired += t is not None
                if r["grok"] is not None and t is not None:
                    fracs.append(t / r["grok"])
                    ok += t <= LEAD_FRAC * r["grok"]
            frac = fmt(median(fracs) if fracs else None, ".2f")
            cells.append(f"{ok}/10 (fired {n_fired}/10; t_fire/grok {frac})")
            if m in MONITORS and arm in TUNED_ARMS:
                passed[m][f"lead {arm}"] = ok >= LEAD_MIN
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Criterion 3: forecast log-MAE of grok_hat (test seeds that grok)",
        "",
    ]
    methods: list[Method] = ["M1", "M2", *BASELINES]
    lines += [
        "| Arm | n | " + " | ".join(methods) + " | best baseline |",
        "|---|---|" + "---|" * (len(methods) + 1),
    ]
    errors: dict[str, dict[Method, list[float]]] = {}
    for arm in GROK_ARMS:
        test = [r for r in runs[arm]["test"] if r["grok"] is not None]
        groks = [float(r["grok"]) for r in test if r["grok"] is not None]
        maes: dict[Method, float] = {}
        errors[arm] = {}
        if not test:
            lines.append(f"| {arm} | 0 | not evaluable: no test seed groks |")
            continue
        for m in methods:
            if applies(m, arm):
                hats = [forecast(m, r, arm, cals[arm]) for r in test]
                maes[m] = log_mae(hats, groks)
                errors[arm][m] = [abs(math.log(h / g)) for h, g in zip(hats, groks)]
        best = min((m for m in BASELINES if m in maes), key=lambda m: maes[m])
        for m in MONITORS:
            if m in maes:
                passed[m][f"forecast {arm}"] = maes[m] < maes[best]
        cells = [fmt(maes.get(m)) if m in maes else "n/a" for m in methods]
        lines.append(
            f"| {arm} | {len(test)} | "
            + " | ".join(cells)
            + f" | {best} {maes[best]:.3f} |"
        )

    lines += ["", "## Criterion 4 (M2 only): Spearman rho with test accuracy", ""]
    lines += [
        "| Arm | median rho M2 | median rho -norm | difference |",
        "|---|---|---|---|",
    ]
    for arm in ["G3", "G4"]:
        pairs = [spearman_pair(r) for r in runs[arm]["test"]]
        rho_m2 = float(np.nanmedian([p[0] for p in pairs]))
        rho_norm = float(np.nanmedian([p[1] for p in pairs]))
        passed["M2"][f"spearman {arm}"] = rho_m2 - rho_norm >= SPEARMAN_MARGIN
        lines.append(
            f"| {arm} | {rho_m2:.3f} | {rho_norm:.3f} | {rho_m2 - rho_norm:+.3f} |"
        )

    lines += ["", "## Verdict per monitor", ""]
    lines += ["| Monitor | Check | Result |", "|---|---|---|"]
    for m in MONITORS:
        for check, ok in passed[m].items():
            lines.append(f"| {m} | {check} | {'pass' if ok else 'FAIL'} |")
        verdict = "PASS" if all(passed[m].values()) else "FAIL"
        lines.append(f"| **{m}** | **all criteria** | **{verdict}** |")
        without = [ok for check, ok in passed[m].items() if not check.endswith("G4")]
        lines.append(
            f"| {m} | sensitivity, not pre-registered: G4 checks dropped | "
            f"{'pass' if all(without) else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n", cals, errors


def band(runs: list[Run], method: Method) -> tuple[np.ndarray, np.ndarray]:
    """Probe steps and the (25th, 50th, 75th) percentiles (3, n) over the runs at each step.

    Each step uses the runs still training then (they stop 1,000 steps after grokking).
    """
    steps = sorted({s for r in runs for s in r["steps"]})
    by_step: dict[int, list[float]] = {s: [] for s in steps}
    for r in runs:
        for s, v in zip(r["steps"], r["scores"][method]):
            if v is not None:
                by_step[s].append(v)
    q = np.array([np.percentile(by_step[s], [25, 50, 75]) for s in steps]).T
    return np.array(steps), q


def plot_scores(
    runs: dict[str, dict[str, list[Run]]], cals: dict[str, Calibration]
) -> None:
    """Rows: arms. Columns: the five firing scores, median and IQR over test seeds."""
    titles = {
        "M1": "M1 Fourier trigger",
        "M2": "M2 self-expression skill",
        "B3": "B3 parameter norm",
        "B4": "B4 weight-FFT fraction",
        "B5": "B5 excluded loss",
    }
    fig, axes = plt.subplots(len(ARMS), len(FIRING), figsize=(16, 2.1 * len(ARMS)))
    for row, arm in enumerate(ARMS):
        test = runs[arm]["test"]
        groks = [r["grok"] for r in test if r["grok"] is not None]
        for col, m in enumerate(FIRING):
            ax = axes[row, col]
            style_axes(ax)
            if not applies(m, arm):
                ax.set_axis_off()
                continue
            steps, q = band(test, m)
            color = ARM_COLORS[col]
            ax.fill_between(steps, q[0], q[2], color=color, alpha=0.25, lw=0)
            ax.plot(steps, q[1], color=color, lw=1.6)
            for g in groks:
                ax.axvline(g, color=MUTED, lw=0.6, ls="--", alpha=0.6)
            if groks:
                ax.axvline(median(groks), color=INK, lw=1.2, ls="--")
            if m == "B3":
                level = cals[ARMS[arm]["calib"]]["b3"]
            elif m == "B5":
                level = B5_RATIO * float(
                    np.median([r["scores"]["B5"][0] or 0 for r in test])
                )
            else:
                level = THRESHOLDS[m]
            if level is not None:  # G4 has no calibrated B3 threshold if it never groks
                ax.axhline(level, color=ARM_COLORS[7], lw=1, ls=":")
            if m in ("B3", "B5"):
                ax.set_yscale("log")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(titles[m], fontsize=9, color=INK, loc="left")
            if col == 0:
                ax.set_ylabel(arm, color=INK, fontsize=10, rotation=0, labelpad=14)
            if row == len(ARMS) - 1:
                ax.set_xlabel("training step", color=MUTED, fontsize=8)
    fig.suptitle(
        "Scores on test seeds 10–19: median (line) and IQR (band). Dashed: grok step "
        "(thin per seed, dark median). Dotted red: firing threshold",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "scores.png", dpi=130)
    plt.close(fig)


def plot_forecast_errors(errors: dict[str, dict[Method, list[float]]]) -> None:
    """Per grokking arm: |ln(grok_hat / grok)| per test seed (dots) and log-MAE (bar)."""
    fig, axes = plt.subplots(1, len(GROK_ARMS), figsize=(14, 3.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, arm in zip(axes, GROK_ARMS):
        style_axes(ax)
        names = list(errors[arm])
        if not names:
            ax.text(
                0.5,
                0.5,
                "not evaluable:\nno test seed groks",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=MUTED,
                fontsize=9,
            )
        for i, m in enumerate(names):
            e = errors[arm][m]
            color = ARM_COLORS[0] if m in MONITORS else MUTED
            ax.bar(i, float(np.mean(e)), width=0.6, color=color, alpha=0.35, lw=0)
            ax.scatter(
                i + rng.uniform(-0.15, 0.15, len(e)), e, s=12, color=color, zorder=3
            )
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_title(arm, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("|ln(grok_hat / grok)|", color=MUTED, fontsize=9)
    fig.suptitle(
        "Forecast error per method (bar: log-MAE; dots: test seeds; blue: monitors, "
        "grey: baselines)",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "forecast_error.png", dpi=130)
    plt.close(fig)


def main() -> None:
    """Load every run, print and write the tables, draw the figures."""
    runs = {
        arm: {
            "calib": [load_run(arm, s) for s in CALIB] if arm in GROK_ARMS else [],
            "test": [load_run(arm, s) for s in TEST],
        }
        for arm in ARMS
    }
    for arm in CONTROL_ARMS:
        del runs[arm]["calib"]
    tables, cals, errors = evaluate(runs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    plot_scores(runs, cals)
    plot_forecast_errors(errors)


if __name__ == "__main__":
    main()
