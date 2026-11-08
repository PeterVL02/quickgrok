# Pre-registration: does linear nowcasting help the 1-layer transformer on a tuned baseline? (2026-11-08)

Written and committed before any tuning or final run. Code: `experiments/nowcast_transformer.py`. Report: `reports/nowcast_transformer.md`.

**Question.** On the tuned MLP (lr 4e-3, wd 2), linear weight nowcasting saved 37% of steps and ×0.80 wall-clock, but only ×1.04 at lr 1.6e-2 (`reports/tuned_retest.md`). Does it still save steps *and* wall-clock time on the 1-layer transformer (`--model transformer`, a + b mod 97, 30% train) against a tuned learning rate?

**Fixed for every run:** `--model transformer --steps 20000 --eval-every 20 --save-every 500`, full-batch AdamW (betas 0.9/0.98). Steps to 99% test accuracy are read at the 20-step eval resolution. A run that never reaches 99% within 20,000 steps counts as "did not grok".

## 1. Baseline tuning (seeds 100–102, `--stop-after-grok 500`)
- Grid: lr ∈ {2e-3, 4e-3, 8e-3} at wd 1, and lr ∈ {4e-3, 8e-3} at wd 2 (5 settings × 3 seeds). lr 1e-3, wd 1 is not rerun: the pilot grokked at 19.6k–29.7k steps there, slower than anything the MLP scaling predicts for the grid.
- A setting **qualifies** if (a) 3/3 seeds reach 99% test accuracy, (b) no run diverges (non-finite loss), and (c) memorisation clearly precedes generalisation: median steps to 99% test ≥ 5 × median steps to 99% train.
- **Choose** the qualifying setting with the lowest median steps to 99% test; ties go to the lower lr, then the lower wd.
- If the choice is lr 8e-3 (the grid edge), add lr 1.6e-2 at the same wd on seeds 100–102 once, and apply the same rule.
- If nothing qualifies, fall back to lr 1e-3, wd 1 (the pilot setting).

## 2. Nowcast tuning (seeds 100–104, chosen baseline setting, `--stop-after-grok 500`)
- Let G = the chosen setting's median steps to 99% test on seeds 100–102, and S = max(1, round(0.06 · G / 8)) the snapshot interval. K = 8 snapshots, max loss ratio 1.5, linear forecaster.
- Three combinations, (jump every J, horizon H):
  - A: J = 8S, H = 8S (the tuned MLP's J = H ≈ 6% of the grok step, snapshots every J/8)
  - B: J = 16S, H = 8S (jump half as often)
  - C: J = 8S, H = 4S (shorter, more conservative horizon)
- **Choose** by lowest median steps to 99% test; ties go to the lower mean, then to A, B, C in that order.
- Plain training on seeds 103–104 at the chosen setting is also run, for reference only (it does not enter the choice).

## 3. Final (seeds 0–9, frozen choices, `--stop-after-grok 1000`)
- Arms: `plain` (chosen lr, wd) and `nowcast` (same plus the chosen J, H, S).
- Steps to 99% test: median and IQR per arm; saving = (plain median − nowcast median) / plain median; 95% percentile-bootstrap CI of the median difference; two-sided Mann–Whitney U test, n = 10 vs 10 (`quickgrok.plot_compare.compare`, `stats.md`). If a run does not grok it is reported, and for the test it is assigned 20,000 steps.
- **Wall-clock:** seeds 0–2 of both arms, run one at a time with no other training process on the GPU, `--stop-after-grok 0` (stops at the 99% step). Plus 3 start-up runs (`--steps 0`). Time per run = process wall-clock − median start-up. Ratio = median nowcast time / median plain time.

## 4. Kill rule
Report **"nowcasting helps on the transformer"** only if
- the median step saving is ≥ 10% with Mann–Whitney p < 0.05, **and**
- the wall-clock ratio is ≤ 0.90.

Otherwise the result is reported as a null. Either outcome is written up in `reports/nowcast_transformer.md`.

## 5. Budget
At most ~30 min of total GPU time for the whole check. If the plan would exceed it, the lr grid or the tuning is shrunk and the report says so. Tuning results are added below as an addendum only if they are done before the final choices are frozen; the final runs use the frozen choices.
