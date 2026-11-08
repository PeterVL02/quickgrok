# Pre-registration: can a training-data-only monitor predict grokking better than cheap baselines?

_Committed 2026-11-07, before any of the runs below were made. Recommended by the project's council review (see PROGRESS.md, local). Any deviation from this document must be reported as a deviation in the results report._

## Question
Does a monitor computed during training predict whether and when a run will grok (reach ≥ 99% test accuracy), better than cheap baselines that need no probing? It may use only the model and the training inputs; test labels and test inputs are never touched.

## Monitors under test (frozen)
All are computed every **50 steps** on hidden layer 0 of `GrokMLP` (512 ReLU neurons). Each monitor produces a score per probe step and a **firing step** t_fire: the first probe step at which its rule holds.

- **M1, Fourier trigger** (task-aware; add arms and controls only). Per neuron: harmonic share (4 harmonics) and check R² on the 20% check split of the training rows. Score = fraction of neurons with share ≥ 0.5 and check R² ≥ 0.8. Fires when the score ≥ 0.10.
- **M2, self-expression** (task-blind; all arms and controls). Per neuron, capture of each input column (u_i, v_i, mean removed) by greedy OMP over q = 8 of the other neurons' columns of the same matrix, as in `scripts/basis_capture.py`. Skill = (capture − Gaussian null) / (1 − null). Score = median skill over neurons and columns. Fires when the score ≥ 0.15.

## Baselines (frozen)
Every forecast of the grok step is evaluated on the test seeds.

- **B1, lr·wd law.** grok_hat = c / (lr·wd), with c = median(grok·lr·wd) over the arm's calibration seeds. Known at step 0.
- **B2, clock.** grok_hat = median grok step of the arm's calibration seeds. Known at step 0.
- **B3, weight norm.** Fires at the first probe step at which the total parameter norm is ≤ the median norm-at-grok of the calibration seeds.
- **B4, data-free weight FFT.** Fraction of neurons whose input columns have weight-based Fourier captured energy ≥ 0.8 (the `fourier` basis in `accel.py`, no data used). Fires at ≥ 0.10. Add arms and controls only.
- **B5, excluded loss** (Nanda et al. 2023, online variant; add arms only). Train loss after removing the key-frequency components from the logits, with the key frequencies taken as the top 5 by DFT energy of the current embedding weights. Fires when it rises to ≥ 2× its value at step 0, or by the rule closest to Nanda's definition that can be implemented; any change is reported as a deviation.

For every monitor and firing baseline (M1, M2, B3, B4, B5): grok_hat = t_fire × r, with r = median(grok / t_fire) over the arm's calibration seeds. A monitor that never fires on a seed that grokks gives grok_hat = the step budget.

## Arms
Settings: one-hidden-layer MLP; full-batch AdamW, betas (0.9, 0.98); eval every 20 steps; stop 1,000 steps after reaching 99% test accuracy.

- **Calibration seeds:** 0–9. Used only to fit c, r and the B3 threshold.
- **Test seeds:** 10–19.

Grokking arms (should grok):

| Arm | Task | Train frac | lr | wd | Step budget |
|---|---|---|---|---|---|
| G1 | a + b mod 97 | 0.3 | 1e-3 | 1.0 | 40,000 |
| G2 | a + b mod 97 | 0.3 | 4e-3 | 2.0 | 20,000 |
| G3 | a · b mod 97 | 0.3 | 4e-3 | 2.0 | 20,000 |
| G4 | S5 composition | 0.5 | 4e-3 | 2.0 | 20,000 |

Control arms (should not grok within the budget):

| Arm | Change vs G2 | Step budget |
|---|---|---|
| C1 | weight decay 0 | 20,000 |
| C2 | train fraction 0.1 | 20,000 |
| C3 | random training labels (fixed permutation of the training targets, seeded) | 20,000 |

If a control run does grok, it is reported and excluded from the false-alarm count for that control.

## Pass criteria for a monitor (all must hold)
1. **False alarms:** it fires on ≤ 1 of 10 test seeds in each control arm.
2. **Lead:** in each tuned grokking arm where it applies (G2–G4), t_fire ≤ 0.8 × grok step on ≥ 8 of 10 test seeds.
3. **Forecast:** in each applicable grokking arm, its log-MAE of grok_hat on the test seeds is lower than that of every baseline that applies to that arm (B1, B2 and B3 everywhere; B4 and B5 on add arms).
4. **M2 only:** on G3 and G4, the median over test seeds of the Spearman correlation between M2's score and test accuracy across probe steps (from step 0 to the grok step + 1,000) exceeds the same correlation for −(weight norm) by ≥ 0.1.

## Decision
- **If at least one monitor passes:** a 2–3 week workshop-paper effort on that monitor against cheap baselines, nothing broader.
- **If neither passes:** write "What survives a tuned baseline in grokking research" as a blog post, keep the teaching material, and stop the project.
