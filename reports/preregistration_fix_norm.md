# Pre-registration: does pinning the weight norm at its grok value advance grokking? (2026-11-08)

Written and committed before any non-smoke run. Experiment file: `experiments/fix_norm.py`; report to follow in `reports/fix_norm.md`.

## Question

Plain and nowcast MLP runs generalise at the same total weight norm (~124 untuned, `reports/nowcast.md`). Is the norm a **lever** (forcing it to the grok value makes grokking happen sooner) or only a **readout** (it falls alongside whatever actually sets the timescale)? The only intervention so far, the norm-only nowcast control, shrank the total L2 norm by ~8% at step ~1,200 (182 vs 198, far above 124) and was undone by step 3,000, so it did not test this.

## Literature checked first (`reports/literature_final.md`)

- Omnigrok (Liu et al. 2023, App. C, Fig. 9b): a 1-layer transformer on modular addition (p = 113), trained at a constant norm of 0.8× the initial norm from the start, "largely eliminates grokking". It reports this qualitatively, holds a small norm from initialisation, and does not pin the norm at the grok value after memorisation.
- Khanh et al. 2026 (arXiv 2606.13753): an embedding MLP on modular addition with the norm clamped at ρ × the free-training grok norm from a step after memorisation (not given); T_grok ∝ exp(7.5ρ). They report that ρ = 1 groks on "essentially the free-training timescale", but give no free-training T_grok at the same settings. That points to "readout at ρ = 1", but the setup differs (embeddings, p = 59, 90% threshold, β2 0.999, wd on matrices only) and the comparison is not quantified, so this experiment is still needed.

## Arms

Untuned MLP: `--model mlp`, lr 1e-3, wd 1, 30% train, a + b mod 97, other flags at their defaults. Seeds 0–9, cap 20,000 steps (`--steps 20000 --stop-after-grok 1000`, eval every 100 steps).

- **(a) plain:** no extra flags. Re-run fresh under the 20k cap (runs/fix_norm/plain). `runs/t6/vanilla` (same settings, 40k cap) is expected to be identical seed for seed; it is a reproducibility check, not the arm.
- **(b) pinned:** `--fix-norm 124 --fix-norm-from 1000`. From step 1,000 on, after every optimizer step, all parameters are rescaled jointly so that their total L2 norm is exactly 124 (the median norm at which plain runs first reach 99% test accuracy, `reports/nowcast.md`). At step 1,000 the plain norm is ~204, so the first rescale shrinks every weight by ~39%. AdamW moments are not reset.

## Metric

Steps until test accuracy first reaches ≥ 99% (logged every 100 steps), median over 10 seeds; two-sided Mann–Whitney U test, pinned vs plain. A run that has not reached 99% by step 20,000 counts as 20,100 (slower than any run that grokked) in the median and the test.

**Stays memorised:** a pinned run fails to stay memorised if its train accuracy is below 99% at any logged step from 2,000 to the end of the run (this allows 1,000 steps to recover from the first rescale). The arm fails to stay memorised if 3 or more of its 10 seeds do.

## Decision rule (verbatim from the brief)

- pinned median ≤ half the plain median AND Mann–Whitney p < 0.01 → "forcing the norm to the grok value advances grokking ≥2× (confounded with logit scale and Adam's effective step size)";
- pinned median within ±15% of plain, or no grok → "readout, not lever";
- anything in between, or pinned runs fail to stay memorised → "inconclusive".

"No grok" means no pinned seed reaches 99% by step 20,000. If the arm fails to stay memorised, the verdict is "inconclusive" whatever the medians say.

## Known confounds (stated in advance)

- **Logit scale.** The MLP is roughly 2-homogeneous in its weights (embedding → ReLU → linear readout; only the biases break this). Scaling all weights by c scales the logits by about c², so a 39% shrink cuts them by ~63%. A sooner grok could come from a lower-confidence, higher-loss regime rather than from the norm "being right".
- **Adam's effective step size.** Adam's update size does not shrink with the weights, so at a smaller norm each step is a larger *relative* change (effective learning rate ∝ lr / ‖w‖). Pinning at 124 from step 1,000 raises the relative step by ~1.6× over the plateau. A speed-up is therefore also consistent with "a larger effective learning rate", which tuning showed is a strong lever on its own (`reports/tuned_baseline.md`).
- **Pinning also removes the weight-decay drift.** Once the norm is fixed, weight decay can no longer shrink it, so the decay acts only through direction.

This experiment cannot separate these three; a positive result would say "the norm (with its scale effects) is a lever", not "the norm per se".

## Compute

20 runs × ≤ 20k steps of the MLP (~0.74 ms per step), well under 10 GPU-minutes.
