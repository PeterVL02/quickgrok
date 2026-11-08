# Reports: findings and conclusions

Status (2026-11-08): **project concluded.** A pre-registered test failed, so per the decision rule fixed in advance the remaining work is a blog post ("What survives a tuned baseline in grokking research") and repo cleanup. All runs use a one-hidden-layer MLP or a 1-layer transformer on a + b mod 97 with 30% of the pairs used for training, unless stated otherwise. Speed-ups are medians of steps to 99% test accuracy over 10 seeds (Mann–Whitney p ≈ 2e-4 unless stated otherwise).

## Bottom line

1. **Grokking reproduces, and neurons become structured long before test accuracy moves.** A probe that uses training data only sees hidden neurons turn into single-frequency waves (square waves in the MLP) at 25–54% of the way to the grokking step.
2. **Swapping neurons for trainable wave modules speeds grokking up, but mostly through prior knowledge.** At the original settings: +41%. A wrong-frequency control still gives +33%, and waves from step 0 grok ~26× sooner. Task-blind replacement shapes (global SVD, k-means levels, self-expression) all stall generalisation in the mid-90s.
3. **The untuned baseline was the biggest effect.** Raising the learning rate (and, more weakly, the weight decay) shortens grokking up to ~18×, roughly like 1/(lr·wd). This is already in the literature (e.g. Pracher et al. 2026, arXiv 2609.26679). Doubling the learning rate alone beat both of our methods at the old settings.
4. **On a tuned baseline (lr 4e-3, wd 2), step savings remain but shrink at the fastest settings.** Nowcasting saves 37%, wave modules 45%, both together 51%. At lr 1.6e-2 the savings are +11% and +24%. Only nowcasting also saves wall-clock time (×0.80); wave modules cost ×1.5.
5. **The pre-registered kill test failed.** Train-only monitors, the Fourier trigger and task-blind self-expression, do not forecast the grokking step better than cheap baselines. What survives: both separate grokking runs from non-grokking controls early, with 0/30 false alarms.

## Reports in order

| Report | Question | Conclusion |
|---|---|---|
| [pilot.md](pilot.md) | Does grokking reproduce, and what do neurons look like? | Yes: memorised by ~500 steps, 99% test at 19.6k–29.7k (transformer, 3 seeds). Fourier fits of neurons reach R² ≈ 0.97 after grokking; plain polynomials ≈ 0. No idle neurons to prune; PCA dimensionality drops together with accuracy, not before. |
| [mlp_baseline.md](mlp_baseline.md) | A cleaner test bed? | The MLP groks at 18.5k [IQR 18.2k–18.7k], a very tight seed spread. Its neurons are square waves in k·x mod p. |
| [accel_probe.md](accel_probe.md) | Does structure appear early enough to act on? | Go: ≥ 10% of neurons pass the trigger at 36–54% of the grok step, while test accuracy is still ~0%. |
| [substitution.md](substitution.md) | Does swapping triggered neurons for wave modules help? | Module +41%, project once +10%, wrong frequency +33%, all at step 0 +96%. Mostly an inductive-bias effect. |
| [nowcast.md](nowcast.md) | Does forecasting the weights and jumping help? | Linear extrapolation +39%, an LSTM no better, Grokfast 0% and a norm-only control 0% (old baseline). |
| [tuned_baseline.md](tuned_baseline.md) | How strong is a tuned baseline? | Grok time falls roughly like 1/(lr·wd), 17.9k → ~1k steps. The learning rate dominates; the product is only a rough guide. |
| [generic_basis.md](generic_basis.md) | Does the method work without knowing the inputs are cyclic (mul, S5)? | No. Global SVD never fires. k-means levels reflect an optimizer artefact present on every task. Self-expression stalls generalisation. |
| [tuned_retest.md](tuned_retest.md) | Do the gains survive the tuned baseline? | In steps, yes: nowcast +37%, module +45%, both +51%. In wall-clock only nowcasting wins. Gains shrink at the fastest settings. |
| [preregistration_monitor.md](preregistration_monitor.md) → [monitor_killtest.md](monitor_killtest.md) | Can a train-only monitor beat cheap baselines at predicting grokking? | Both monitors FAIL the forecast criterion. 0/30 false alarms on non-grokking controls. S5 never grokked at the tuned settings (deviation). Decision: blog post and stop. |

## Lessons worth keeping

- **Tune the baseline before claiming a speed-up.** The biggest effect in this project was an untuned learning rate.
- **Control experiments decide the interpretation.** The wrong-frequency control turned "jumping to the solution" into "a strong prior".
- **Constraining beats snapping:** neurons projected once drift back.
- **Steps are not seconds:** probing and parametrising can cost more time than the steps they save.
- **Pre-register kill tests, and check that every arm behaves as assumed before freezing it.** S5 at the tuned settings never grokked.
