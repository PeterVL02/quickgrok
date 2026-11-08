# Final literature check: pinned norms and weight forecasting (2026-11-08)

Two questions from the final review: (1) does Omnigrok already show that pinning the weight norm at the right value removes the plateau on modular addition? (2) correct citations for the weight-forecasting ("nowcasting") work. The papers were read through a web-fetch tool, not in full by hand; quotes are short excerpts.

## 1. Constrained-norm training on modular addition

**Omnigrok** — Ziming Liu, Eric J. Michaud, Max Tegmark, *Omnigrok: Grokking Beyond Algorithmic Data*, ICLR 2023, arXiv [2210.01117](https://arxiv.org/abs/2210.01117).

- Method (§2): "we perform the constrained minimization by rescaling the model weights back to their original norm" after each unconstrained step.
- **Modular addition (Appendix C, Figure 9b):** a 1-layer transformer on modular addition (p = 113). Shrinking the initialisation (α = 0.8 × the standard norm) *and* holding the norm constant over training: "Constrained optimization at constant weight norm (α = 0.8) largely eliminates grokking", with "test and train accuracy improving almost concurrently". The figure is titled "Constraining a small weight norm eliminates grokking". Main text (§5): "we also show the ability to eliminate grokking in Appendix C."
- Appendix B (Figure 8): the same transformer trained at constant norms α ∈ [0.1, 4] gives the L/U loss-vs-norm shapes. It reports no time-to-generalise.
- The MNIST constrained-norm results (§4, Figure 3) are loss-landscape computations (minimise over directions at a fixed norm), not training-time measurements.

**Does Omnigrok answer our question? No, not for our setup.** It holds a *small norm from initialisation* on a transformer and reports the result qualitatively (one figure, no step counts). It does not pin an already-memorised network at the *norm where free training groks*, and it does not compare steps to generalise against free training. Part 2 (`reports/fix_norm.md`) was therefore run.

**Closest prior work, found in passing:** Truong Xuan Khanh, Doan Hoang Viet, Luu Duc Trung, Phan Thanh Duc, *The Weight Norm Sets the Grokking Timescale: A Causal Delay Law*, arXiv [2606.13753](https://arxiv.org/abs/2606.13753) (June 2026, preprint).

- Setup: an embedding + 2-layer MLP on modular addition (p = 29–97), full-batch AdamW with wd on matrices only, grok threshold 90% test accuracy.
- Intervention: after every step from a point after memorisation (t_int, value not stated), all weights are projected globally to ‖W‖ = ρ‖W‖c, where ‖W‖c is the norm at which free training groks. AdamW moments are not reset.
- Result: delay T_grok ∝ exp(α·ρ) with α ≈ 7.5 across four moduli (R² = 0.996). Pinning above ‖W‖c slows grokking a lot (Table 1, p = 59, lr 1e-3: ρ = 0.85 → 1,492 steps, 1.00 → 4,077, 1.15 → 13,426, 1.30 → 27,925); the norm moves the delay ~19× against ~2× for the learning rate.
- At ρ = 1 the run "groks on essentially the free-training timescale (no delay)". They give no free-training T_grok number to quantify this.
- So pinning at the grok norm is reported **not** to remove the plateau, while pinning *off* that norm changes the delay strongly. This is consistent with our result.

## 2. Weight-forecasting papers (for the References)

| Paper | Authors, year, venue | Uses "nowcasting"? |
|---|---|---|
| *Introspection: Accelerating Neural Network Training By Learning Weight Evolution* | Abhishek Sinha, Mausoom Sarkar, Aahitagni Mukherjee, Balaji Krishnamurthy; ICLR 2017; arXiv [1704.04959](https://arxiv.org/abs/1704.04959), [OpenReview](https://openreview.net/forum?id=Hkg8bDqee) | No: "introspection network" / "weight evolution". (Author order as on arXiv; the ICLR anthology swaps Sarkar and Mukherjee.) |
| *Learning to Boost Training by Periodic Nowcasting Near Future Weights* (WNN, the Weight Nowcaster Network) | Jinhyeok Jang, Woo-Han Yun, Won Hwa Kim, Youngwoo Yoon, Jaehong Kim, Jaeyeon Lee, Byungok Han; ICML 2023, PMLR 202:14730–14757 ([proceedings](https://proceedings.mlr.press/v202/jang23b.html)) | Yes, it introduces the term. |
| *Accelerating Training with Neuron Interaction and Nowcasting Networks* (NiNo) | Boris Knyazev, Abhinav Moudgil, Guillaume Lajoie, Eugene Belilovsky, Simon Lacoste-Julien; ICLR 2025; arXiv [2409.04434](https://arxiv.org/abs/2409.04434) | Yes, it builds on WNN. |

All three learn the forecaster on other training runs (Introspection, WNN, NiNo) and apply it periodically. The nowcaster in this repo is fitted online on the run's own history (a per-weight line or an online LSTM).

**Also relevant (one line each):**

- Khanh et al. 2026, arXiv [2606.13753](https://arxiv.org/abs/2606.13753): the pinned-norm delay law above, the closest prior work to `reports/fix_norm.md`.
- Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee, *Grokfast: Accelerated Grokking by Amplifying Slow Gradients*, arXiv [2405.20233](https://arxiv.org/abs/2405.20233) (2024). It amplifies the slow component of the gradient history, a close cousin of extrapolating the slow weight drift.
