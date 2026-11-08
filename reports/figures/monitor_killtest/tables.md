## Grok steps (first eval step with test acc >= 99%)

| Arm | Split | Grokked | Median | Min–max |
|---|---|---|---|---|
| G1 | calib | 10/10 | 18,380 | 17,560–19,300 |
| G1 | test | 10/10 | 18,030 | 17,060–18,560 |
| G2 | calib | 10/10 | 2,920 | 2,440–3,140 |
| G2 | test | 10/10 | 2,880 | 2,580–3,200 |
| G3 | calib | 10/10 | 2,760 | 2,420–3,120 |
| G3 | test | 10/10 | 2,880 | 2,580–3,040 |
| G4 | calib | 0/10 | — | — |
| G4 | test | 0/10 | — | — |
| C1 | test | 0/10 | — | — |
| C2 | test | 0/10 | — | — |
| C3 | test | 0/10 | — | — |

## Calibration (seeds 0–9; controls use G2's)

| Arm | c (B1) | clock (B2) | B3 norm threshold | r M1 (n) | r M2 (n) | r B3 (n) | r B4 (n) | r B5 (n) |
|---|---|---|---|---|---|---|---|---|
| G1 | 18.380 | 18,380 | 123.86 | 3.00 (10) | 1.38 (10) | 1.00 (10) | 2.49 (10) | — (0) |
| G2 | 23.360 | 2,920 | 119.41 | 3.97 (10) | 2.41 (10) | 1.00 (10) | 3.49 (10) | — (0) |
| G3 | 22.080 | 2,760 | 118.45 | n/a | 2.26 (10) | 0.99 (10) | n/a | n/a |
| G4 | — | — | — | n/a | — (0) | — (0) | n/a | n/a |

## Criterion 1: false alarms (test seeds; fires at any probe step)

| Control | Grokked (excluded) | M1 | M2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| C1 | 0 | 0/10 | 0/10 | 0/10 | 0/10 | 1/10 |
| C2 | 0 | 0/10 | 0/10 | 10/10 | 0/10 | 0/10 |
| C3 | 0 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

## Criterion 2: lead, t_fire <= 0.8 x grok (test seeds)

| Arm | M1 | M2 | B3 | B4 | B5 |
|---|---|---|---|---|---|
| G1 | 10/10 (fired 10/10; t_fire/grok 0.34) | 10/10 (fired 10/10; t_fire/grok 0.73) | 0/10 (fired 10/10; t_fire/grok 1.02) | 10/10 (fired 10/10; t_fire/grok 0.41) | 0/10 (fired 0/10; t_fire/grok —) |
| G2 | 10/10 (fired 10/10; t_fire/grok 0.25) | 10/10 (fired 10/10; t_fire/grok 0.42) | 0/10 (fired 10/10; t_fire/grok 1.01) | 10/10 (fired 10/10; t_fire/grok 0.29) | 0/10 (fired 0/10; t_fire/grok —) |
| G3 | n/a | 10/10 (fired 10/10; t_fire/grok 0.43) | 0/10 (fired 10/10; t_fire/grok 1.01) | n/a | n/a |
| G4 | n/a | 0/10 (fired 10/10; t_fire/grok —) | 0/10 (fired 0/10; t_fire/grok —) | n/a | n/a |

## Criterion 3: forecast log-MAE of grok_hat (test seeds that grok)

| Arm | n | M1 | M2 | B1 | B2 | B3 | B4 | B5 | best baseline |
|---|---|---|---|---|---|---|---|---|---|
| G1 | 10 | 0.062 | 0.023 | 0.028 | 0.028 | 0.016 | 0.055 | 0.802 | B3 0.016 |
| G2 | 10 | 0.057 | 0.032 | 0.043 | 0.043 | 0.029 | 0.055 | 1.938 | B3 0.029 |
| G3 | 10 | n/a | 0.059 | 0.056 | 0.056 | 0.044 | n/a | n/a | B3 0.044 |
| G4 | 0 | not evaluable: no test seed groks |

## Criterion 4 (M2 only): Spearman rho with test accuracy

| Arm | median rho M2 | median rho -norm | difference |
|---|---|---|---|
| G3 | 0.997 | 0.858 | +0.138 |
| G4 | 0.833 | 0.533 | +0.300 |

## Verdict per monitor

| Monitor | Check | Result |
|---|---|---|
| M1 | false alarms C1 | pass |
| M1 | false alarms C2 | pass |
| M1 | false alarms C3 | pass |
| M1 | lead G2 | pass |
| M1 | forecast G1 | FAIL |
| M1 | forecast G2 | FAIL |
| **M1** | **all criteria** | **FAIL** |
| M1 | sensitivity, not pre-registered: G4 checks dropped | FAIL |
| M2 | false alarms C1 | pass |
| M2 | false alarms C2 | pass |
| M2 | false alarms C3 | pass |
| M2 | lead G2 | pass |
| M2 | lead G3 | pass |
| M2 | lead G4 | FAIL |
| M2 | forecast G1 | FAIL |
| M2 | forecast G2 | FAIL |
| M2 | forecast G3 | FAIL |
| M2 | spearman G3 | pass |
| M2 | spearman G4 | pass |
| **M2** | **all criteria** | **FAIL** |
| M2 | sensitivity, not pre-registered: G4 checks dropped | FAIL |
