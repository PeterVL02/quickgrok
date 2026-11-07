Steps to 99% test accuracy vs **vanilla** (median 2,920).

| Arm | Grokked | Median | Speed-up | 95% CI | Mann–Whitney p |
|---|---|---|---|---|---|
| nowcast_linear | 10/10 | 1,840 | +37% | [+24%, +41%] | 0.00017 |
| module | 10/10 | 1,620 | +45% | [+33%, +48%] | 0.00017 |
| module_random_k | 10/10 | 2,030 | +30% | [+18%, +33%] | 0.00018 |
| nowcast_plus_module | 10/10 | 1,420 | +51% | [+40%, +55%] | 0.00016 |
