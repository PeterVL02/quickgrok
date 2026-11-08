Steps to 99% test accuracy vs **vanilla** (median 18,400).

| Arm | Grokked | Median | Speed-up | 95% CI | Mann–Whitney p |
|---|---|---|---|---|---|
| shadow_linear | 10/10 | 18,400 | +0% | [-4%, +4%] | 1 |
| linear | 10/10 | 11,150 | +39% | [+36%, +42%] | 0.00018 |
| lstm | 10/10 | 11,350 | +38% | [+34%, +41%] | 0.00017 |
| linear_norm_only | 10/10 | 18,400 | +0% | [-5%, +5%] | 0.94 |
