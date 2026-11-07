Steps to 99% test accuracy vs **vanilla** (median 18,400).

| Arm | Grokked | Median | Speed-up | 95% CI | Mann–Whitney p |
|---|---|---|---|---|---|
| project | 10/10 | 16,600 | +10% | [+6%, +14%] | 0.00021 |
| module | 10/10 | 10,800 | +41% | [+37%, +44%] | 0.00018 |
| module_random_k | 10/10 | 12,350 | +33% | [+29%, +38%] | 0.00018 |
| module_all_step0 | 10/10 | 700 | +96% | [+93%, +99%] | 0.00015 |
| module_all_step0_random_k | 10/10 | 1,450 | +92% | [+89%, +95%] | 0.00017 |
