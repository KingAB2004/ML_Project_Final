# Results - week1_v1

Standalone benchmarks: the baseline corpus and construction pipeline are unavailable and the original work is Chinese, so comparability rests on the instruments (item wording published in `prompts/`), not on shared code.

## Per-arm summary

| arm | distribution | success | ip | pri | aels | crs | rac | basic |
|---|---|---|---|---|---|---|---|---|
| base_instruct | in | 0.406 (n=30) | 0.042 (n=30) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.111 (n=30) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.935 (n=30) | 0.594 (n=30) | 0.856 (n=27) | 0.813 (n=30) |
| cellA_mono_ungated | in | 0.390 (n=30) | 0.065 (n=30) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.124 (n=30) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.938 (n=30) | 0.596 (n=30) | 0.849 (n=30) | 0.815 (n=30) |
| cellB_mono_gated | in | - | - | - | - | - | - | - |
| cellC_dec_ungated | in | - | - | - | - | - | - | - |
| cellD_dec_gated | in | - | - | - | - | - | - | - |
| reactive_baseline | in | 0.422 (n=30) | 0.045 (n=30) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.123 (n=30) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.929 (n=30) | 0.592 (n=30) | 0.853 (n=29) | 0.817 (n=30) |

## Conformal critic gate

- alpha=0.1, tau=0.5, lambda_hat=1.000, n_calibration=98, bound=hb
- calibration distribution: `own_generator_v1`
- The bound holds only for turns exchangeable with this calibration split. Arms marked `out` above report a MEASURED violation rate, not a guarantee.

### Risk-coverage frontier

| alpha | lambda_hat | release_rate | risk | risk_ucb | vacuous |
|---|---|---|---|---|---|
| 0.050 | 1.000 | 1.000 | 0.000 | 0.033 | False |
| 0.100 | 1.000 | 1.000 | 0.000 | 0.033 | False |
| 0.200 | 1.000 | 1.000 | 0.000 | 0.033 | False |

## Honesty notes

- IP and PRI measure resistance realized by a SIMULATED user; transfer to human reactance is out of scope.
- The four agent roles ran sequentially on one resident model; concurrency was not achieved on a 12 GB budget.
- See `reports/substitutions.md` for every reduced-scale substitution.

