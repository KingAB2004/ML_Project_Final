# Results - v3_50

Standalone benchmarks: the baseline corpus and construction pipeline are unavailable and the original work is Chinese, so comparability rests on the instruments (item wording published in `prompts/`), not on shared code.

## Per-arm summary

| arm | distribution | success | ip | pri | aels | crs | rac | basic |
|---|---|---|---|---|---|---|---|---|
| base_instruct | in | 0.383 (n=40) | 0.082 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.212 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.838 (n=40) | 0.583 (n=40) | 0.779 (n=40) | 0.804 (n=40) |
| calib_dec_ungated | in | - | - | - | - | - | - | - |
| cellA_mono_ungated | in | 0.381 (n=40) | 0.078 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.215 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.855 (n=40) | 0.592 (n=40) | 0.781 (n=40) | 0.801 (n=40) |
| cellB_mono_gated | in | 0.356 (n=40) | 0.059 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.217 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.834 (n=40) | 0.585 (n=40) | 0.761 (n=40) | 0.801 (n=40) |
| cellC_dec_ungated | in | 0.379 (n=40) | 0.095 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.232 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.838 (n=40) | 0.599 (n=40) | 0.760 (n=40) | 0.775 (n=40) |
| cellD_dec_gated | in | 0.362 (n=40) | 0.062 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.200 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.829 (n=40) | 0.601 (n=40) | 0.752 (n=40) | 0.757 (n=40) |
| reactive_baseline | in | 0.388 (n=40) | 0.085 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.212 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.872 (n=40) | 0.605 (n=40) | 0.796 (n=40) | 0.808 (n=40) |
| sft_with_thoughts | in | 0.392 (n=40) | 0.103 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.272 (n=40) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.840 (n=40) | 0.597 (n=40) | 0.774 (n=40) | 0.805 (n=40) |

## Conformal critic gate

- alpha=0.1, tau=0.2, lambda_hat=0.125, n_calibration=300, bound=hb
- calibration distribution: `own_generator_v1`
- The bound holds only for turns exchangeable with this calibration split. Arms marked `out` above report a MEASURED violation rate, not a guarantee.

### Risk-coverage frontier

| alpha | lambda_hat | release_rate | risk | risk_ucb | vacuous |
|---|---|---|---|---|---|
| 0.050 | 0.000 | 0.107 | 0.013 | 0.032 | False |
| 0.100 | 0.125 | 0.543 | 0.057 | 0.086 | False |
| 0.200 | 1.000 | 1.000 | 0.127 | 0.166 | False |

## 2x2: decomposition x gate (Success Rate)

- main effect of decomposition: 0.002
- main effect of the gate: -0.021
- interaction: 0.009

A gate-only effect is a legitimate result and is reported as such.

## Pre-registered paired comparisons (Success Rate)

| comparison | diff | ci | n |
|---|---|---|---|
| cellD_dec_gated - cellA_mono_ungated | -0.019 | [-0.039, 0.002] | 40 |

Resampling unit is the profile. A CI containing zero is reported as no effect.

## Honesty notes

- IP and PRI measure resistance realized by a SIMULATED user; transfer to human reactance is out of scope.
- The four agent roles ran sequentially on one resident model; concurrency was not achieved on a 12 GB budget.
- See `reports/substitutions.md` for every reduced-scale substitution.

