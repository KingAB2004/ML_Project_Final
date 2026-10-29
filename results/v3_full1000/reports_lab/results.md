# Results - baselines_1000

Standalone benchmarks: the baseline corpus and construction pipeline are unavailable and the original work is Chinese, so comparability rests on the instruments (item wording published in `prompts/`), not on shared code.

## Per-arm summary

| arm | distribution | success | ip | pri | aels | crs | rac | basic |
|---|---|---|---|---|---|---|---|---|
| base_instruct | in | 0.383 (n=100) | 0.065 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.192 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.836 (n=100) | 0.587 (n=100) | 0.774 (n=100) | 0.801 (n=100) |
| calib_dec_ungated | in | - | - | - | - | - | - | - |
| cellA_mono_ungated | in | 0.382 (n=100) | 0.067 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.181 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.858 (n=100) | 0.589 (n=100) | 0.778 (n=100) | 0.801 (n=100) |
| cellB_mono_gated | in | 0.376 (n=100) | 0.061 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.186 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.833 (n=100) | 0.594 (n=100) | 0.762 (n=100) | 0.795 (n=100) |
| cellC_dec_ungated | in | 0.380 (n=100) | 0.089 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.204 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.850 (n=100) | 0.600 (n=100) | 0.780 (n=100) | 0.781 (n=100) |
| cellD_dec_gated | in | 0.369 (n=100) | 0.073 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.192 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.831 (n=100) | 0.602 (n=100) | 0.764 (n=100) | 0.779 (n=100) |
| gate_alpha_005 | in | 0.356 (n=100) | 0.063 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.188 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | - | - | - | - |
| gate_alpha_020 | in | 0.363 (n=100) | 0.088 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.196 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | - | - | - | - |
| reactive_baseline | in | 0.415 (n=100) | 0.089 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.191 (n=100) (judge-human agreement: NOT YET MEASURED - run human_eval/) | 0.868 (n=100) | 0.603 (n=100) | 0.812 (n=100) | 0.809 (n=100) |

## Conformal critic gate

- alpha=0.1, tau=0.2, lambda_hat=0.200, n_calibration=300, bound=hb
- calibration distribution: `own_generator_v1`
- The bound holds only for turns exchangeable with this calibration split. Arms marked `out` above report a MEASURED violation rate, not a guarantee.

### Risk-coverage frontier

| alpha | lambda_hat | release_rate | risk | risk_ucb | vacuous |
|---|---|---|---|---|---|
| 0.050 | 0.000 | 0.130 | 0.020 | 0.041 | False |
| 0.100 | 0.200 | 0.727 | 0.067 | 0.098 | False |
| 0.200 | 1.000 | 1.000 | 0.107 | 0.144 | False |

## 2x2: decomposition x gate (Success Rate)

- main effect of decomposition: -0.005
- main effect of the gate: -0.008
- interaction: -0.004

A gate-only effect is a legitimate result and is reported as such.

## Pre-registered paired comparisons (Success Rate)

| comparison | diff | ci | n |
|---|---|---|---|
| cellD_dec_gated - cellA_mono_ungated | -0.013 | [-0.025, -0.001] | 100 |

Resampling unit is the profile. A CI containing zero is reported as no effect.

## Honesty notes

- IP and PRI measure resistance realized by a SIMULATED user; transfer to human reactance is out of scope.
- The four agent roles ran sequentially on one resident model; concurrency was not achieved on a 12 GB budget.
- See `reports/substitutions.md` for every reduced-scale substitution.

