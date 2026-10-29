# Need-state memory calibration - memory_v2

Each final memory node's confidence is scored as a forecast that it is one of the profile's true chain needs (judge match >= 5/7). Brier = REL - RES + UNC: lower REL is better calibrated, higher RES better discrimination.

| set | nodes | real | Brier | REL | RES | ECE | AUROC | P(real|confirmed) | P(real|disconfirmed) |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 785 | 0.903 | 0.162 | 0.0763 | 0.0025 | 0.167 | 0.600 | 0.914 | 0.667 |
| mem_needstate | 785 | 0.903 | 0.162 | 0.0763 | 0.0025 | 0.167 | 0.600 | 0.914 | 0.667 |

Profile-grouped 5-fold recalibration from the evidence the memory stores: Brier 0.162 -> 0.088, log loss 0.537 -> 0.318.

P(real | disconfirmed) is the share of TRUE needs the memory blocked from re-proposal.
