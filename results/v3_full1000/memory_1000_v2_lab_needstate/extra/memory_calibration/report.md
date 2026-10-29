# Need-state memory calibration - memory_1000_v2

Each final memory node's confidence is scored as a forecast that it is one of the profile's true chain needs (judge match >= 5/7). Brier = REL - RES + UNC: lower REL is better calibrated, higher RES better discrimination.

| set | nodes | real | Brier | REL | RES | ECE | AUROC | P(real|confirmed) | P(real|disconfirmed) |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 749 | 0.904 | 0.167 | 0.0813 | 0.0037 | 0.178 | 0.660 | 0.974 | 0.800 |
| mem_needstate | 749 | 0.904 | 0.167 | 0.0813 | 0.0037 | 0.178 | 0.660 | 0.974 | 0.800 |

Profile-grouped 5-fold recalibration from the evidence the memory stores: Brier 0.167 -> 0.087, log loss 0.613 -> 0.314.

P(real | disconfirmed) is the share of TRUE needs the memory blocked from re-proposal.
