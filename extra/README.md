# extra/ - analyses beyond the SOP

Six analyses that run on what the main pipeline already wrote under `runs/<id>/` (dialogues, judge scores,
memory). They never change a main-pipeline file; everything they write goes to `runs/<id>/extra/<name>/`
(`summary.json`, `report.md`, and a plot when matplotlib is installed). They reuse the project's own pieces:
the `LLM` class and its cache and one-model-at-a-time rule, the judge's score parser and its separation guard,
`counterfactual.replay_user_reply`, the Generator prompt, `grounding.classify_rung` and `memory.similarity`.
Only numpy is needed beyond the main requirements.

| Script | Question | Method | Model calls |
|---|---|---|---|
| `label_seeker.py` | What state is the seeker in, turn by turn? | Judge rates each seeker reply on 4 items (disclosed, opening, guarded, pulling back) and a fixed rule gives a state | judge, one per seeker turn |
| `markov.py` | How likely and how fast does each arm reach disclosure? | Absorbing Markov chain: fundamental matrix N = (I - Q)^-1, P(disclosed by turn H), expected turn of disclosure, LR tests for equal chains and for the first-order assumption | none (needs `label_seeker`) |
| `survival.py` | Does deeper probing speed up or slow down disclosure? | Kaplan-Meier, log-rank, RMST; Cox model with a time-varying depth covariate, Breslow ties, cluster-robust SE by profile | none (needs `label_seeker`) |
| `need_sets.py` | Can the system give a set of possible needs that contains the true one 90 % of the time? | Temperature scaling (MLE), split conformal prediction (LAC and APS) per conversation position | analyzer, 4 per session |
| `eig_probe.py` | Which next move learns most about the need for the least reactance? | Expected information gain (entropy) minus lambda times PRI; Lagrangian dual for a reactance budget | base model (moves, simulated replies, beliefs), then judge |
| `psychometrics.py` | Do IP and PRI measure what they claim? | Cronbach alpha, parallel analysis, ML factor analysis by EM with varimax, graded response IRT by EM with Gauss-Hermite quadrature | none |
| `memory_calibration.py` | Are the need-state memory's confidences honest? | Brier score with Murphy decomposition, ECE, AUROC, status audit, cross-validated logistic recalibration | judge, one per memory node |
| `memory_eval.py` | Does success hold up across time gaps, and does a denied inference come back? | Success by gap bucket (profile bootstrap); denied-inference detection and re-proposal count, dialogue level and from disconfirmed memory nodes | none (CPU) |
| `scaling.py` | Does the supporter improve with size, and do the thoughts help small models more? | power law L = a N^-b on validation loss and reply NLL (profile bootstrap CI on b), paired thoughts gain per size and its slope in log N | none (reads training logs, `scripts/scaling_eval.sh` output, `pvi.py` output) |
| `pvi.py` | How much usable information do the thoughts carry about the reply? | pointwise V-information between the with- and wo-thoughts adapters, per validation turn | GPU forward passes, 2 per turn per size |
| `lora_geometry.py` | Where in the network does the thoughts supervision land? | SVD of every LoRA update from its r x r core: effective/stable rank, depth profile, subspace similarity; `--truncate`: rank-k validation loss (Eckart-Young) | CPU; GPU for `--truncate` |

Each script's docstring states its model, its assumptions and what its guarantee does and does not cover.

## Running

Ollama with the two project models must be up for the scripts that call a model (`backend: ollama` in
`configs/models.yaml`, as on the lab server). Order:

```
python extra/psychometrics.py      --run runs/v3_50                     # CPU, seconds
python extra/label_seeker.py       --run runs/v3_50                     # judge pass
python extra/markov.py             --run runs/v3_50                     # CPU
python extra/survival.py           --run runs/v3_50                     # CPU
python extra/need_sets.py          --run runs/v3_50 --profiles data/profiles/profiles.jsonl
python extra/eig_probe.py          --run runs/v3_50 --profiles data/profiles/profiles.jsonl   # after need_sets
python extra/memory_calibration.py --run runs/v3_50                     # judge pass
python extra/test_extra.py                                              # 24 checks, echo backend, no GPU
```

The scaling study (Qwen2.5 0.5B / 3B / 7B, sizes and paths in `_shared.MODEL_SIZES`) writes to
`runs/scaling/extra/{scaling,pvi,lora_geometry}/`. `scripts/local_analyses.sh` runs `lora_geometry.py
--truncate`, `pvi.py` and `scaling.py` for every size whose two adapters exist; `scripts/scaling_eval.sh`
(lab server) produces the test-profile scores `scaling.py` reads. `_lm.py` holds the shared GPU code
(4-bit base + adapter, teacher-forced log-probabilities laid out exactly as `src/train.py` trained).

For the local snapshot use `--run results/v2/runs/v3_50 --profiles results/v2/data/profiles/profiles.jsonl`
(the reports are then written inside `results/v2/runs/v3_50/extra/`). For the 1000-profile run use
`--run runs/full_1000`.

Approximate model calls at the v2 size (40 test profiles): `label_seeker` about 5,700 judge calls,
`need_sets` about 800 analyzer calls, `eig_probe` about 1,900 calls at the defaults (40 decisions x 5 moves
x 3 replies), `memory_calibration` about 640 judge calls. Every call is cached, so a rerun is free.

**Do not run these with `--backend echo` in a real checkout.** The cache key does not include the backend,
so stub replies would be replayed later by a real run. `test_extra.py` uses an isolated temporary cache;
for a wiring check of the command-line scripts, use a scratch copy of the repository.

## Notes

- Depth in `survival.py` is measured with `grounding.classify_rung` on the supporter text, the same rule
  for every arm. The logged `ladder_rung` is the rung the pipeline permitted, not the depth of the turn
  (the monolithic listener writes a fixed L2), so it cannot be compared across arms.
- `need_sets.py` calibrates on calibration profiles (`calib_dec_ungated`) and tests on test profiles in an
  arm with the same architecture and simulator (`cellC_dec_ungated`), which is what exchangeability needs.
- A continuous-time Markov model of how needs change between sessions was considered and dropped: the corpus
  chooses each between-session transition with fixed weights (resolved 0.35, intensified 0.45, displaced 0.20,
  `sessions.pick_transition`), independent of the time gap. A rate model fitted to it would only recover that
  sampler. `memory_calibration.py` takes its place: it checks the memory against ground truth instead.
- Every result here is measured on simulated seekers and read by an LLM judge; transfer to people is out of
  scope, and the judge-human agreement study (`human_eval/`) is still the missing piece for IP and PRI.
