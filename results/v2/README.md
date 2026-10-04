# v2 results: 50-profile corpus, QLoRA fine-tune, all baselines (Phases A, B, C)

This folder is a frozen snapshot of the **second full run** of the COCCON_NEW pipeline, copied on 3 October 2026
from the lab server (`cse@10.50.28.201`, RTX 3060 12 GB, Ollama). On the server the run directory is called
`runs/v3_50` (the third launch, sized to 50 corpus profiles); it is the "v2" that `results/v1/README.md` says
the numbers to report come from. It contains every fix in `BUGS.md` up to B50.

The point of this run is to prove the whole pipeline end to end at a small scale (dataset, fine-tuning,
testing, baselines) before the 1000-profile run. With 40 test profiles, most differences between arms are
inside their confidence intervals: read the numbers as a working pipeline, not as findings.

---

## 1. Setup

| Item | Value |
|---|---|
| Supporter, simulated seeker, agents | `qwen2.5:7b-instruct` (Ollama, q4) |
| Judge | `mistral-nemo:12b-instruct-2407-q4_K_M` (Ollama), 1 sample per item at temperature 0 |
| Fine-tuning base | `Qwen/Qwen2.5-7B-Instruct`, 4-bit nf4 (transformers + bitsandbytes) |
| Seeds | 400 situations from EmpatheticDialogues (`data/seeds/`) |
| Profiles | 400 hidden-need profiles, 0 rejected; resistance low 100 / medium 200 / high 100 |
| Splits (by profile, no overlap) | train 288, val 40, **test 40**, calibration 32 |
| Corpus profiles | **50** = first 45 train + first 5 val (`data/profiles/profiles_corpus50.jsonl`) |
| Gate calibration | 25 calibration profiles (never the test profiles) |
| Reduced settings | 2 counterfactual (PRI) rollouts, 4 parallel requests, 2 training epochs |
| Configs actually used | `configs/` (the server copies; they differ from the repo's defaults on purpose) |

---

## 2. Phase B: the dataset (training corpus)

| Stage | Count |
|---|---|
| Sessions generated (50 profiles, 2–4 linked sessions each) | 153, avg 20.2 turns |
| Rejected by the filter | 5 (3 non-English turns, 2 seeker recites the hidden need too early) |
| Later sessions of a rejected sequence (dropped, as designed) | 8 |
| Kept and annotated | **140 sessions from 48 profiles**, avg 20.3 turns |
| Turn annotations (Analysis + Strategy per supporter turn) | 1,280 |
| SFT examples, `with_thoughts` and `wo_thoughts` | **1,149 train / 131 val** each |

Files: `data/corpus/` (sessions, filtered, rejected, annotated, annotations) and `data/sft/`.

Known issues in this corpus (fixed in the code since, not in these files):
- 63 annotation lines contain the profile term "terminal need" (B43). The annotation prompt now forbids it;
  the 1000-profile dataset is built with the fixed prompt.
- The AELS quality judge rejected nothing; every rejection was structural (known limitation in `BUGS.md`).

---

## 3. Phase A: baselines and the 2 × 2 (untuned base model)

Scores are normalised 0–1. Higher is better except IP and PRI. 40 test profiles per arm; arms with memory run
2–4 linked sessions per profile (121 sessions), the rest one (40).

| Arm | Sessions | Success | AELS | CRS | RAC | Basic | IP ↓ | PRI observed ↓ | PRI counterfactual [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|
| `base_instruct` | 40 | 0.383 | 0.838 | 0.583 | 0.779 | 0.804 | 0.082 | 0.212 | −0.001 [−0.016, 0.015] |
| `reactive_baseline` | 40 | 0.388 | 0.872 | 0.605 | 0.796 | 0.808 | 0.085 | 0.212 | −0.002 [−0.016, 0.013] |
| `cellA_mono_ungated` | 121 | 0.381 | 0.855 | 0.592 | 0.781 | 0.801 | 0.078 | 0.215 | +0.003 [−0.006, 0.011] |
| `cellB_mono_gated` | 121 | 0.356 | 0.834 | 0.585 | 0.761 | 0.801 | **0.059** | 0.217 | −0.004 [−0.011, 0.004] |
| `cellC_dec_ungated` | 121 | 0.379 | 0.838 | 0.599 | 0.760 | 0.775 | 0.095 | 0.232 | +0.008 [−0.001, 0.017] |
| `cellD_dec_gated` | 121 | 0.362 | 0.829 | 0.601 | 0.752 | 0.757 | **0.062** | **0.200** | −0.005 [−0.013, 0.002] |

- **Leak check:** 0 leaking turns out of 6,449 supporter turns across all arms (including calibration).
- **2 × 2 on Success:** main effect of decomposition +0.002, of the gate −0.021, interaction +0.009.
  cellD − cellA = −0.019 [−0.039, 0.002]: no significant difference.
- **The gate works as designed:** it cuts IP by about 25–35 % (0.078 to 0.059 monolithic, 0.095 to 0.062
  decomposed) for about 0.02 of Success.
- **Gate paths:** cellB released 49 % / revised 39 % / fallback 13 %; cellD 45 % / 30 % / **25 %**. One in four
  cellD turns is the reflective fallback, which likely explains its lower Success and Basic.
- **Counterfactual PRI** is about 0 for every arm: the supporter's moves provoke no more resistance than a
  neutral reflection. No arm differs significantly.

### Conformal gate calibration

| alpha | tau | lambda_hat | n (turns) | release rate | risk (upper bound) |
|---|---|---|---|---|---|
| 0.10 | **0.20** | **0.125** | 300 | 0.543 | 0.057 (0.086) |

tau was lowered from 0.5 to 0.2 because at 0.5 no calibration turn counted as a violation, so the gate released
everything (`conformal/calibration_tau050.json` keeps that fit).

---

## 4. Phase C: QLoRA fine-tune and test

| Item | Value |
|---|---|
| Arm | `with_thoughts` (target = `<analysis><strategy><response>`) |
| LoRA | r 32, alpha 64, dropout 0.05, all attention + MLP projections (80.7 M trainable, 1.05 %) |
| Batch / lr / epochs | micro 1 × accumulation 16, lr 2e-4 cosine, 2 epochs (144 steps, 64 min) |
| Training loss | 1.11 to 0.50 |
| Validation loss | **0.761 after epoch 1**, 0.770 after epoch 2 (slight overfit) |
| Adapter kept | the epoch-1 checkpoint (best validation loss), `runs/v3_50/adapter_with_thoughts/` (308 MB) |

The first attempt crashed at the end of epoch 1 (out of memory in the validation pass, B50); its log is
`runs/v3_50/phaseC50_oom_b50.log`. The rerun after the fix is `phaseC50.log`.

### Fine-tuned supporter vs the base model (40 test profiles, one session each)

| Arm | Success | AELS | CRS | RAC | Basic | IP ↓ | PRI observed ↓ | PRI counterfactual [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| `base_instruct` | 0.383 | 0.838 | 0.583 | 0.779 | 0.804 | 0.082 | 0.212 | −0.001 [−0.016, 0.015] |
| `sft_with_thoughts` | **0.392** | 0.840 | 0.597 | 0.774 | 0.805 | 0.103 | 0.272 | +0.009 [−0.009, 0.027] |

- 0 leaking turns out of 394: no `<analysis>` or `<strategy>` text reached the seeker.
- Success and the published instruments are on par with the base model.
- **The fine-tuned model is pushier:** IP 0.103 vs 0.082 and observed resistance 0.272 vs 0.212. It learned
  the corpus supporter's need-probing style. Significance at n = 40 is unknown; watch this in the 1000-profile
  run. If it persists, the corpus supporter needs a softer style.
- `sft_wo_thoughts` was not trained in this run.

---

## 5. Files

| Path | What it is |
|---|---|
| `runs/v3_50/results_all.md` | Generated report with every arm, including `sft_with_thoughts` (regenerated locally) |
| `runs/v3_50/results_v3.md` | The report as written on the server after Phase A (no fine-tuned arm yet) |
| `runs/v3_50/dialogues/` | Every evaluation dialogue, one file per arm (plus `calib_dec_ungated`) |
| `runs/v3_50/scores/<arm>/` | Judge outputs per instrument, PRI rollouts and `pri_summary.json` |
| `runs/v3_50/conformal/` | Gate calibration, calibration rows, alpha sweep (and the tau 0.5 versions) |
| `runs/v3_50/memory/` | Need-state memory per profile and arm |
| `runs/v3_50/arm_summary_*.json`, `train_summary.json` | Per-arm run metadata, training config |
| `runs/v3_50/calls.jsonl` | Every model call (role, model, prompt hash, sizes) |
| `runs/v3_50/*.log` | `phaseA50`, `phaseB50`, `phaseC50`, the crashed `phaseC50_oom_b50`, and `run_v3` |
| `runs/v3_50/adapter_with_thoughts/` | The fine-tuned LoRA adapter (checkpoints not copied) |
| `data/`, `configs/`, `reports/` | Seeds, profiles, corpus, SFT files; the configs used; seed statistics |

The phase scripts that produced this run are `scripts/phaseA50.sh`, `phaseB50.sh`, `phaseC50.sh` and
`run_v3.sh` in the repository.
