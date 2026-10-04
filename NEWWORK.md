# NEWWORK.md: work beyond the SOP, and deviations from it

The SOP (`modified_SOP_12340340_12340370.pdf`) defines four enhancements, QLoRA fine-tuning with the
with/without-thoughts ablation, and an evaluation protocol. Everything in this file goes **beyond** that scope,
or **changes** something the SOP or `PLAN.md` specified. Work that simply implements the SOP is described in
`README.md`, `PLAN.md` and `planstudy.md`, not here. Bug fixes are in `BUGS.md`.

Each entry says what was added, why, where the code is, and where its results are.

---

## 1. New analyses (not in the SOP)

### 1.1 Seeker-state dynamics: absorbing Markov chain and survival analysis
- **What:** the judge labels every seeker reply as guarded / opening / withdrawn / disclosed
  (`extra/label_seeker.py`). `extra/markov.py` fits an absorbing Markov chain per arm (fundamental matrix
  N = (I - Q)^-1, P(disclosure by turn H), expected turn of disclosure, likelihood-ratio tests for equal chains
  across arms and for the first-order assumption). `extra/survival.py` treats disclosure as a time-to-event:
  Kaplan-Meier, log-rank, restricted mean survival time, and a Cox model with probing depth as a time-varying
  covariate, Breslow ties and cluster-robust standard errors by profile.
- **Why:** the SOP measures resistance per turn (IP, PRI) but not *how a conversation moves* toward
  disclosure, or whether deeper probing speeds it up or slows it down.
- **Status:** code and tests done; not yet run on a full run (one judge pass, ~5.7k calls at v2 size).

### 1.2 Conformal prediction sets for the hidden need (`extra/need_sets.py`)
- **What:** at four points in a conversation, a set of candidate needs guaranteed to contain the true one
  with probability 1 - alpha (split conformal, LAC and APS scores, after temperature scaling by maximum
  likelihood), calibrated on calibration profiles and tested on test profiles.
- **Why:** the SOP uses conformal risk control only for the critic gate. This applies the same guarantee to
  the need inference itself: how uncertain is the system about the hidden need, turn by turn?
- **Status:** code and tests done; not yet run.

### 1.3 Information gain vs reactance for the next move (`extra/eig_probe.py`)
- **What:** for candidate moves at each ladder rung, the expected information gain about the need (entropy
  reduction over simulated replies) minus lambda x PRI, with the Lagrangian dual for a reactance budget.
- **Why:** frames "how deep to probe" as an explicit information-versus-reactance trade-off.
- **Status:** code and tests done; not yet run.

### 1.4 Psychometric validation of IP and PRI (`extra/psychometrics.py`)
- **What:** Cronbach's alpha, item-total correlations, parallel analysis, maximum-likelihood factor analysis
  (EM, varimax), and a graded response IRT model (Bock-Aitkin EM with Gauss-Hermite quadrature) on the judge's
  item ratings.
- **Why:** the SOP proposes two new instruments; this checks whether their items measure one coherent
  construct each, as a psychological scale should. CPU only.
- **Status:** code and tests done; can run on the v2 scores at any time.

### 1.5 Calibration of the need-state memory (`extra/memory_calibration.py`)
- **What:** each final memory belief's confidence scored as a forecast that it is a true need (judge match
  against the profile's need chain): Brier score with Murphy decomposition, ECE, AUROC, and cross-validated
  logistic recalibration.
- **Why:** the SOP gives memory nodes a confidence but never checks that the numbers are honest.
- **Status:** code and tests done; to be run on the memory arms.

### 1.6 Scaling study: 0.5B / 3B / 7B (`extra/scaling.py`, `scripts/scale_local.sh`, `scripts/scale_3b.sh`, `scripts/scaling_eval.sh`)
- **What:** the same QLoRA recipe and the same 1000-profile SFT files on Qwen2.5-0.5B, -3B and -7B-Instruct,
  both arms each. Fits L(N) = a N^-b (with a bootstrap CI on b) and measures the thoughts gain per size and
  its slope in log N. Test-profile evaluation of every size runs against **the same 7B seeker**: a smaller
  supporter is loaded beside the 7B base (`evaluate.py --supporter-base`, `llm.TransformersBackend`).
- **Why:** the SOP fine-tunes one model. The study asks whether the Analysis/Strategy supervision helps small
  models more than large ones, and how performance scales with size. 1.5B was dropped for time.
- **Status:** 0.5B done (laptop); 3B training (laptop, ~Mon 5 Oct 19:00); 7B training (lab, ~Tue 6 Oct
  08:00); test evaluation queued on the lab server. Results: `results/v3_full1000/`.

### 1.7 Pointwise V-information of the thoughts (`extra/pvi.py`)
- **What:** PVI = log2 p_with(reply | context, thoughts) - log2 p_wo(reply | context) per validation turn,
  using the two adapters (Ethayarajh et al. 2022); mean = usable information in the gold thoughts.
- **Why:** the ablation gives one aggregate number; PVI says per turn how much the thoughts inform the reply
  and finds annotations that hurt (PVI < 0).
- **Status:** 0.5B done (8.9 bits/turn); 3B and 7B run automatically when their adapters exist.
  Results: `results/v3_full1000/analyses/pvi/`.

### 1.8 Geometry of the LoRA updates (`extra/lora_geometry.py`)
- **What:** SVD of every update dW = s B A from its r x r core: effective and stable rank, energy in the top
  directions, distribution over depth, subspace similarity between the two arms (Hu et al. 2021), and
  validation loss after rank-k truncation (Eckart-Young).
- **Why:** shows where in the network the thoughts supervision lands and how much of rank 32 is used.
- **Status:** 0.5B done; others automatic. Results: `results/v3_full1000/analyses/lora_geometry/`.

### 1.9 Memory metrics computed without a model (`extra/memory_eval.py`)
- **What:** Success Rate by gap since the previous session; detection of denied inferences and their
  re-proposal (dialogue level, and from disconfirmed memory nodes). With `--judge`: transition recall (does the
  memory reflect what changed for the seeker) and abstention on deliberately ambiguous profiles.
- **Why:** the SOP promises "Success Rate as well as recall" and targets conflict detection and abstention;
  `PLAN.md` 12.4 lists these metrics, but no code computed them.
- **Status:** CPU parts run on v2 (`results/v2/runs/v3_50/extra/memory_eval/`): no measurable gap effect, and
  re-proposal is untestable there because the simulated seekers never deny an inference (see 2.1).

### 1.10 ES-MemEval as a question-answering benchmark (`extra/memeval_qa.py`)
- **What:** each memory type ingests a user's real past sessions (18 users, 401 dated sessions) and answers
  the benchmark's 1,427 questions (information extraction, user modelling, temporal reasoning, conflict
  detection, abstention) from its brief; the judge grades against the gold answers.
- **Why:** the SOP says "evaluate on ES-MemEval" without a protocol. The dataset is a memory QA benchmark, so
  this is the protocol that matches it. The earlier converter (`src/external.py`) did not match the dataset's
  real format and produced 18 empty profiles; the simulated-dialogue arm built on it was removed.
- **Status:** code and stub-model test done; to run on the lab server (~6-7 h).

---

## 2. Changes to how the system is evaluated

### 2.1 Corrective seeker for the memory arms (`prompts/user_corrective.md`, `simulator: corrective`)
- **What:** an evaluation-only addition to the seeker prompt: when the supporter states a reading of the
  seeker's needs that is wrong, the seeker says so plainly ("no, that's not really it") without revealing the
  real need, and does not deny a correct reading.
- **Why:** in v2 the seekers made 0 explicit denials of 101 supporter inferences (they resist by deflecting),
  so the need-state memory never recorded a disconfirmation and the SOP's central memory claim (a denied
  inference is never re-proposed) could not be tested. `PLAN.md` 12.4 anticipated this ("needs harder denial
  scripting"). The corpus and training data are unchanged; only the memory arms use it.

### 2.2 The seeker moves on between evaluation sessions (`advance_profile: true`)
- **What:** in the memory arms, the profile is advanced between sessions exactly as in corpus generation
  (`sessions.advance_profile`: resolved / intensified / displaced), and the change is told to the seeker.
- **Why:** before, every evaluation session reused the same unchanged profile, so memory faced a static target
  and there was no ground truth for transition recall. The SOP's Fig. 2 describes evolving users.

### 2.3 Memory arms redefined (`configs/arms.yaml`)
- **What:** all five memory arms now use the decomposed agents, the gate off, the base model, 2-4 linked
  sessions (including `mem_none`, which used to run a single session), the corrective seeker and profile
  advancement. Each session records `memory_brief_after`, what the memory would hand the next session.
- **Why:** the monolithic listener has no Analyzer, so it never writes a belief (cells A/B held 0 memory
  nodes in v2); the gate adds a calibration confound unrelated to memory; and the session-isolated baseline
  must still have follow-up sessions to compare against.

### 2.4 A second, smaller supporter beside the resident seeker (`evaluate.py --supporter-base`)
- See 1.6. Needed so every size of the scaling study faces the same 7B seeker.

---

## 3. Infrastructure added
- `scripts/full_run.sh`: one command for a fresh GPU machine (used for the RTX 5090 run).
- `scripts/collect_results.sh`, `scripts/auto_publish.sh`: copy results from `runs/` into the tracked
  `results/` folder and push them to GitHub as they arrive.
- `extra/dataset_stats.py`: corpus statistics. `src/train.py` now saves the full loss history and best
  checkpoint in `train_summary_<arm>.json`.
- `results/README.md` and one README per results folder explaining every number; `planstudy.md` as a study
  guide.

---

## 4. Deviations from the SOP / PLAN configuration (substitutions)

The SOP appendix asks that every configuration change forced by hardware or time be documented.

| Component | Planned | Actual | Reason |
|---|---|---|---|
| Conformal gate tolerance tau | 0.5 | **0.2** | At 0.5 no calibration turn was a violation (judge IP max 0.357 on 300 turns), so the gate was inert |
| Counterfactual PRI rollouts | 5 | 2 in the 50-profile run (5 in the 1000-profile config) | Time on the 3060 |
| Judge samples per item | 3 | 1 at temperature 0 | Greedy answers repeat; extra samples tripled judge time |
| Fine-tuning epochs | 2-3 | 2 (50-profile run), 3 (1000-profile run) | Epoch 3 overfits on 0.5B (validation loss rises); the best checkpoint is kept, so no harm beyond time |
| Micro-batch x accumulation | 1 x 16 | 2 x 8 for the RTX 5090 run (effective batch 16 unchanged; the 5090 copy shipped with 4 x 4) | Memory headroom on a 24 GB laptop card |
| Evaluation batch | Trainer default (8) | equal to the micro-batch | Out of memory at the end-of-epoch evaluation on 12 GB (BUGS.md B50) |
| Scaling study sizes | (not in SOP) | 0.5B, 3B, 7B; 1.5B dropped | Laptop GPU time |
| Memory arms | full system (gated) | decomposed, ungated | See 2.3 |
| ES-MemEval | simulated profiles | question answering over its real sessions | See 1.10 |
| ExTES | second evaluation profile set | cross-corpus arms only; no Success Rate | ExTES has no annotated hidden need |
| Concurrency of the four agents | concurrent preferred | sequential | One model resident on a 12 GB card (as the SOP allows) |

---

## 5. Still open against the SOP
- **Judge-human agreement** on a stratified sample (SOP III-C, "reported alongside every PRI number"): the
  forms exist (`human_eval/`), the human ratings have not been done.
- **Alpha sweep and the critic's recovery-versus-cost curve** (Expected Outcome 3): arms `gate_alpha_005/010/020`
  are defined but not run.
- **Out-of-distribution gate violation rate** on ExTES profiles (SOP III-B): arms defined, ExTES downloaded,
  not run.
- **Memory comparison** (Expected Outcome 4): arms and metrics ready (sections 1.9, 1.10, 2), not run yet.
- **Full-size 2 x 2, baselines, and the fine-tuned supporters on the test profiles** (Expected Outcomes 1-2):
  running on the RTX 5090 (`full_run.sh`).
- **Public release** of the pipeline and corpus: the GitHub repository is private for now.
