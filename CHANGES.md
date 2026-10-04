# Deviations from PLAN.md, and why

Everything in `PLAN.md` / the SOP is implemented. This file records where the code differs from the plan's
letter, and what was added that the plan described but did not name as a file. Nothing here changes a claim,
a metric, or an experiment.

## Renames

| Plan | Code | Why |
|---|---|---|
| `src/profile.py` | `src/profiles.py` | `profile` is a standard-library module name. With `src/` on `sys.path` our file shadows it, which breaks any dependency that imports the stdlib one. Same contents, same CLI. |

## Files added (the plan described the work but folded it into another module)

| File | What it is | Plan reference |
|---|---|---|
| `src/calibrate.py` | Runnable two-pass gate calibration: critic scores, then judge IP labels, then the risk curve. Split out because the critic and the judge are different models and only one may be resident, so it cannot be a function call inside `conformal.py`. | Sec. 10.4 |
| `src/fit_readiness.py` | Fits the readiness value function on corpus sessions and writes its reliability curve. | Sec. 13.2 |
| `src/teacher_rollouts.py` | Runs the privileged pacing teacher and builds the distillation dataset. | Sec. 13.3 |
| `src/external.py` | Maps ExTES and ES-MemEval into our profile schema, flagging every field we generated and marking sets where Success Rate is not applicable. | Sec. 14.2 |
| `CHANGES.md` | This file. | - |

## Implementation decisions the plan left open

1. **`echo` backend in `llm.py`** (new). A deterministic stub backend that answers in the right shape for the
   three prompt kinds (rating scale, JSON schema, free text). Every stage takes `--backend echo`, so the whole
   pipeline and the entire check suite run with no GPU, no downloads, and no network. It is not a model and is
   never used for results. Its canned lines vary by prompt hash, because a constant stub would trip the
   repetition filter and make dry runs fail for a reason unrelated to the pipeline.
2. **`LLM.view(role)` (RoleView) and per-call adapter toggling.** The plan requires the four agent roles, the
   corpus generator and the user simulator to share one resident base model. A view reuses the parent's
   backend and cache, applies that role's sampling parameters, and disables the supporter LoRA for roles that
   are not the supporter (`model.disable_adapter()` on Transformers, `lora_request=None` on vLLM). Without
   this, "one model at a time" and "the simulator is not the fine-tuned supporter" could not both hold.
3. **Readiness estimator is closed-form ridge regression** over the interpretable feature list, implemented
   with numpy only. The plan allowed a gradient-boosted or ridge model; ridge keeps the controller auditable
   and removes a scikit-learn dependency. Reliability curve and calibration error are reported either way.
4. **Hoeffding-Bentkus bound implemented locally** (`conformal.py`) with `math.comb` bisection rather than
   pulling in a conformal library. The plan asked for a library "if available"; the bound used is recorded in
   `calibration.json` under `bound`, so the choice is visible in the artifact.
5. **Rung classifier heuristic** (`grounding.classify_rung`): a hedge alone is not depth. "It sounds like a
   heavy week" is a reflection (L0); a hedge plus an inference marker (want / need / because / …) is L2. Without
   this distinction every ordinary reflection would be flagged as overreaching and the fallback rate would be
   meaningless.
6. **English-only screen is per turn**, not an average over the transcript. One Chinese turn among ten English
   ones has to be caught, and a transcript average hides it.
7. **`filter.py` prunes broken tails at the profile level.** If session 2 fails, sessions 3+ are dropped rather
   than kept, because the sequence they belong to no longer exists. The count is reported.
8. **Scores are written to `runs/<id>/scores/<arm>/`.** The plan wrote `scores/`; with one directory per arm,
   two arms cannot overwrite each other's files, and `report.py` reads either layout.
9. **Baseline memories expose the `NeedStateMemory` surface** (`brief`, `render_brief`,
   `forbidden_inferences`, `propose`, …, with the writes refused). This keeps `evaluate.py` free of branches on
   the memory switch, so an arm difference is a component difference and nothing else.
10. **`train.py --arm pacing_distill` requires `--resume-adapter`.** The plan specifies the pacing stage as a
    second pass over the same adapter stack; the CLI refuses to start a fresh adapter, so that cannot be done
    by accident.
11. **Removed a helper that would have broken the residency rule.** An earlier `counterfactual.rollouts_for_turn`
    took a simulator and a judge at once. PRI replay is therefore written as two passes: generate every
    seeker reply with the simulator resident, release it, then score with the judge resident.
12. **Eval-set path** for our own held-out profiles is `data/profiles/profiles_test.jsonl` (what
    `profiles.py --split` writes), not `data/corpus/`.

## Deliberate non-implementations, and what stands in

| Thing | Status | Stand-in |
|---|---|---|
| Real model weights, fine-tuning, corpus generation at scale | Not run | Every stage is runnable; `--backend echo` exercises the wiring. `train.py` needs a GPU and is untested here. |
| ExTES / ES-MemEval raw downloads | Not present | `src/external.py` exits with instructions naming `data/raw/PROVENANCE.md` and the PLAN Sec. 12.4 fallback when the file is missing. |
| EmpatheticDialogues download | Optional | `seeds.py` reads `data/raw/empatheticdialogues.jsonl` if present, else pulls it through `datasets`. |
| Sentence-transformer embeddings for the dense-retrieval baseline | Optional | Bag-of-words cosine fallback; `DenseMemory.backend` records which was used so the report can say so. |
| Crisis-content handling | Screen only | Flagged sessions go to `data/quarantine/` for review and are excluded from training; the rate is reported rather than silently dropped. |

## What was reused from the upstream COCOON repository

Copied into `reference_cocoon/` as reference, not imported: `speaker_active.md` (Chinese proactive
resistance-conditioned simulator - the behavioural spec for our English prompt), `speaker_en.md` (English
reactive speaker, our reactive baseline), and `listen.md` / `comfortScale.md` / `racScale.md` (AELS,
Comforting Responses, RAC item sets, cross-checked against our own implementations in `prompts/`). No upstream
Python is used: it hardcodes GPT-4o, internal cluster URLs (`http://gpuXX:...`), author-absolute paths and
placeholder API keys, and is Chinese-first. `tests/test_all.py` includes a check that our IP items do not drift
between the judge prompt and the critic prompt, which is the failure mode copying prompts around invites.

---

# Update: `modified_SOP_12340340_12340370.pdf` (read 2026-09-29)

The modified SOP drops **Enhancement 5 (adaptive disclosure pacing)** and is otherwise unchanged in
substance. Four enhancements, four deliverables. Everything else in this codebase still matches it.

What changed in the document:

| Place | Original SOP | Modified SOP |
|---|---|---|
| §II enhancement list | five modular enhancements | **four**; item (5), the pacing policy, is gone |
| Fig. 1 gap caption | Gap 3 "Addressed by Enh. 3 & 5" | "Addressed by Enh. 3" |
| Fig. 2 caption | memory, need-state tracking, **and adaptive pacing** | memory and need-state tracking |
| Fig. 3 | Analyzer/Strategist/Generator/Critic + **readiness → Adaptive Pacing** box | same pipeline without the pacing box; caption now "metrics gate the critic, and responses update the need-state memory" |
| §III sections | A–E enhancements, F fine-tuning, G evaluation | A–D enhancements, **E** fine-tuning, **F** evaluation |
| §III-E (old) | full adaptive-pacing section: ladder, readiness as a value, privileged-to-deployable distillation, fixed-pacing and greedy arms | **removed entirely** |
| Limitation 2 solution | "Enhancements 3 and 5 … through the pacing policy as a per-user control signal" | "Enhancement 3" only |
| Differentiators | four bullets, the memory one ending "alongside a readiness controller we intend to distil from a privileged view" | same four bullets, that clause removed |
| §VI outcomes | 5 deliverables | **4**; "Adaptive pacing" deliverable gone |
| Team split | Arpit: memory **and pacing**; Ashish: architecture **and metrics** | shared: corpus, fine-tuning **and turn-level metrics**; Arpit: temporal dynamics / need-state memory; Ashish: inference architecture / multi-agent |
| References | 9 | 9 (unchanged: [5] privileged proactivity is still cited for the hidden-layer persona in E1, [8] reactance for PRI, [9] event-level memory for the E4 baseline) |

Unchanged, and still implemented as written: the conformal critic gate with the intrusiveness budget, the
2 × 2 ablation, the typed grounding contract with `denied_inference`, IP and PRI with counterfactual
attribution and the three validity controls, need-state memory with its four baselines, QLoRA fine-tuning
with the `w/o thoughts` arm, the 12 GB serial-residency budget with two distinct models, the evaluation sets
(own reticent profiles, ExTES, ES-MemEval) and the judge-human agreement requirement.

## What was removed from the code (done)

| Deleted | Was |
|---|---|
| `src/readiness.py` | readiness value function, feature extraction, decay, reliability curve, `session_return` |
| `src/fit_readiness.py` | readiness fitting stage |
| `src/teacher_rollouts.py` | privileged teacher rollouts and the distillation dataset |
| `src/baselines/pacing_fixed.py`, `src/baselines/pacing_greedy.py` | fixed-pacing and unconstrained-probing arms |

| Trimmed | Change |
|---|---|
| `src/pacing.py` | reduced to the ladder: `LADDER`, `at_most`, `cap`, `permitted_rung`. `AdaptivePolicy`, `FixedRungPolicy`, `FixedRatioPolicy`, `GreedyPolicy`, `make_policy`, `PrivilegedTeacher`, `build_distillation_dataset`, `rung_entropy` are gone. `permitted_rung` now takes analyzer confidence and the memory brief - the two caps that belong to E2 and E4 - and no readiness value. |
| `src/agents.py` | `AgentPipeline` no longer takes `policy` or `readiness_model`; `step()` drops `pri_history`/`carried_readiness`; `TurnResult.readiness` removed; the rung comes straight from `permitted_rung`. |
| `src/baselines/listener_mono.py` | same signature change. |
| `src/evaluate.py` | no readiness model, no carried readiness across sessions, no `session_return`, no pacing switch in the arm summary. |
| `src/build_sft.py` | `pacing_rows_from_session` and `pacing_teacher_seed.jsonl` removed. |
| `src/train.py` | `--arm pacing_distill` removed; `--resume-adapter` is now a generic continue-training flag. |
| `src/report.py` | pacing frontier and readiness calibration sections removed; the fourth pre-registered comparison is now `mem_needstate` vs `mem_event`. |
| `src/common.py` | `runs/<id>/readiness/` is no longer created. |
| `configs/default.yaml` | the `pacing:` block is gone; `agents.analyzer_confidence_floor` still drives the rung cap. |
| `configs/arms.yaml` | `pacing` switch removed from `defaults` and from every arm; the nine `pace_*` arms and `supporter: sft_pacing_distilled` are gone. 22 arms remain. |
| `tests/test_all.py` | readiness and E5 pacing checks replaced by three ladder checks plus `scope.enhancement_5_is_absent`, which fails if any of the removed modules or symbols comes back. 64 checks. |

The ladder stays because Enhancement 2 rests on it: the Strategist is restricted to exploratory moves when
Analyzer confidence is below the floor, and `grounding.check_draft` raises `overreaching_depth` when a draft
reads deeper than the permitted rung. Neither is a pacing policy; both are in the modified SOP.

## Local serving through Ollama (added)

The SOP forbids proprietary APIs, and Ollama is now a first-class backend: `backend: ollama` in
`configs/models.yaml`, `ollama_tag` per role, `ollama_host` (or `OLLAMA_HOST`). `OllamaBackend` posts to
`/api/chat` with the standard library only - no Python client dependency - and `release()` sends
`keep_alive: 0` so the weights are dropped before the next role loads, which is how the one-model-at-a-time
rule survives on a 12 GB card. `scripts/fetch_data.py --ollama` pulls every tag in the config.

Training is the exception: Ollama serves GGUF and cannot fine-tune, so `src/train.py` stays on the
transformers backend, and serving the fine-tuned supporter through Ollama requires merging the adapter,
converting to GGUF, and `ollama create`.

---

# Update: metrics and harness for the baseline paper's Tables 3 and 4

| Added | What |
|---|---|
| `prompts/scale_basic.md` | now the **six** published basic metrics - Fluency, Diversity, Empathy, Information, Humanoid, Skillfulness - instead of four. |
| `src/metrics.py` | `basic_breakdown` (per-metric means plus `basic_avg_100`, rescaled to the paper's 0-100), `crs_dimensions` (Affective Improvement, Negative Helper Evaluations), `rac_dimensions` (Supportiveness, Management). Item groupings are explicit constants and are printed in the report: the instruments publish items, not groupings. |
| `src/convert_corpus.py` | flattens ESConv / ExTES / SMILE-style releases into our SFT format so one base model can be fine-tuned per corpus. Targets are response-only, because those corpora have no Analysis/Strategy - the same condition as our `w/o thoughts` arm. |
| `configs/arms.yaml` | eight `corpus_*` arms (ours, ours-w/o-thoughts, ESConv, ExTES, SMILE, plus three on the ExTES profile set) and an `adapters:` map. 26 arms. |
| `src/evaluate.py` | resolves the adapter from that map per arm, and warns loudly when an arm asks for a fine-tuned supporter that has no adapter mapped. This removes the manual `models.yaml` edit per arm. |
| `src/train.py` | accepts any `corpus_<name>` arm produced by the converter. |
| `src/report.py` | a **Training-corpus comparison** table: training dataset x {SR, Basic Avg, Aff, Neg, Sup, Man}, for our profiles and the ExTES profiles. |
| `scripts/run_all.py` | with `--train`, converts and fine-tunes on every external corpus whose raw file is present. |
| `tests/test_all.py` | three new checks: dimension groupings and the 0-100 rescale, the converter's response-only output and role normalisation, and adapter resolution per arm. 69 checks. |

Not implemented, deliberately: the paper's ESC-Rank evaluator (needs seven unreleased LoRA adapters) and the
released Chinese supporter checkpoints (SoulChat, MeChat, EmoLLM) - this project is English and open-weights,
and the SOP already states the comparison rests on the instruments, not on their code.
