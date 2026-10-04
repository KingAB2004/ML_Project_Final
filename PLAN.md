# planoutput.md — Full Implementation Plan

> **Scope note (2026-09-29).** This plan was written against the original SOP. The modified SOP
> (`modified_SOP_12340340_12340370.pdf`) drops **Enhancement 5, adaptive disclosure pacing**, and
> with it Sections 13, the pacing arms of 14.4, the readiness rows of the traceability table, and
> deliverable 5. The code has been trimmed to match; see `CHANGES.md`. Everything else below still
> stands. The disclosure ladder survives as an Enhancement 2 mechanism (depth cap), not as a policy.

**Project:** Enhancing Proactive Emotional Support Systems with Open-Source LLMs — Multi-Agent Architecture, Need-State Memory, and Psychological Evaluation.
**Source of truth:** `SOP_12340340_12340370.pdf` (Arpit Bhomia 12340340, Ashish Ranjan 12340370). Baseline paper: COCOON, "Look Beyond Feeling" (EMNLP 2025) = reference [1] in the SOP.
**Purpose of this document:** a complete, self-contained specification. A code-generating LLM should be able to produce the entire codebase from this file alone, without reading the SOP, the paper, or the upstream repository. It contains no code — only what each component is, what it reads, what it writes, how it decides, how it is wired to the others, and what check proves it works.

> Everything in the SOP is in scope: Enhancement 1 (corpus with temporal structure), Enhancement 2 (multi-agent + grounding contract + conformal critic gate), Enhancement 3 (IP and PRI with counterfactual attribution), Enhancement 4 (need-state trajectory memory), Enhancement 5 (adaptive disclosure pacing from a privileged view), Section III-F (fine-tuning on open models), Section III-G (evaluation protocol and baselines), Section IV (hardware/dataset envelope), Section VI (five expected deliverables). Nothing is dropped. Where the SOP itself says a route may fail, this plan names the decision gate and the fallback instead of pretending the risk away.

---

## Table of contents

0. Design rules that govern the whole build
1. Requirement trace: SOP claim → module → experiment → metric → deliverable
2. Hardware and model envelope
3. Repository layout and what each file owns
4. Configuration, run identity, determinism, logging
5. The LLM access layer (single resident model, role switching)
6. Data schemas (every artifact on disk)
7. Phase 0 — smoke checks before any real generation
8. Phase 1 — Enhancement 1: corpus construction with temporal structure
9. Phase 2 — Fine-tuning the supporter (SOP III-F)
10. Phase 3 — Enhancement 2: multi-agent architecture, grounding contract, conformal critic gate
11. Phase 4 — Enhancement 3: turn-level psychological metrics (IP, PRI) and judge validity
12. Phase 5 — Enhancement 4: need-state trajectory memory
13. Phase 6 — Enhancement 5: adaptive disclosure pacing
14. Phase 7 — Evaluation protocol, baselines, ablations, statistics, reporting
15. End-to-end wiring: the inference loop, turn by turn
16. Schedule, bottlenecks, compute budget
17. Risk register with concrete fallbacks
18. Acceptance criteria and self-checks per phase
19. Ethics, safety, and honesty constraints
20. Release artifacts

---

## 0. Design rules that govern the whole build

These rules are not style preferences. They are what makes a 12 GB single-GPU project with five interacting enhancements finishable. Violating them is how this project fails.

**R1 — Every stage reads files and writes files.** A stage is one script with a command-line entry point. It reads artifacts produced by earlier stages, writes new artifacts to a new directory, and exits. No orchestration framework, no message bus, no long-lived service, no in-memory pipeline spanning stages. Consequence: if annotation breaks, annotation is rerun, not the corpus generation that cost 20 GPU-hours.

**R2 — One model resident on the GPU at a time. Always.** Four agent roles are four system prompts over the same loaded base, not four models. The judge never shares a process with the generator. Corpus generation and judging are offline batch passes, not co-resident with inference. This is a hard constraint from the 12 GB budget (SOP IV-B) and it dictates the phase ordering: all generation for an arm finishes and is written to disk, then the generator is unloaded, then the judge is loaded and sweeps the whole directory.

**R3 — Structured output augments, never replaces, raw dialogue history.** This is the first of the SOP's two contract rules (III-B). Every agent and every judge prompt contains the verbatim user turns in addition to any structured state. There is no stage in this system where a component sees only a summary.

**R4 — Nothing enters persistent belief state ungrounded.** This is the second contract rule, and it is the bridge between Enhancement 2 and Enhancement 4. Any inference written to memory carries at least one verbatim user span, with session id, turn index and timestamp. A write without a span is rejected by the writer, not filtered later.

**R5 — Resumability by construction.** Each output record is keyed by a deterministic id. A stage skips ids already present in its output file. Crashing halfway through a 900-profile generation must cost only the current record.

**R6 — Every number reported comes with its sample size, its seed, and its judge-agreement figure.** The SOP commits to this explicitly (III-G: "report every headline number with the judge–human agreement measured on its stratified subsample"). The reporting stage refuses to emit a table cell that lacks n.

**R7 — Calibration knobs stay exposed.** Time-gap distributions, decay rates, filter thresholds, the intrusiveness budget α, the tolerance τ, ladder rung definitions, rollout counts — all live in one config file, none are hard-coded at a call site. Real judges drift and real generators run hot; the knobs are how the project is tuned without editing logic.

**R8 — Honest degradation.** When a configuration exceeds the hardware budget, the plan's instruction is to reduce scale and record the substitution in `reports/substitutions.md` (SOP Appendix). Never silently switch to a proprietary API. The SOP forbids proprietary APIs at every stage; the code should have no code path that can call one.

---

## 1. Requirement trace: SOP claim → module → experiment → metric → deliverable

This table is the contract between the SOP and the codebase. Every row must be implemented; the reporting stage has one section per row.

| # | SOP commitment | Module(s) that implement it | Experiment that tests it | Primary metric | Deliverable (SOP VI) |
|---|---|---|---|---|---|
| E1 | English proactive corpus with multi-session temporal structure, content-level Analysis/Strategy annotation | `seeds.py`, `profile.py`, `dialogue.py`, `sessions.py`, `filter.py`, `annotate.py` | Corpus statistics + annotation-ablation reproduction (`w/o thoughts`) | Success Rate; AELS survival rate; yield | 1 |
| E1a | Seed from EmpatheticDialogues, stratify by ESConv problem type, fall back to fully LLM-synthesized if yield is poor | `seeds.py` decision gate | Yield gate G1 (Sec. 8.9) | surviving profiles per 100 seeds | 1 |
| E1b | Proactivity injected at role-play time; simulated user did not seek help and initially resists | `prompts/user_proactive.md` | Resistance-onset check | mean first-turn disclosure depth | 1 |
| E1c | Linked sessions separated by sampled Δt; profile advanced between sessions | `sessions.py` | Cross-session recall/SR experiments in E4 | SR on session-3 terminal need | 1, 4 |
| E2 | Decompose listener into Analyzer / Strategist / Generator / Critic | `agents.py` | 2×2 ablation (decomposed × gated) | Success Rate, IP | 2 |
| E2a | Every Analyzer inference carries supporting quotes and a confidence score | `grounding.py` | Grounding-violation audit | span-validity rate | 2 |
| E2b | Strategist restricted to exploratory moves when Analyzer confidence is low | `agents.py` policy table | Low-confidence subgroup analysis | IP on low-confidence turns | 2 |
| E2c | Critic under a conformal risk gate with error budget α, nonconformity from IP | `conformal.py` | Risk-coverage sweep over α | empirical violation rate vs α | 2, 3 |
| E2d | Typed grounding with "an inference the user has denied" as highest-weight signal | `grounding.py` category table | Denied-inference re-proposal count | re-proposals per dialogue | 2 |
| E2e | Guarantee scope: exchangeability; cross-corpus arms are out of distribution and report empirical violation, not a guarantee | `conformal.py` + reporting | ExTES / ES-MemEval arms | empirical violation rate | 2, 3 |
| E2f | Critic falls back to a plain reflective statement if revision fails | `agents.py` revision loop | Fallback-rate accounting | fallback rate, cost curve | 2, 3 |
| E3 | Intrusiveness Penalty (supporter turn) | `metrics.py` | All arms | IP | 3 |
| E3a | Psychological Reactance Index (user's next turn), reactance = anger + negative cognition | `metrics.py` | All arms | PRI | 3 |
| E3b | Counterfactual attribution: replace supporter turn with neutral reflective control, re-generate user turn, PRI = r̄1 − r̄0 over N rollouts with CI | `counterfactual.py` | PRI computation itself | PRI with 95% CI | 3 |
| E3c | Three validity controls: resistance level withheld from judge; judge a different open model from generator/simulator; stratified human subsample with agreement reported | `metrics.py`, `human_eval/` | Judge reliability study | Cohen κ / Krippendorff α, Spearman ρ | 3 |
| E4 | Need chain persisted as revisable belief with status hypothesis/confirmed/disconfirmed, spans, timestamps, confidence | `memory.py` | Memory-baseline comparison | Success Rate + recall | 4 |
| E4a | Revise rather than append; resolved / disconfirmed-and-blocked / reinstated-at-reduced-confidence | `memory.py` update rules | Conflict-detection and abstention probes | re-proposal rate, abstention correctness | 4 |
| E4b | Decay on unconfirmed hypotheses proportional to elapsed interval | `memory.py` | Long-gap subgroup | SR by Δt bucket | 4 |
| E4c | Retrieval as a question: surface current best hypothesis + outstanding evidence, not similarity lookup | `memory.py` brief | Retrieval-mode ablation | SR vs dense-retrieval baseline | 4 |
| E4d | Baselines: session-isolated, flat running summary, dense retrieval, event-level [9] | `baselines/memory_*.py` | Memory arm table | SR, recall | 4 |
| E5 | Graded disclosure ladder: reflective acknowledgement → tentative inference → naming the terminal need | `pacing.py` ladder | Pacing arm table | SR, IP | 5 |
| E5a | Readiness defined as expected final disclosure outcome given history (a value function), persisted across sessions with decay ∝ Δt | `readiness.py` | Readiness-quality check | calibration of r̂ vs realized outcome | 5 |
| E5b | Privileged-to-deployable distillation: teacher sees profile/terminal need/scripted resistance, deployed policy sees conversation only | `pacing.py` + `train.py` stage 2 | Teacher-vs-student gap | SR, IP, rung-agreement | 5 |
| E5c | Fixed-pacing baselines spanning the original study's range + unconstrained information-seeking arm | `baselines/pacing_*.py` | Pacing arm table | SR vs IP frontier | 5 |
| F | Supervised fine-tuning of an open instruction-tuned base on the corpus with per-turn Analysis and Strategy in the target sequence; 4-bit QLoRA + gradient checkpointing; `w/o thoughts` arm | `build_sft.py`, `train.py` | Annotation ablation | SR, AELS, CRS, RAC | 1, 2 |
| G | Evaluate on own reticent profiles, ExTES profiles, ES-MemEval; report SR, basic metrics, AELS/CRS/RAC, IP/PRI, judge–human agreement | `evaluate.py`, `report.py` | Full protocol | all | 1–5 |
| IV | Open weights only, two distinct models, serial residency, document any reduced-scale substitution | `llm.py`, `configs/`, `reports/substitutions.md` | — | VRAM peak log | all |

---

## 2. Hardware and model envelope

### 2.1 The budget

Target: **one 12 GB GPU** (RTX 3060 class), as stated in SOP IV-B. The plan must also run, unchanged except for config values, on a 16–24 GB card; the only differences are batch size, quantization choice, and rollout counts. Nothing in the architecture may assume more than 12 GB.

### 2.2 Model roles — exactly two distinct models

The SOP fixes this: generation and role-play may share one model, the judge must be neither the supporter nor the simulator.

| Role | Model | Quantization | Approx VRAM | Loaded during |
|---|---|---|---|---|
| Corpus generator + user simulator (role-play) + the four agent roles | `Qwen2.5-7B-Instruct` | AWQ / GPTQ 4-bit for inference | ~6 GB weights + KV | Phases 1, 3, 5, 6 generation passes |
| Supporter under test | same base + QLoRA adapter from Phase 2 | 4-bit base, fp16 adapter | ~7 GB at train time, ~6.5 GB at inference | Phases 2–7 |
| Judge / annotator / filter | `Mistral-Nemo-Instruct-2407` (12B) — different family | AWQ 4-bit | ~8–9 GB | every scoring pass, run alone |
| Sentence embeddings (dense-retrieval memory baseline only) | `BAAI/bge-small-en-v1.5` | fp32 CPU | CPU only | Phase 5 baseline |
| Readiness regressor (E5) | small head over frozen embeddings | — | CPU | Phase 6 |

**Software stack** (SOP IV-B): Python, PyTorch, HuggingFace Transformers and Datasets for models and data; PEFT plus bitsandbytes for 4-bit quantized low-rank adapter training; a batched inference backend (vLLM, or Transformers with manual batching where vLLM cannot hold the chosen quantization on 12 GB); a conformal-prediction / risk-control library for the critic gate of Sec. 10.4, with the bound it uses recorded in the calibration artifact; sentence-transformers for the embedding features of the dense-retrieval baseline and the readiness estimator; scikit-learn for the readiness regressor and the agreement statistics. Pin every version in `requirements.txt` and copy the resolved versions into each run directory.

Substitution rules, to be recorded if used: if the 12B judge will not fit alongside required KV cache, drop to `Llama-3.1-8B-Instruct` (still a different family from Qwen) and note it. If AWQ checkpoints are unavailable for a chosen model, use GPTQ or bitsandbytes NF4 and note the change. Never substitute a proprietary API.

### 2.3 Serial schedule discipline

Generation and judging never overlap. The runner is a shell script per phase that: loads generator → produces all dialogues for all arms of that phase into `runs/<run_id>/dialogues/` → exits the process (freeing VRAM) → loads judge → sweeps the directory → writes `runs/<run_id>/scores/`. Because scoring is a separate pass over files, adding a metric later never requires regenerating dialogues.

### 2.4 Throughput assumptions to size the work

Use these as planning estimates and replace them with measured numbers after Phase 0: a 4-bit 7B on a 12 GB card at batch 8–16 with vLLM-style continuous batching yields roughly 300–700 output tokens/second aggregate; a supporter turn plus agent scaffolding is ~400–900 output tokens; a judge scale pass is ~120 output tokens. Every phase below states its token cost in these units so the schedule can be re-derived when the measured rate differs.

---

## 3. Repository layout and what each file owns

Flat, script-per-stage, no package ceremony. One responsibility per file; the file name is the stage name used on the command line and in run directories.

```
newsop/
  configs/
    default.yaml            # every knob in the project
    models.yaml             # model ids, quantization, sampling params per role
    arms.yaml               # experiment arms: which components on/off per arm
  prompts/
    user_proactive.md       # E1 English proactive, resistance-conditioned user simulator
    user_reactive.md        # reactive confiding baseline simulator
    profile_seed.md         # profile + need-chain construction
    profile_advance.md      # between-session profile advancement
    listen_stage.md         # two-stage generation: listening phase supporter
    suggest_stage.md        # two-stage generation: suggestion phase supporter
    annotate_turn.md        # content-level Analysis + Strategy annotation
    agent_analyzer.md
    agent_strategist.md
    agent_generator.md
    agent_critic.md
    scale_aels.md           # Active-Empathic Listening Scale (filtering + eval)
    scale_crs.md            # Comforting Responses Scale
    scale_rac.md            # RAC Scale
    scale_success.md        # Success Rate: did the supporter identify the terminal need
    scale_basic.md          # fluency / empathy / coherence / identification basics
    metric_ip.md            # Intrusiveness Penalty items
    metric_pri.md           # Psychological Reactance Index items
    control_reflective.md   # neutral reflective control turn for counterfactual replay
    teacher_privileged.md   # E5 privileged pacing teacher
  src/
    common.py               # ids, seeds, jsonl io, run dirs, span validation, retries
    llm.py                  # the only place a model is loaded or sampled
    judge.py                # scale rendering + score parsing + aggregation
    seeds.py                # Phase 1.1 seed harvesting and stratification
    profile.py              # Phase 1.2-1.3 need chain + persona
    dialogue.py             # Phase 1.4 two-stage single-session generation
    sessions.py             # Phase 1.5 multi-session linking with time gaps
    filter.py               # Phase 1.6 AELS + structural quality filtering
    annotate.py             # Phase 1.7 Analysis/Strategy annotation
    build_sft.py            # Phase 2 target-sequence construction (+ w/o thoughts arm)
    train.py                # Phase 2 QLoRA SFT; Phase 6 second-stage distillation
    grounding.py            # Phase 3 typed grounding contract and validators
    agents.py               # Phase 3 Analyzer/Strategist/Generator/Critic + revision loop
    conformal.py            # Phase 3 risk-controlled gate (calibrate + apply)
    metrics.py              # Phase 4 IP and PRI judges
    counterfactual.py       # Phase 4 intervention replay for PRI
    memory.py               # Phase 5 need-state trajectory memory
    pacing.py               # Phase 6 ladder, privileged teacher, deployed policy
    readiness.py            # Phase 6 readiness value estimation + cross-session decay
    evaluate.py             # Phase 7 arm runner (dialogue rollouts per arm)
    report.py               # Phase 7 tables, CIs, agreement, substitutions
    baselines/
      memory_none.py        # session-isolated
      memory_summary.py     # flat running summary
      memory_dense.py       # dense retrieval over turns
      memory_event.py       # event-level baseline in the style of [9]
      pacing_fixed.py       # fixed rung / fixed question-ratio arms
      pacing_greedy.py      # unconstrained information-seeking arm
      listener_mono.py      # monolithic single-pass listener (2x2 ablation cell)
  human_eval/
    sample.py               # stratified subsample selection for human rating
    forms/                  # rubric sheets the two team members fill in
    agreement.py            # kappa / alpha / rho between judge and humans
  data/                     # immutable inputs and produced corpus
  runs/                     # one directory per run_id, everything reproducible
  reports/                  # final tables, figures, substitutions.md
  tests/                    # the runnable self-checks named in Sec. 18
```

**Why this shape.** The stage boundaries are exactly the points at which the SOP demands an inspection point (III-B: "no boundary at which a safety check could intervene" is the limitation being fixed). Each agent boundary is a file boundary and a JSON artifact boundary, so a human can read what the Analyzer believed before the Generator spoke. That auditability is the contribution, not the decomposition itself.

### 3.1 Upstream COCOON code: what to take, what to ignore

The upstream repo in `COCOON/` is evaluation scaffolding for Chinese data on a GPT-4o backend and internal cluster URLs. Treat it as documentation, not dependency.

| Upstream item | Verdict | Use here |
|---|---|---|
| `prompt/speaker_active.md` (Chinese, proactive, has `{need}`) | **Behavioral spec** | Translate the behavior into `prompts/user_proactive.md`: user did not seek help, feels surprise/guardedness, withholds needs early, opens only under safe guidance, dialectical about advice, short informative turns. Do not translate literally; write English that reads naturally. |
| `prompt/speaker_en.md` (English, reactive, omits `{need}`) | **Evidence + baseline** | Confirms no English proactive corpus follows from released artifacts; becomes `prompts/user_reactive.md` for the reactive baseline arm. |
| `prompt/listen.md`, `comfortScale.md`, `racScale.md` | **Cross-check only** | They confirm the item sets and the 1–7 anchors for AELS / CRS / RAC and a parse-friendly "Question number: Score" output format. Re-derive our item wording from the published instruments and record it verbatim in `prompts/`, then diff against these to make sure no instrument was misread. |
| `eval/scorer/*.py` | **Do not inherit** | Hardcoded `sk-##` keys, GPT-4o, author-absolute paths, empty data paths, `input()` calls. The only transferable idea is the regex-per-item score parse with a retry loop and a per-item average over n samples — reimplement that in `judge.py` properly. |
| `eval/LLMchat.py` | **Do not inherit** | Chinese-first, references files that do not exist in the repo, `maxTurn` read from the wrong config. The transferable idea is the two-history design: the simulator and the supporter each keep their own message list with roles swapped, plus a flat transcript string for judging. Keep that idea in `dialogue.py`. |
| `eval/model.py`, `yaml/*` | **Do not inherit** | Proprietary endpoints. `llm.py` replaces all of it. |
| `utils.py` | **Do not inherit** | `gbk` encoding default and an unconditional `'./'` prefix are Chinese-corpus artifacts. `common.py` replaces it. |
| `eval/scorer/runGetEscRank.py` | **Out of scope** | Needs seven unreleased LoRA adapters. Not part of our protocol. |
| COCOON corpus / construction pipeline | **Not released** | This is why E1 exists and why comparison is standalone (SOP III-G). |

---

## 4. Configuration, run identity, determinism, logging

### 4.1 One config file

`configs/default.yaml` holds every knob. Nothing numeric may be written at a call site. The config is copied verbatim into each run directory, so a run is reproducible from its own folder. Knobs, grouped, with the defaults this plan assumes:

| Group | Knob | Default | Meaning / why it must be tunable |
|---|---|---|---|
| corpus | `n_profiles_target` | 1200 attempted | surviving count after filtering is what matters |
| corpus | `sessions_per_profile` | sample 2–4, mean 3 | multi-session structure of Fig. 2 |
| corpus | `turns_per_session` | 8–12 supporter turns | ≈19 total turns, matching the baseline's average length |
| corpus | `gap_distribution` | log-uniform, 1 day – 8 weeks | creates the recency-vs-relevance tension E4 needs |
| corpus | `need_chain_depth` | 3 (surface feeling → intermediate → terminal need) | the SOP's bounded need-inference chain |
| corpus | `resistance_levels` | {low, medium, high} stratified 1:2:1 | the hidden variable the supporter's behavior can move |
| filter | `aels_min_mean` | 5.0 / 7 | filtering threshold; re-tune after Phase 0 |
| filter | `aels_min_item` | 3 / 7 | reject dialogues that fail any single listening dimension |
| judge | `judge_samples_per_item` | 3, temperature 0.0 with nucleus off | stability of scale scores |
| conformal | `alpha` (intrusiveness error budget) | 0.10 | the SOP's α |
| conformal | `tau` (IP tolerance defining a violation) | IP_norm > 0.5 | what counts as "exceeding the tolerance" |
| conformal | `delta` (UCB confidence) | 0.10 | distribution-free bound confidence |
| conformal | `n_calibration_turns` | 300 | held-out, judge-labelled |
| critic | `max_revisions` | 2 | then fall back to reflective statement |
| pri | `n_rollouts` | 5 per condition | PRI = mean over rollouts |
| pri | `turns_sampled_per_dialogue` | 4 | cost control; sampled by stratified position |
| memory | `decay_half_life_days` | 14 for hypothesis-status nodes | decay ∝ elapsed interval |
| memory | `reinstate_confidence_factor` | 0.5 | resurfacing concern returns at reduced confidence |
| memory | `min_spans_to_confirm` | 2 | promotion hypothesis → confirmed |
| memory | `min_spans_to_reopen_disconfirmed` | 2 new contradicting spans | otherwise re-proposal stays blocked |
| pacing | `ladder` | L0 reflect, L1 open explore, L2 tentative inference, L3 name terminal need | the graded ladder |
| pacing | `readiness_thresholds` | L1≥0.35, L2≥0.55, L3≥0.75 | mapping readiness to the highest permitted rung |
| pacing | `readiness_decay_half_life_days` | 14 | cross-session persistence with decay |
| eval | `n_dialogues_per_arm` | 150 | statistical power vs compute |
| eval | `bootstrap_resamples` | 10000 | paired bootstrap CIs |
| eval | `seed` | 20260928 | global root seed |

### 4.2 Run identity and determinism

A run id is `<UTC timestamp>-<short git sha>-<arm name>`. Everything a run writes goes under `runs/<run_id>/`, including a copy of the config, the resolved model ids and quantization, the measured peak VRAM, and the library versions.

Determinism rule: a per-record seed is derived from the root seed plus the record id plus the stage name, so the same profile regenerates identically whether it is record 3 or record 800 of a batch, and PRI rollout *k* of the factual condition pairs with rollout *k* of the control condition. Sampling temperatures are per role in `models.yaml`: user simulator warm (≈0.8–0.9) because a stiff simulator makes resistance unrealistic; supporter moderate (≈0.7); Analyzer/Strategist/Critic cold (≤0.3) because their outputs are parsed; judge 0.0.

### 4.3 Logging

Every model call appends one line to `runs/<run_id>/calls.jsonl`: role, prompt hash, token counts, latency, sampling params, retry count, and whether structured parsing succeeded on the first attempt. This file answers "why did this arm cost 9 hours" and "how often does the Analyzer emit unparseable JSON" without instrumenting anything twice. A per-phase summary of it goes into the report.

---

## 5. The LLM access layer

`llm.py` is the only module that loads weights or samples tokens. Everything else asks it for text or for parsed structure. It provides four capabilities and no more:

1. **Load one model by role name** from `models.yaml`, with quantization and context length from config. Loading a second model while one is resident is an error, not a fallback — this enforces R2 at the code level rather than by discipline.
2. **Batch chat completion**: given a list of (system prompt, message list) pairs, return completions. Batching is what makes a 12 GB card usable; every stage should hand it hundreds of prompts at once rather than looping one at a time.
3. **Structured completion with repair**: given a prompt and a target schema, sample, attempt to parse, and on failure re-prompt once with the parse error appended, then a second time with a strictly reduced schema, then give up and record the failure. Failures are recorded as records with a `parse_failed` flag, never dropped silently — the failure rate is itself a number the report shows.
4. **Deterministic cache**: key = hash(role, model id, system prompt, messages, sampling params, seed). Hits are served from `runs/<run_id>/cache/` or a shared cache directory. This is what makes reruns after a crash cheap and makes the 2×2 ablation affordable, because arms that share a prefix of computation share cache entries.

No streaming, no async orchestration, no agent framework. The agent loop of Phase 3 is a for-loop over four prompt templates.

---

## 6. Data schemas

Schemas are specified as field tables. Every artifact is JSON or JSONL with these fields; a stage validates its inputs against the table and fails loudly on a missing required field rather than defaulting.

### 6.1 Seed situation — `data/seeds/seeds.jsonl`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `seed_id` | string | yes | `sd######` |
| `source` | enum: `empatheticdialogues` \| `synthetic` | yes | which of the SOP's two routes produced it |
| `source_ref` | string | yes | original dataset row identifier, for traceability |
| `situation_text` | string | yes | human-written narrative (ED) or sampled situation (synthetic) |
| `emotion_label` | string | yes | ED emotion label, or generated |
| `problem_type` | string | yes | ESConv problem-type category used for stratification |
| `length_tokens` | int | yes | used by the sustainability screen |
| `sustains_chain` | bool | yes | screen result: can this situation plausibly support a 3-step need chain |
| `screen_reason` | string | yes | one sentence, so a rejected seed can be audited |

### 6.2 Profile — `data/profiles/profiles.jsonl`

The profile is COCOON's four dimensions (emotion, feeling, need, memory) plus the additions the SOP specifies: the bounded need chain, the stratified persona with a hidden concern, and a scripted resistance level.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `profile_id` | string | yes | `p######` |
| `seed_id` | string | yes | provenance |
| `route` | enum: `seeded` \| `synthetic` | yes | reported per SOP III-A ("report which of the two routes produced the released corpus") |
| `emotion` | string | yes | emotional state label + short gloss |
| `feeling` | string | yes | surface feeling as the user would express it |
| `need_chain` | list of 3 nodes | yes | see 6.3; node 0 = surface feeling, node 2 = terminal need |
| `terminal_need` | string | yes | node 2 text, duplicated for convenience |
| `memory` | list of events | yes | recent events; each has `text`, `when_relative`, `salience` 0–1 |
| `persona_surface` | object | yes | observable traits the supporter may see: communication style, verbosity, tone, topics volunteered |
| `persona_hidden` | object | yes | what the supporter must uncover: `terminal_need`, `resistance_level`, `disclosure_triggers`, `disclosure_blockers` |
| `resistance_level` | enum low/med/high | yes | scripted; never shown to the judge (E3 control 1) |
| `problem_type` | string | yes | stratification key |
| `created_by` | object | yes | model id, prompt hash, seed |

### 6.3 Need-chain node (used inside profiles and, with status, inside memory)

| Field | Type | Meaning |
|---|---|---|
| `node_id` | string | `nd###`, stable across sessions |
| `depth` | 0/1/2 | surface feeling / intermediate / terminal need |
| `text` | string | the inference in one clause |
| `parent_id` | string or null | chain structure |

### 6.4 Session and turn — `data/corpus/sessions.jsonl`

One record per session; sessions of one profile are linked by `prev_session_id` and carry the sampled gap.

| Field | Type | Meaning |
|---|---|---|
| `session_id` | string | `p######-s#` |
| `profile_id` | string | owner |
| `session_index` | int | 1-based |
| `prev_session_id` | string or null | link (Fig. 2) |
| `gap_days_from_prev` | float or null | sampled Δt, injected into the follow-up context |
| `profile_snapshot` | object | the profile **as advanced for this session** (needs partially resolved, others intensified, new events, decayed salience) |
| `stage_boundary_turn` | int | turn index where generation switched from listening phase to suggestion phase |
| `turns` | list of turn objects | see below |
| `qc` | object | filter scores and verdict, filled by `filter.py` |
| `annotations_complete` | bool | filled by `annotate.py` |

Turn object:

| Field | Type | Meaning |
|---|---|---|
| `turn_index` | int | 0-based, supporter opens (the system initiates — this is a proactive system) |
| `role` | enum `supporter` \| `user` | — |
| `text` | string | the utterance |
| `phase` | enum `listening` \| `suggestion` | two-stage generation provenance |
| `analysis` | string or null | content-level free-form Analysis label (supporter turns only) |
| `strategy` | string or null | content-level free-form Strategy phrase (supporter turns only) |
| `ladder_rung` | enum L0–L3 or null | disclosure depth of this supporter turn (E5; annotated for all corpus turns so the teacher has labels) |
| `disclosure_depth` | int 0–3 or null | for user turns: how much of the need chain the user has revealed by this turn |
| `meta` | object | model id, sampling params, seed, retries |

### 6.5 Annotation record (kept separately so annotation can be rerun) — `data/corpus/annotations.jsonl`

| Field | Meaning |
|---|---|
| `session_id`, `turn_index` | join key |
| `analysis` | content-level psychological analysis of the user's state at that point, free-form phrase (deliberately *not* a fixed taxonomy — this is the baseline paper's stated advantage and we preserve it) |
| `strategy` | content-level description of the move the supporter is making |
| `ladder_rung` | L0–L3 |
| `annotator` | judge model id + prompt hash |
| `parse_failed` | bool |

### 6.6 Analyzer output (E2) — one JSON per supporter turn

| Field | Type | Meaning |
|---|---|---|
| `emotional_state` | object | `label`, `gloss`, `evidence_spans`, `confidence` 0–1 |
| `implicit_needs` | list | each: `text`, `depth`, `evidence_spans`, `confidence`, `status_hint` |
| `resistance_estimate` | object | `level` low/med/high, `evidence_spans`, `confidence` |
| `open_questions` | list of strings | what is still unknown; feeds the Strategist and the memory brief |
| `grounding_report` | object | per-claim validation result from `grounding.py` |

An **evidence span** is: `session_id`, `turn_index`, `char_start`, `char_end`, `quote`. Validation requires `quote` to appear verbatim in the referenced user turn at the referenced offsets. A claim whose span fails validation is downgraded to confidence 0 and cannot be written to memory (R4).

### 6.7 Strategist output (E2)

| Field | Meaning |
|---|---|
| `plan` | content-level strategy phrase (same vocabulary style as corpus annotation, so training and inference agree) |
| `justification_chain` | ordered list of short steps from evidence to plan |
| `target_node_id` | which need-chain node this move probes, or null |
| `requested_rung` | L0–L3 the Strategist wants |
| `allowed_rungs` | the set permitted by the pacing policy and the confidence rule |
| `restricted_reason` | set when the Analyzer's confidence was below threshold and only exploratory moves were allowed |

### 6.8 Critic output (E2)

| Field | Meaning |
|---|---|
| `ip_pred` | critic's own IP estimate for the draft, normalized 0–1 |
| `item_scores` | per-item IP rubric scores |
| `grounding_violations` | list of typed violations found in the draft, with category and weight |
| `nonconformity` | the scalar score the gate consumes |
| `decision` | `release` \| `revise` \| `fallback` |
| `feedback` | structured revision instructions when `revise` |
| `revision_index` | 0,1,2 |

### 6.9 Memory node (E4) — `runs/<run_id>/memory/<profile_id>.jsonl`

| Field | Type | Meaning |
|---|---|---|
| `node_id` | string | stable; matches profile chain nodes only in the oracle case, otherwise system-generated |
| `depth` | 0/1/2 | position in the inferred chain |
| `text` | string | the belief |
| `parent_id` | string or null | chain structure |
| `status` | enum `hypothesis` \| `confirmed` \| `disconfirmed` \| `resolved` | the SOP's three statuses plus `resolved` for a need that was acted on |
| `confidence` | float 0–1 | current, after decay |
| `confidence_raw` | float | before decay, so decay is auditable |
| `supporting_spans` | list of evidence spans | with timestamps |
| `contradicting_spans` | list of evidence spans | the disconfirmation record, first-class |
| `first_seen_at`, `last_updated_at` | ISO timestamps + session ids | — |
| `blocked_from_reproposal` | bool | set when disconfirmed |
| `reinstated_count` | int | times a resurfaced concern came back at reduced confidence |
| `history` | list of transitions | each: from-status, to-status, trigger span, timestamp, reason |

### 6.10 Judge score record — `runs/<run_id>/scores/<scale>.jsonl`

| Field | Meaning |
|---|---|
| `target` | what was scored: session id, or session id + turn index |
| `scale` | `aels` \| `crs` \| `rac` \| `success` \| `basic` \| `ip` \| `pri` |
| `items` | item number → score (1–7, or rubric-specific) |
| `mean`, `normalized` | aggregate and 0–1 normalization |
| `n_samples` | judge samples averaged |
| `judge_model` | id, so the report can assert judge ≠ supporter ≠ simulator |
| `context_redactions` | list of what was withheld, e.g. `resistance_level`, `terminal_need` — the audit trail for E3 control 1 |
| `parse_failed` | bool |

### 6.11 PRI intervention record — `runs/<run_id>/scores/pri_rollouts.jsonl`

| Field | Meaning |
|---|---|
| `session_id`, `turn_index` | the supporter turn treated as the intervention |
| `condition` | `factual` \| `control` |
| `rollout_index` | 0..N−1, paired across conditions by seed |
| `supporter_text` | actual turn, or the neutral reflective control |
| `user_reply` | the re-generated user turn being scored |
| `reactance_score` | judge's PRI score for that reply |
| `pri` | computed only in the aggregate record: r̄1 − r̄0 with CI |

### 6.12 Conformal calibration artifact — `runs/<run_id>/conformal/calibration.json`

| Field | Meaning |
|---|---|
| `alpha`, `tau`, `delta` | the budget, the violation tolerance, the UCB confidence |
| `n_calibration` | number of held-out judge-labelled turns |
| `score_name` | definition of the nonconformity score |
| `lambda_grid` | thresholds evaluated |
| `risk_curve` | per-λ empirical risk and its upper confidence bound |
| `lambda_hat` | selected threshold |
| `distribution_id` | generator + profile distribution the calibration is valid for |
| `provenance` | which split, which run, which judge |

### 6.13 Readiness record (E5) — `runs/<run_id>/readiness/<profile_id>.jsonl`

| Field | Meaning |
|---|---|
| `session_id`, `turn_index` | when |
| `readiness` | r̂ ∈ [0,1], the estimate used to cap the ladder |
| `features` | the inputs the estimator saw (so it is auditable) |
| `carried_from_prev_session` | r̂ at end of previous session |
| `decayed_by` | gap days and the decay factor applied |
| `permitted_rung` | the cap that resulted |
| `realized_outcome` | filled after the session ends: did the user verbalize the terminal need, and at what turn |

---

## 7. Phase 0 — smoke checks before any real generation

One day of work that prevents weeks of waste. Nothing in Phases 1–7 starts until all five pass, and their measured numbers replace the estimates in Sec. 2.4.

1. **Residency check.** Load the generator alone, measure peak VRAM at the configured context length and batch size; unload; load the judge alone, same measurement. Write both to `reports/vram.md`. If either exceeds the budget, change quantization or context length now, not after a failed 12-hour run.
2. **Structured-output check.** Send 50 Analyzer-style prompts and 50 judge-scale prompts; record first-attempt parse rate. If the judge's first-attempt parse rate is below ~90%, simplify the response format (numbered "item: score" lines, exactly as the upstream prompts did, beat free-form JSON for small models) before building on it.
3. **Throughput check.** Measure output tokens/second at the chosen batch size for both models. Recompute the Sec. 16 schedule from the measured rate.
4. **Span-validation check.** Hand `grounding.py` a handful of hand-written claims with correct, off-by-a-few-characters, and fabricated spans, and confirm it accepts only the first. This validator gates every memory write in the project; it must be correct before anything writes memory.
5. **End-to-end thin slice.** Two profiles, one session each, three supporter turns, one filter pass, one annotation pass, one AELS score, one IP score. This exercises every file boundary in the project at a cost of minutes, and it is the regression test to re-run after any refactor.

---

## 8. Phase 1 — Enhancement 1: corpus construction with temporal structure

This phase is the critical path: every later enhancement consumes its output. Build it in the order below, and gate it at 8.9 before scaling up.

### 8.1 Seeds (`seeds.py`)

Purpose: obtain situations to build profiles from, following the SOP's preference for human-written narratives over LLM-sampled situations, while keeping the fully-synthetic route as a declared fallback.

Inputs: EmpatheticDialogues (human-written conversational narratives, reference [4]); ESConv (reference [2]) for its problem-type taxonomy.

Steps:
1. Download both datasets into `data/raw/`. Record dataset version/commit in `data/raw/PROVENANCE.md`.
2. From EmpatheticDialogues, take the situation field of each conversation, deduplicate near-identical situations, and drop those that are help requests rather than narratives (the SOP's point is that ED situations are narrated, not asked).
3. Derive the ESConv problem-type list once and store it as a fixed list in `data/raw/problem_types.json`. Assign each ED situation a problem type by a judge-model classification pass (single-label, from the fixed list, plus an `other` bucket). This is the stratification key.
4. **Sustainability screen.** For each candidate, ask the judge whether the situation plausibly supports a three-step need chain (surface feeling → intermediate need → terminal need) and require a one-sentence reason. This is where the SOP's own concern — "many of those situations are short and may not sustain a three-step need chain" — is tested per item rather than assumed.
5. Stratified sample: balance across problem types, and within type across ED emotion labels, up to `n_profiles_target`.

Output: `data/seeds/seeds.jsonl` (6.1), plus `reports/seed_stats.md` with per-type counts and screen pass rates.

### 8.2 Need chain (`profile.py`, part 1)

For each surviving seed, construct the bounded need-inference chain: depth 0 is the surface feeling as the user would say it, depth 1 an intermediate unmet need, depth 2 the terminal need. Three constraints, each enforced by a validator, not by prompt hope:

- **Boundedness:** exactly three nodes; a generator that returns four is re-prompted, then rejected.
- **Non-triviality:** depth 2 must not be a paraphrase of depth 0. Check with an embedding-similarity ceiling plus a judge yes/no ("is the terminal need merely a restatement of the surface feeling?"). This is the single most important quality check in the whole corpus, because a chain whose terminal need is obvious makes Success Rate meaningless and makes the entire proactive premise vacuous.
- **Inferability:** the chain must be recoverable from the situation plus plausible user disclosure — a judge pass asks whether a skilled listener could reach depth 2 from this material.

### 8.3 Persona with a hidden layer (`profile.py`, part 2)

Split the profile into what the supporter can observe and what it must uncover, per SOP III-A. `persona_surface` holds communication style, verbosity, tone, and the topics the user will volunteer unprompted. `persona_hidden` holds the terminal need, the scripted `resistance_level`, and explicit `disclosure_triggers` (supporter behaviors that lower resistance: accurate reflection, validation, patience) and `disclosure_blockers` (behaviors that raise it: premature advice, naming an undisclosed feeling, repeated direct questioning).

The triggers and blockers are what make resistance *responsive* rather than a fixed constant: the simulator's prompt instructs it to move along the resistance dimension according to them. Without this the whole intrusiveness story cannot be exercised, because nothing the supporter does would change the user's guardedness.

Stratify `resistance_level` 1:2:1 (low:medium:high) so every arm has all three and subgroup analysis is possible.

### 8.4 Two-stage session generation (`dialogue.py`)

Purpose: generate one session between a proactive supporter and the resistant user simulator, in two phases as in the baseline methodology.

Wiring (this is the loop the baseline repo gets right and should be kept): the simulator and the supporter each hold their own message list with roles mirrored — what is `assistant` for one is `user` for the other — plus a flat human-readable transcript that judges consume. The simulator's system prompt is `prompts/user_proactive.md` filled from the profile snapshot, including `need` (the Chinese proactive prompt has this field; the released English one does not, which is exactly the gap E1 closes).

1. **The system opens.** A proactive greeting/check-in, not a response to a request. The user did not seek help. First supporter turn comes from a small pool of natural openers (varied, config-listed) so the corpus is not all one sentence.
2. **Listening phase** (`prompts/listen_stage.md`): supporter reflects, validates, explores, and probes gently; no advice. Runs until either `stage_boundary` conditions are met (user disclosure depth ≥ 1 and at least k turns) or a turn cap.
3. **Suggestion phase** (`prompts/suggest_stage.md`): supporter may offer perspective and concrete suggestions tied to the inferred need. The user remains dialectical about advice rather than compliant — the simulator prompt forbids bare "okay, I'll try".
4. Each turn is recorded with its phase, and each user turn with the `disclosure_depth` the simulator was instructed to be at, so the readiness labels of E5 exist without a second pass.
5. Termination: turn cap from config, or a judge-free structural check (user has verbalized the terminal need and the supporter has responded to it).

Output: session records (6.4) without `qc` or annotations.

### 8.5 Multi-session linking (`sessions.py`)

This is the part that makes Limitation 3 addressable, so it is specified tightly.

1. Sample the session count per profile (2–4) and the gaps Δt (log-uniform, 1 day–8 weeks). Log-uniform matters: it produces both "next day" and "six weeks later" in the same corpus, which is what makes recency-vs-relevance a real tension rather than a uniform blur.
2. **Advance the profile** between sessions with `prompts/profile_advance.md`, under explicit rules rather than free-form drift: a need the previous session acted on moves toward resolved; an unaddressed need may intensify; new events accumulate into `memory` with recent timestamps; the salience of earlier events decays with the gap; occasionally a new concern displaces the old one as the focus (Fig. 2's "new concern emerges"). Record, per session, which of these transitions were applied — this is the ground truth against which E4's memory updates are scored.
3. **Inject the gap into the follow-up context**: the supporter's context for session *k*>1 states the elapsed interval in natural terms ("it's been about three weeks"), and the simulator's context states what has changed. Both must be explicit, because a memory system cannot be evaluated on time it was never told about.
4. Keep chain node ids stable across sessions so cross-session need tracking is measurable; a displaced concern introduces new node ids rather than mutating old ones.

### 8.6 Quality filtering (`filter.py`)

Two layers, cheap first.

**Structural (no model calls):** turn count in range; alternating roles; no empty or truncated turns; no verbatim repetition of a previous turn by either side; no leakage of profile internals into user text (the simulator must not recite its own `terminal_need` verbatim in the first two turns, and must never quote the prompt); English only; no crisis content flagged by the safety screen of Sec. 19 escaping into training data without review.

**Model-based (AELS, `prompts/scale_aels.md`):** the judge scores the supporter on the Active-Empathic Listening Scale (sensing / processing / responding items, 1–7). Keep a session if mean ≥ `aels_min_mean` and no item < `aels_min_item`. Store all scores in `qc` even for rejected sessions — the rejection distribution is a reported corpus statistic and also the first evidence of whether the generator is good enough to be a teacher.

Critically, **filter at the profile level, not only the session level**: if session 2 of a profile fails, the profile's later sessions are broken as a temporal sequence. Either regenerate that session (up to 2 attempts) or drop the profile's tail from that point, and record which.

### 8.7 Annotation (`annotate.py`)

For every supporter turn, produce content-level `Analysis` and `Strategy` labels — free-form phrase summaries, deliberately **not** a fixed atomic taxonomy, because the free-form content level is the baseline paper's claimed advantage over ESConv and ExTES and this project inherits that design choice rather than quietly replacing it.

Steps: for each supporter turn, give the judge the dialogue up to and including that turn plus the profile snapshot, and ask for (a) an Analysis phrase describing what is going on psychologically for the user right now, (b) a Strategy phrase describing the move this turn makes, and (c) the ladder rung L0–L3 that this turn actually occupies. Write to `annotations.jsonl` with the judge id. Consistency pass: cluster Strategy phrases by embedding, inspect the 20 largest clusters by hand, and if the label vocabulary has collapsed to three phrases, raise judge temperature slightly or enrich the prompt with more diverse few-shot examples and re-run. A degenerate label vocabulary silently destroys the fine-tuning signal, and nothing downstream will report it.

### 8.8 Sizing and cost

Per session: ~10 supporter + ~10 user turns ≈ 20 generations ≈ 3–5k output tokens; AELS filtering ≈ 150 tokens; annotation ≈ 10 turns × ~80 tokens. With 900 surviving profiles × 3 sessions that is ≈ 2,700 sessions ≈ 10–14M output tokens for generation plus ≈ 3M for judging. At the Phase 0-measured rate, expect this in tens of GPU-hours; it is the single largest compute item in the project. Consequences to plan for: generate in shards by problem type, write after every record, and never regenerate what is already on disk.

### 8.9 Decision gate G1 — the SOP's declared fork

The SOP commits to a choice here rather than a claim: seed from EmpatheticDialogues if the yield holds, otherwise build profiles fully LLM-synthesized, and **report which route produced the released corpus**.

Run the gate on a 100-seed pilot, all stages through filtering. The seeded route wins if all three hold: ≥50% of seeds survive the sustainability screen; ≥60% of generated sessions pass AELS filtering; and a hand review of 20 profiles finds the need chains non-trivial and the terminal needs genuinely hidden. If any fails, switch `route` to `synthetic` (profiles sampled from the judge/generator with 5–8 hand-written seed examples in the prompt, as the baseline did), rerun the pilot, and record both pilots' numbers in `reports/route_decision.md`. Either way the field `route` on every profile carries the answer, and the report states it.

### 8.10 Release

`data/corpus/` is released with: the profiles, the sessions with annotations, the filter scores including rejections, all prompts, the config, the model ids and quantization settings, and a datasheet stating language (English), construction route, generator model, judge model, sizes, the resistance stratification, the gap distribution, and the known limitation that the corpus is machine-generated and its users are simulated.

---

## 9. Phase 2 — Fine-tuning the supporter (SOP III-F)

### 9.1 Target sequence construction (`build_sft.py`)

The point of the SOP's fine-tuning design is that intermediate reasoning is **learned, not prompted**: the Analysis and Strategy annotations go in the *target* sequence, ahead of the response.

For each supporter turn of each surviving session, build one training example:
- **Input:** the system prompt for the supporter role; the dialogue history verbatim; for sessions *k*>1, the elapsed-interval statement and the memory brief in the exact format that inference will use (Sec. 15). Training-time and inference-time context must be byte-identical in shape, or the adapter learns a format that never occurs at deployment.
- **Target:** a fixed-delimiter sequence of Analysis, then Strategy, then the response text. Loss is computed on target tokens only.
- **Two arms**, produced from the same corpus by a flag: `with_thoughts` (Analysis + Strategy + response) and `w/o thoughts` (response only). The second is the SOP's reproduction check against the published annotation ablation and must be built now, not retrofitted.

Also emit the **pacing-teacher dataset** stub here (input = history, target = ladder rung) so Phase 6's second stage has its data already keyed to the same examples.

Splits: by **profile**, never by turn or session, so no profile appears in both train and test. Reserve: train ~80%, validation ~10%, test ~10%. Carve the conformal calibration set (Sec. 10.4) out of a slice that is disjoint from both training and the final test profiles.

### 9.2 Training (`train.py`)

4-bit quantized base with low-rank adapters and gradient checkpointing, per SOP III-F, sized for the 12 GB budget: LoRA rank 16–32 on attention and MLP projections, sequence length 2048 (raise only if the memory brief pushes examples over it), micro-batch 1 with gradient accumulation to an effective batch of 16, cosine schedule, 2–3 epochs, bf16 compute, paged optimizer. Log train/validation loss; keep the best-validation checkpoint; store the adapter under `runs/<run_id>/adapter/`.

Two adapters come out of this phase: `sft_with_thoughts` and `sft_wo_thoughts`. Both are evaluated in Phase 7. The pacing distillation of Phase 6 is a **second stage over the same adapter stack** (SOP III-F), i.e. it continues from `sft_with_thoughts` rather than training a separate model — this matters for the VRAM budget and for the claim that pacing is a policy layer on the same supporter.

### 9.3 Check

The fine-tuned supporter must beat the un-tuned base of the same model on the corpus test split on Success Rate and AELS, using the Phase 7 harness. If it does not, the corpus, not the trainer, is the suspect: inspect label degeneracy (8.7) and chain triviality (8.2) before touching hyperparameters.

---

## 10. Phase 3 — Enhancement 2: multi-agent architecture, grounding contract, conformal critic gate

The SOP is explicit that decomposition is infrastructure and **the contract plus the gate are the contribution**. Build them in that order of care: the grounding validator first, then the agents, then the gate.

### 10.1 The grounding contract (`grounding.py`) — build this before the agents

Typed grounding adapted to this domain. Each claim an agent makes is assigned a category, and each category has a validation rule and a gate weight. The category list, with the SOP's decisive category first:

| Category | What it is | Validation | Weight |
|---|---|---|---|
| `denied_inference` | the draft re-proposes, or presupposes, an inference the user has already denied | cross-check draft content against memory nodes with status `disconfirmed`, and against user turns containing explicit denial of a prior supporter inference | **highest** — this is the textbook signature of premature interpretation and the category with no counterpart in factual grounding |
| `unsupported_state_attribution` | asserts an emotional state the user has not disclosed | requires ≥1 valid supporting span; fails if the span does not contain the attributed state | high |
| `fabricated_fact` | asserts a life fact absent from the history | span must exist verbatim | high |
| `overreaching_depth` | names the terminal need before the permitted ladder rung allows | compare draft rung (classified) against `allowed_rungs` | medium |
| `unsolicited_advice` | advice in the listening phase or before readiness | phase + readiness check | medium |
| `stale_reference` | refers to a past session detail whose memory node has decayed below threshold or was resolved | memory lookup | low |

Validators are mechanical wherever possible (span-in-text, status lookup, phase check) and judge-based only where they cannot be (rung classification of a draft). Mechanical checks never call a model, which is what makes the contract cheap enough to run on every turn.

Output per turn: a `grounding_report` listing violations with category, weight, and the offending text. This report is consumed twice — by the Critic as part of its nonconformity score, and by `memory.py` as the write permission check (R4).

### 10.2 The four agents (`agents.py`)

All four are the same loaded base model with different system prompts and different sampling temperatures. They run sequentially. The SOP's stated preference is concurrency, with sequential execution or offloading as the adaptation under hardware constraints — on a 12 GB card, sequential is the configuration, and the report should say so rather than implying concurrency was achieved.

1. **Analyzer** — reads the dialogue history verbatim plus the memory brief; emits the structured psychological state of 6.6: emotional state, implicit needs, resistance level, and `open_questions`. Every inference carries supporting quotes and a confidence in [0,1]. Claims failing span validation are zeroed.
2. **Strategist** — reads the history verbatim *and* the Analyzer output (R3: augment, never replace); emits a content-level strategy plan with a justification chain, a target chain node, and a requested rung. **Confidence rule:** if the Analyzer's confidence on the target need is below the configured threshold, `allowed_rungs` is restricted to exploratory moves (L0–L1) and `restricted_reason` is set. This is the mechanism by which uncertainty produces gentleness instead of confident overreach.
3. **Generator** — reads the history verbatim plus the strategy plan; drafts the supporter utterance at or below the permitted rung. Nothing else; it does not decide policy.
4. **Critic** — reads the history, the Analyzer state, the plan, and the draft; scores the draft on the E3 IP rubric, runs the grounding checks, and emits the nonconformity score and a decision.

Revision loop: on `revise`, the Critic's structured feedback is appended to the Generator's prompt and the draft is regenerated, up to `max_revisions`. If the last revision still fails the gate, **fall back to a plain reflective statement** (SOP III-B): a templated, low-risk reflection of the user's last turn produced at L0 with no new inference. Record every fallback; the fallback rate and the number of revisions per released turn are the "recovery-versus-cost curve" the SOP promises as deliverable 3.

### 10.3 The monolithic comparison cell (`baselines/listener_mono.py`)

One prompt, one forward pass, producing Analysis + Strategy + response in a single generation (the fine-tuned supporter's native format). This is the "monolithic listener" arm of the 2×2. It must use the same adapter and the same corpus-trained format, or the ablation confounds decomposition with training.

### 10.4 Conformal risk gate (`conformal.py`)

What the SOP asks for, precisely: instead of a tuned confidence threshold (whose calibration does not transfer across dialogue distributions), expose an intrusiveness error budget α, calibrate a nonconformity score derived from IP on a held-out split, and obtain a distribution-free bound on the fraction of released turns that exceed a tolerance. That is **risk control**, not just prediction-set coverage, and it should be implemented as such.

**Definitions.**
- *Tolerance* τ: a turn is a **violation** if the independent judge's IP, normalized to [0,1], exceeds τ (default 0.5).
- *Nonconformity score* s: a cheap, in-pipeline scalar available at decision time, computed by the Critic. Define s = w₁·(Critic's IP estimate) + w₂·(weighted grounding-violation mass), with the `denied_inference` category dominating the second term. s must be monotone in "how risky this draft looks", which is all the calibration needs.
- *Gate*: release the draft if s ≤ λ; otherwise revise, then fall back.

**Calibration procedure.**
1. Take `n_calibration_turns` supporter turns from held-out profiles, drawn from the same generator and profile distribution the system will be deployed on (exchangeability is the requirement — see below).
2. For each, record the pipeline's s and the judge's IP; label violation = IP_norm > τ.
3. For a grid of λ, compute the empirical risk R(λ) = fraction of turns that *would be released* at λ and are violations. R is monotone non-decreasing in λ: a looser gate releases more, including more bad ones.
4. For each λ, compute an upper confidence bound on R(λ) at confidence 1−δ using a distribution-free bound for a bounded (here, binary) loss — a Hoeffding-style bound, or the tighter Hoeffding–Bentkus form used in risk-control work. Use a standard conformal/risk-control library if one is available, and record which bound was used.
5. Select λ̂ = the **largest** λ whose upper bound satisfies UCB(R(λ)) ≤ α. Largest, because among safe thresholds the loosest one preserves the most utility (fewest needless revisions and fallbacks).
6. Write `calibration.json` (6.12), including the whole risk curve, not just λ̂ — the curve is what lets the report show the α-sweep.

**Deployment.** `agents.py` reads λ̂ and applies the gate. The α-sweep experiment re-selects λ̂ for α ∈ {0.05, 0.10, 0.20} and reports, per α: empirical violation rate on the in-distribution test split, fallback rate, revision count, Success Rate, and IP. This is the risk-coverage frontier and it is the central evidence for E2.

**Scope of the guarantee — state it in the report, not only here.** The bound holds only for held-out turns exchangeable with the calibration split: same generator, same profile distribution. The cross-corpus arms (ExTES profiles, ES-MemEval) are deliberately out of distribution; there the **empirical violation rate is reported as a measured quantity, never as a guarantee**, and the gap between the in-distribution and out-of-distribution violation rates is itself the finding — it measures how far the gate transfers. The code should enforce this honesty structurally: `calibration.json` carries a `distribution_id`, every evaluation record carries the distribution it ran on, and `report.py` refuses to print the word "guarantee" for any cell whose distribution differs from the calibration's.

### 10.5 The 2×2 ablation

Because decomposition is infrastructure and the gate is the claim, a single decomposition ablation cannot say which factor pays. Cross them:

| | ungated critic | gated critic (conformal λ̂) |
|---|---|---|
| **monolithic listener** | cell A — the fine-tuned baseline | cell B — gate applied to single-pass output |
| **decomposed agents** | cell C — four agents, critic advisory only | cell D — the full proposed system |

All four cells report Success Rate and Intrusiveness Penalty, on the same profiles, with the same n, paired. The informative outcome is not a joint win: if the gate carries the effect and decomposition alone contributes nothing, that is the result to report, and the SOP already commits to reporting it that way. `report.py` should compute the two main effects and the interaction explicitly so the answer is read off the table rather than argued.

---

## 11. Phase 4 — Enhancement 3: turn-level psychological metrics

### 11.1 Intrusiveness Penalty (`metrics.py`, `prompts/metric_ip.md`)

Unit: one **supporter** turn, scored in the context of the dialogue up to and including it. Constructs, from the clinical notion of premature interpretation and from privacy-management accounts of disclosure. Items (each 1–7, "not at all" to "very much"), which must be written out verbatim in the prompt file and in the report so any difference from other implementations stays visible:

1. The supporter gave advice before the seeker had finished explaining their situation.
2. The supporter asserted a feeling or motive the seeker had not expressed.
3. The supporter named a need or cause the seeker had not acknowledged.
4. The supporter pressed for information the seeker had signalled reluctance to give.
5. The supporter interpreted the seeker's situation more deeply than the seeker had invited.
6. The supporter disregarded a boundary the seeker had set.
7. The supporter's move was likely to feel exposing rather than supportive.

Aggregation: IP = mean of items, normalized to [0,1] as (mean − 1)/6. Higher is worse. Also report the per-item profile in the appendix; item 3 alone is the sharpest signal for premature naming of the terminal need and is worth showing separately.

Judge context and redactions: the judge sees the dialogue and the surface feeling; it does **not** see `resistance_level` or, for IP, the `terminal_need` (otherwise "named the need" becomes trivially checkable rather than judged from the interaction). Record every redaction in `context_redactions`.

### 11.2 Psychological Reactance Index (`metrics.py`, `prompts/metric_pri.md`)

Unit: the **user's next turn**, scored for resistance aroused by the preceding supporter move. Operationalized, per reference [8], as the conjunction of anger and negative cognition, which in dialogue surfaces as deflection, topic change, shortened responses, and increased defensiveness. Items (1–7):

*Anger component:* 1. irritation or annoyance toward the supporter; 2. hostility or sharpness in tone.
*Negative-cognition component:* 3. disagreement with or counter-arguing against the supporter's framing; 4. dismissal of the supporter's suggestion; 5. assertion of autonomy ("I know what I'm doing", "that's not it").
*Behavioral markers:* 6. deflection or topic change away from what the supporter raised; 7. withdrawal — a markedly shorter or more closed reply.

Aggregation: the reactance score r for a turn is the mean of items, normalized to [0,1]. Report the two components separately as well, since reactance as a conjunction means a turn scoring high on anger alone is a different phenomenon from one scoring high on counter-arguing alone.

Because PRI reads the actual next turn, it measures **realized** resistance — this is the distinction from observational frequency counts that score only the client side.

### 11.3 Counterfactual attribution (`counterfactual.py`)

To separate supporter-provoked resistance from pre-existing resistance, treat each supporter turn as an intervention.

For a sampled supporter turn at index t:
1. **Factual condition.** Keep the dialogue prefix through turn t as generated. Re-generate the user's reply N times with the simulator (seeds k = 0..N−1). Judge each reply's reactance → r₁ᵏ.
2. **Control condition.** Replace turn t with a **neutral reflective control**: a templated, content-matched restatement of the user's immediately preceding turn with no inference, no advice, no question beyond an open invitation (`prompts/control_reflective.md`, generated at temperature ≈0.2 and length-matched within a tolerance so the contrast is not confounded by verbosity). Re-generate the user's reply N times with the **same seeds** k → r₀ᵏ.
3. **PRI = r̄₁ − r̄₀**, reported with a 95% confidence interval from a paired bootstrap over the N seed-matched pairs, aggregated across scored turns. Positive PRI means the actual supporter move provoked resistance beyond what a neutral reflection would have.
4. The simulator used for replay must be in the same state it was at turn t: same profile snapshot, same session context, same accumulated history. Replay resets the simulator's history to the prefix; it never continues a contaminated conversation.

Cost: per scored turn, 2N user generations + 2N judge calls. With N=5 and 4 turns per dialogue over 150 dialogues, that is 6,000 user generations and 6,000 judge calls per arm — the second-largest compute item in the project, and the reason turns are sampled (stratified by position: one early, two middle, one late) rather than exhaustively scored. Rollout count and sampling density are config knobs so the study can be scaled down with the substitution recorded.

### 11.4 The validity problem and the three mandatory controls

The SOP states the threat plainly: Enhancement 1 scripts a resistance level into the profile, and the same model family plays the user and could score the turn — so a supporter could lower PRI by matching the script rather than by being less intrusive, and a judge could reward its own generation style. Three controls, all implemented, all reported:

1. **Resistance level withheld from the judge.** Enforced in the prompt builder, and recorded per score record in `context_redactions`. `report.py` asserts that no PRI record was produced with `resistance_level` in context; a violation is a failed run, not a footnote.
2. **Judge is a different open model** from both the fine-tuned supporter and the user simulator. Enforced in `judge.py` by comparing model ids at call time and refusing to score if they match. This separation is required *by PRI specifically*, because PRI scores the simulator's own turn.
3. **Human agreement on a stratified subsample.** `human_eval/sample.py` draws ~100 turns stratified by arm, resistance level, and judge IP decile. Both team members rate them against the same published item wording, blind to arm and to each other. `agreement.py` reports Cohen's κ (or weighted κ for ordinal items), Krippendorff's α across the three raters including the judge, and Spearman ρ between judge mean and human mean. **Every headline IP/PRI number in the report carries this agreement figure next to it** (SOP III-G).

Interpretation limit to state in the report: PRI measures resistance realized by a *simulated* user. Transfer to human reactance is explicitly out of scope.

### 11.5 How E3 feeds back

Two consumers, both already specified: the **Critic** uses the IP rubric as its scoring instrument and the derived nonconformity score for the gate (10.4), and the **readiness estimator** consumes realized resistance as a feature and as part of its training signal (13.2). Fig. 3 of the SOP shows exactly this: the turn-level metrics both gate the critic and update the readiness estimate. Implementation consequence: the IP item list lives in exactly one prompt file, imported by both the Critic and the judge, so the thing being gated and the thing being measured cannot drift apart.

---

## 12. Phase 5 — Enhancement 4: need-state trajectory memory

The unit of memory is the need chain itself, kept as a revisable belief — not a store of facts, not a general-purpose graph. The SOP is explicit about why: a strong recent baseline shows structurally simple event-level memory matching elaborate graph retrieval [9], so another triplet store would spend the budget on a component whose marginal value is contested. The hypothesis here is that the useful structure in emotional support is the inference chain, and that what memory must carry forward is **which links have been tested**.

### 12.1 State and operations (`memory.py`)

State: the node set of 6.9 per profile, persisted as JSONL with an append-only transition history for auditability (the current state is a fold over transitions, which makes every belief change explainable after the fact).

Operations, each with an exact rule:

- **propose(node, spans, confidence)** — create a `hypothesis` node. Rejected outright if no span validates (R4), or if an equivalent node is `disconfirmed` and `blocked_from_reproposal`. Equivalence is embedding similarity above a threshold plus a judge tie-break for borderline cases, and the check runs against the whole node set including resolved nodes.
- **confirm(node, span)** — promote `hypothesis` → `confirmed` when supporting spans reach `min_spans_to_confirm` and at least one is an explicit user affirmation (the user agreeing with an inference, not merely mentioning the topic).
- **disconfirm(node, span)** — set `disconfirmed`, record the contradicting span, set `blocked_from_reproposal`. Triggered by explicit user denial of the inference. This is the highest-value operation in the module because it is what feeds the `denied_inference` grounding category.
- **resolve(node)** — set `resolved` when the need was acted on and the user indicated it helped. A resolved node is not deleted; it remains retrievable as history and it makes "your last session helped with X" possible.
- **decay()** — run at session start. Confidence of `hypothesis` nodes decays with elapsed interval on the configured half-life; `confirmed` and `disconfirmed` do not decay, because a tested link stays tested. Below a floor, a hypothesis becomes dormant (not surfaced in briefs, still on record).
- **reinstate(node)** — a dormant or resolved concern that resurfaces after a long gap returns as `hypothesis` at `reinstate_confidence_factor` × its previous confidence, with `reinstated_count` incremented.
- **reopen(node)** — the only path out of `blocked_from_reproposal`, requiring `min_spans_to_reopen_disconfirmed` new contradicting-of-the-disconfirmation spans. Deliberately hard: re-proposing a rejected inference is the failure this module exists to prevent.
- **brief(question)** — retrieval as a question, not a lookup. Returns: the current best terminal-need hypothesis with its confidence; the chain path to it; the **outstanding evidence** — which links are untested and what would test them (sourced from the Analyzer's `open_questions`); the list of forbidden inferences (disconfirmed, with the user's own denial quoted); and resolved items. Ranking is driven by *what the system does not yet know*, not by similarity to the latest turn. This asymmetry versus dense retrieval is the design claim, and the retrieval-mode ablation of 12.3 is what tests it.

### 12.2 Wiring into the turn loop

At session start: `decay()`, then `brief()` is rendered into the Analyzer's and Generator's context as a bounded block (token budget from config), always **alongside** the raw history (R3). After the Critic releases a turn and the user replies, the Analyzer's next pass proposes/confirms/disconfirms based on that reply, and all writes pass through `grounding.py` first. Memory is therefore written once per turn, at one place, from validated claims only.

### 12.3 Baselines (`baselines/memory_*.py`)

All four run in the same harness with only the memory component swapped, so differences are attributable:

| Arm | What it does | What it isolates |
|---|---|---|
| `memory_none` | session-isolated; no cross-session state | the value of having any memory |
| `memory_summary` | flat running natural-language summary, regenerated each session | whether structure beats prose |
| `memory_dense` | embed every past turn, retrieve top-k by similarity to the current turn | whether need-state beats similarity retrieval — the central comparison |
| `memory_event` | event-level memory in the style of [9]: extract atomic events with timestamps, retrieve by recency+relevance | the strong simple baseline the SOP names |
| `memory_needstate` | this plan's module | — |

### 12.4 What is measured

- **Success Rate** (primary, per SOP: report SR "rather than merely recall") — fraction of profiles where the supporter identifies the terminal need, judged by `prompts/scale_success.md` against the ground-truth `terminal_need`, on the final session of each sequence.
- **Recall** of ground-truth profile transitions: of the advancement events applied in 8.5 (resolved / intensified / displaced), how many the memory state reflects correctly at session end.
- **Re-proposal rate**: count of supporter turns that re-propose a disconfirmed inference. This is the conflict-detection failure the SOP targets; the need-state arm should be near zero by construction and the baselines should not be, and if the baselines also score zero the probe was too easy and needs harder denial scripting.
- **Abstention correctness**: on profiles where the evidence genuinely does not identify a terminal need (build ~10% of profiles this way deliberately — ambiguous seeds with two plausible terminal needs), does the system abstain from naming one rather than guessing? Measured as the rate of confident naming when evidence is insufficient.
- **SR by Δt bucket**: short gap vs long gap, to show whether decay behaves as intended.
- **Cross-corpus**: repeat the arm table on ES-MemEval [6], so memory is tested on data this project did not generate. Note in the code and the report that ES-MemEval users are help-seekers without supporter annotation, so only the memory/SR portion transfers, not the proactivity. If ES-MemEval is not obtainable (a 2026 venue — availability must be verified early, in Phase 0), the fallback is to construct a held-out multi-session split from ExTES profiles plus this project's own generator with a **different** seed and a different generator temperature, and to label the substitution clearly as a weaker external-validity test rather than an independent corpus.

---

## 13. Phase 6 — Enhancement 5: adaptive disclosure pacing

The baseline paper established that probing too insistently degrades comfort, by steering a *global* questioning ratio and observing the decline. That identifies the effect and stops: a fixed constant gives a user who is opening up and a user who is deflecting identical pacing. This phase converts the finding into a per-user control mechanism.

### 13.1 The ladder (`pacing.py`)

A graded ladder over disclosure depth, operationalizing self-disclosure as movement through progressively more intimate layers, where the depth one may safely reach depends on what has already been reciprocated:

| Rung | Name | What the supporter may do | What it must not do |
|---|---|---|---|
| L0 | reflective acknowledgement | restate, validate, stay with the feeling | no inference, no question beyond an open invitation |
| L1 | open exploration | ask open, non-leading questions about what the user has already raised | no interpretation, no naming of unstated states |
| L2 | tentative inference | offer a hedged hypothesis about an intermediate need, invited and checkable ("I might be wrong, but it sounds like…") | no assertion, no terminal need |
| L3 | naming the terminal need | state the terminal need explicitly and work with it | only if readiness permits and evidence is confirmed |

The ladder is enforced at two points: the Strategist's `allowed_rungs`, and the Critic's `overreaching_depth` grounding check on the draft. Enforcing it only in the prompt does not work — models drift upward when they think they have figured the user out, which is exactly the failure being prevented.

### 13.2 Readiness as a value (`readiness.py`)

An ad-hoc decay over recent resistance would be neither interpretable nor comparable across users. Define readiness at turn t as the **expected final disclosure outcome given the conversation history** — a value in the reinforcement-learning sense, where a well-formed intermediate score equals the conditional expectation of the eventual return.

- **Return definition.** For a session, the return is the realized disclosure outcome: 1.0 if the user verbalized the terminal need, 0.66 if an intermediate need, 0.33 if only surface feeling, 0 if the user disengaged; with a small penalty term for turns whose PRI exceeded a threshold, so a session that extracted disclosure by provoking resistance does not count as a clean success. Store per session in 6.13 as `realized_outcome`.
- **Training the estimator.** From the corpus, every supporter-turn position gives a (history, return) pair — Monte Carlo regression onto the eventual return. Keep the estimator small and cheap: features = frozen sentence embeddings of the last k turns, plus interpretable scalars (mean PRI so far, count of deflections, user turn-length trend, current disclosure depth, rung history, session index, gap since last session, memory brief's confidence in the terminal need). Fit a gradient-boosted or ridge regressor on CPU. This deliberately does **not** require a second GPU model, and the feature list keeps the controller auditable, which the SOP explicitly wants.
- **Calibration check.** Bin predicted readiness into deciles and compare the mean predicted value against the mean realized outcome per bin — a reliability curve. A value function that is not calibrated is not a value function, and this check is the difference between "readiness" being a real quantity and being a vibe.
- **Cross-session persistence.** Readiness at the end of a session is stored per profile and carried into the next, decayed toward the population prior with a half-life proportional to the elapsed interval (`readiness_decay_half_life_days`). A user who resisted last time is approached more gently on return; a user who resisted a year ago is approached near the prior. Record `carried_from_prev_session` and `decayed_by` so the behavior is inspectable.
- **Mapping to the ladder.** The permitted rung is the highest rung whose `readiness_thresholds` entry is ≤ r̂, further capped by the Analyzer's confidence rule and by the memory brief (L3 requires the terminal-need node to be `confirmed`, never merely `hypothesis`). The ladder is thus the readout of a learned value, not the policy itself.

### 13.3 Learning from a privileged view

During corpus construction the ground-truth profile, terminal need, and scripted resistance are all available; the deployed supporter sees only the conversation. This asymmetry is the mechanism to exploit (transferred from reference [5], not invented here), with the difference that the objective is need identification subject to an intrusiveness bound rather than persuasion maximization, and that readiness persists across sessions.

1. **Privileged teacher** (`prompts/teacher_privileged.md`): a policy that sees the profile, terminal need, scripted resistance, triggers and blockers, and the full history, and chooses a rung per turn. Because it knows the hidden state, it can pace correctly — probing deep when the user is scripted as ready, holding at L0 when the user is scripted as guarded.
2. **Teacher rollouts:** run the teacher over training profiles, recording (history, permitted rung, chosen rung, resulting outcome, resulting IP/PRI). Keep only rollouts that satisfy the intrusiveness bound: sessions whose mean IP stays under τ. An unfiltered teacher teaches the student to probe, which is precisely the behavior being controlled.
3. **Distillation to the deployable policy:** a second training stage over the same adapter stack (SOP III-F) on inputs that contain **only conversation-visible information** — history, memory brief, readiness estimate — with the teacher's rung as the target. The student never sees the profile. Report the teacher-student gap in SR and IP; a large gap means the conversation genuinely underdetermines pacing, which is itself a finding worth stating.
4. **Consistency requirement:** the readiness estimator's features must be computable at deployment. Any feature that needs the profile is a leak; `readiness.py` should assert the feature set is a subset of the deployable view, so the leak cannot happen by accident.

### 13.4 Pacing baselines and the testable claim

The claim: adaptive pacing raises Success Rate **without** raising Intrusiveness Penalty, whereas aggressive probing raises both.

| Arm | Policy | Role |
|---|---|---|
| `fixed_L0`…`fixed_L3` | always the same rung | isolates adaptivity from any particular pacing level |
| `fixed_ratio_{0.2,0.4,0.6,0.8}` | fixed global questioning ratio, spanning the original study's range | direct analogue of the baseline paper's steering experiment |
| `pacing_greedy` | unconstrained information-seeking: identify the need as fast as possible | the aggressive-probing control that should raise both SR and IP |
| `pacing_adaptive` | readiness-driven ladder, this plan | — |

Report every arm as a point in the (SR, IP) plane and draw the frontier. The claim is supported only if the adaptive arm sits above the fixed-arm frontier, and the greedy arm shows the SR-and-IP co-increase. Any other pattern gets reported as it is.

---

## 14. Phase 7 — Evaluation protocol, baselines, ablations, statistics, reporting

### 14.1 Why the comparison is standalone

Neither the baseline corpus nor its construction pipeline is released, and the original work is Chinese while this one is English, so a number-for-number comparison against their published table is neither available nor meaningful. Results are therefore published as **standalone benchmarks**, and comparability rests on the **instruments**, not on shared code: AELS, the Comforting Responses Scale, and the RAC Scale are published psychological instruments, implemented here for English with open judge models, with the item wording reported verbatim so any difference from the original scoring stays visible rather than assumed away.

### 14.2 Evaluation sets

| Set | Source | What it tests |
|---|---|---|
| Own reticent profiles (held-out) | this project's E1 corpus, test split by profile | in-distribution performance; the conformal guarantee's valid domain |
| ExTES user profiles [3] | external, reactive | strongest reactive baseline's profiles; cross-distribution transfer |
| ES-MemEval [6] | external, multi-session, help-seeker users, no supporter annotation | memory and pacing on data this project did not generate |

For external sets, profiles must be adapted into the schema of 6.2 by a documented mapping pass, and any field this project's simulator needs but the source lacks (e.g. a scripted resistance level) is either generated and flagged as generated, or the arms that require it are marked not-applicable. Never silently fabricate a ground-truth terminal need for an external profile and then report Success Rate against it as if it were annotated.

### 14.3 Metrics reported for every arm

| Group | Metrics | Unit | Judge context |
|---|---|---|---|
| Primary | Success Rate (need identification) | dialogue | ground-truth terminal need shown to the judge |
| Turn-level (this project's) | IP, PRI (with CI) | turn | resistance level and (for IP) terminal need withheld |
| Psychological scales | AELS, Comforting Responses Scale, RAC Scale | dialogue | surface feeling + need shown, per the instruments' framing |
| Basic | fluency, coherence, empathy, informativeness | dialogue | — |
| Systems | fallback rate, revisions per turn, empirical violation rate vs α, tokens and wall-clock per dialogue, peak VRAM | run | — |
| Validity | judge–human agreement (κ, α, ρ) on the stratified subsample | per metric | — |

### 14.4 Arm matrix

`configs/arms.yaml` defines every arm as a set of component switches so the runner is one loop over arms rather than one script per experiment. The switches: supporter (base / `sft_wo_thoughts` / `sft_with_thoughts` / +distilled pacing), architecture (monolithic / decomposed), gate (off / conformal at α), memory (none / summary / dense / event / need-state), pacing (fixed rung / fixed ratio / greedy / adaptive), simulator (reactive / proactive), evaluation set.

The required arms, grouped by the deliverable they serve:

- **Deliverable 1** — reactive baselines (ExTES-style, and a plain instruct model) vs the fine-tuned supporter; `with_thoughts` vs `w/o thoughts` (the annotation-ablation reproduction).
- **Deliverable 2** — the 2×2 of 10.5, all four cells.
- **Deliverable 3** — the α-sweep, the fallback/recovery-cost curve, and the judge reliability study.
- **Deliverable 4** — the five memory arms of 12.3, on own sequences and on ES-MemEval.
- **Deliverable 5** — the pacing arms of 13.4.

Keep arms comparable: same profiles, same seeds, same n, and the same simulator and judge across every arm of a comparison. When an arm must differ in a second respect, say so in the table caption rather than hiding it.

### 14.5 Statistics

Paired designs wherever the same profile runs in both arms (which is nearly everywhere). Report the mean difference with a 95% paired-bootstrap CI over `bootstrap_resamples`; use the profile, not the turn, as the resampling unit for dialogue-level metrics, because turns within a dialogue are not independent. For the 2×2, report both main effects and the interaction. Correct across the families of comparisons within a deliverable (Holm), and say which comparisons were pre-registered in this plan versus which were exploratory — the arm list above is the pre-registration. Every cell carries n. No claim rests on a difference whose CI includes zero, no matter how much it was hoped for.

### 14.6 Reporting (`report.py`)

Reads only `runs/*/scores/*.jsonl` and the calibration and agreement artifacts, and writes `reports/results.md` plus the figures. Hard rules built into it: refuse a cell without n; refuse a cell whose judge model equals the supporter or simulator model; attach the agreement figure to every IP/PRI headline; label out-of-distribution violation rates as measured, not guaranteed; and emit `reports/substitutions.md` listing every reduced-scale substitution (smaller model, fewer rollouts, smaller n, fewer sessions) with the reason, per the SOP's Appendix commitment.

Figures worth producing: the (SR, IP) frontier with all pacing arms; the risk-coverage curve over α with the in- and out-of-distribution violation rates; the 2×2 interaction plot; SR by memory arm and by Δt bucket; the readiness reliability curve; the revision/fallback cost curve.

---

## 15. End-to-end wiring: the inference loop, turn by turn

This is the runtime path of Fig. 3 of the SOP, stated as an ordered sequence so the integration is unambiguous. Everything here is one process with one resident model plus CPU-side components.

**Session start.** Load profile-linked memory; run `decay()`; load readiness carried from the previous session and apply the elapsed-interval decay; render the memory brief and the elapsed-interval statement into the supporter context. If session index is 1, memory is empty and readiness is the population prior.

**Per turn:**
1. **Analyzer** ← raw history (verbatim) + memory brief. → structured state with spans and confidences; `open_questions`.
2. **`grounding.py`** validates every Analyzer claim's spans; failed claims are zeroed and cannot reach memory.
3. **`readiness.py`** updates r̂ from the features of the current state (including PRI-derived resistance signals from previous turns). → permitted rung, capped by Analyzer confidence and by memory status for L3.
4. **Strategist** ← raw history + Analyzer state + permitted rungs + memory brief. → plan, justification chain, target node, requested rung (restricted to exploratory if confidence is low).
5. **Generator** ← raw history + plan. → draft utterance.
6. **Critic** ← raw history + state + plan + draft. → IP estimate, grounding violations, nonconformity s, decision against λ̂.
7. If `revise`: feedback appended, back to step 5, up to `max_revisions`. If still failing: **fallback** to the templated reflective statement at L0. Record which path was taken.
8. Emit the turn. **User replies** (simulator in evaluation; a real person would be here in deployment, which this project does not do).
9. **Memory write:** the Analyzer's next pass, reading the new user reply, issues propose/confirm/disconfirm/resolve calls; each passes `grounding.py` first. Disconfirmations set the re-proposal block that step 6's highest-weight check will enforce from now on.
10. Append everything — the four agent artifacts, the decision path, the memory transitions, the readiness record — to the run directory. This is the intermediate inspection point whose absence is Limitation 1.

**Session end.** Score the session's `realized_outcome`; persist readiness; persist memory; write the session record. Scoring by the judge happens later, as a separate pass, after the generator is unloaded (R2).

---

## 16. Schedule, bottlenecks, compute budget

Two bottlenecks dominate and both are known in advance: **corpus generation** (Phase 1, tens of GPU-hours, Sec. 8.8) and **PRI counterfactual replay** (Phase 4, 2N generations + 2N judge calls per scored turn, Sec. 11.3). Everything else is small by comparison. Plan the calendar around those two and keep their knobs (`n_profiles_target`, `sessions_per_profile`, `n_rollouts`, `turns_sampled_per_dialogue`, `n_dialogues_per_arm`) as the levers when the calendar slips.

| Order | Work | Depends on | Output that unblocks others |
|---|---|---|---|
| 1 | Phase 0 smoke checks; dataset availability verification (especially ES-MemEval) | — | measured throughput and VRAM; the schedule becomes real |
| 2 | Schemas, `common.py`, `llm.py`, `judge.py`, `grounding.py` | 1 | every later stage |
| 3 | Phase 1 pilot (100 seeds) + gate G1 | 2 | the route decision |
| 4 | Phase 1 at scale, sharded by problem type | 3 | the corpus — critical path for everything |
| 5 | Phase 2 SFT, both arms | 4 | the supporter under test |
| 6 | Phase 3 agents + grounding + conformal calibration | 5 | the gated system and the 2×2 |
| 7 | Phase 4 metrics + counterfactual + human agreement study | 5 (needs dialogues to score) | IP/PRI for every arm; the gate's nonconformity target |
| 8 | Phase 5 memory + baselines | 4, 6 | memory results |
| 9 | Phase 6 readiness + teacher + distillation + pacing arms | 4, 5, 7 | pacing results |
| 10 | Phase 7 evaluation sweep + report | all | the deliverables |

Parallelizable by person, following the SOP's division: the corpus pipeline and fine-tuning are joint; temporal dynamics (memory, pacing) is one owner; inference architecture and measurement (agents, gate, metrics) is the other. The two owners share `grounding.py` and the schemas, so those are written jointly and frozen early — a mid-project schema change is the most expensive avoidable event in this plan.

Cheap-first rule for every phase: run the pilot at 1/20 scale, look at ten examples by hand, then scale. Hand-reading ten dialogues catches degenerate label vocabularies, simulators that instantly confess their terminal need, and judges that give everything a 6 — three failures that no aggregate metric surfaces until it is too late.

---

## 17. Risk register with concrete fallbacks

| Risk | Signal that it is happening | Fallback (and what to report) |
|---|---|---|
| ED situations too short to sustain a 3-step chain | sustainability screen pass rate < 50% at gate G1 | switch to the fully LLM-synthesized route; report both pilots' numbers and the `route` field (SOP III-A commits to exactly this) |
| Simulator discloses the terminal need in turn 1 or 2 | filter's leakage check fires often; SR near 1.0 for every arm | strengthen resistance instructions, raise the initial resistance level, add an explicit early-turn withholding rule; if SR stays saturated, the corpus is too easy — regenerate with deeper chains |
| Degenerate annotation vocabulary | 3 phrases cover >80% of Strategy labels | diversify few-shot examples, raise judge temperature slightly, re-annotate; the fine-tuning signal depends on this |
| Judge gives everything 5–6 (ceiling effect) | low variance in AELS/CRS/RAC | add explicit anchor descriptions per item, force per-item justification before the score, or switch judge model; report the change |
| Critic's IP estimate uncorrelated with the judge's IP | calibration risk curve is flat; λ̂ meaningless | the gate cannot work on that score — add grounding-violation mass weight, or make the Critic score the specific items that correlate; report the correlation as part of the calibration artifact |
| Conformal bound vacuous (λ̂ at the strictest end, everything falls back) | fallback rate > 50% at α=0.1 | report it as a finding (the pipeline cannot meet that budget), and show the α-sweep so the achievable budget is visible; do not quietly raise α and present it as the default |
| 12 GB insufficient for judge + required context | Phase 0 residency check fails | smaller judge from a different family, shorter judge context via transcript truncation with the truncation rule reported, or reduced-scale run; record in `substitutions.md` |
| ES-MemEval unobtainable | verified in Phase 0 | held-out generator-shifted split as in 12.4, clearly labelled a weaker external test |
| PRI replay too expensive | Phase 4 projected wall-clock exceeds the schedule | reduce N to 3 and turns per dialogue to 2, widen the CI honestly, record the substitution; do not drop the control condition — without it PRI has no attribution and the metric loses its point |
| Teacher-student distillation collapses to one rung | rung entropy near zero in student rollouts | rebalance the teacher-rollout dataset across rungs, add the readiness feature explicitly to the student input, or keep the ladder as a hard rule-based cap over a readiness regressor and report that the learned policy did not beat the rule |
| Memory brief crowds out the raw history at 2048 tokens | truncation warnings; degraded SR with memory on | shrink the brief (top-1 hypothesis + forbidden list only), raise sequence length if VRAM allows; never solve it by dropping raw turns (R3) |
| Both team members rate the human subsample the same way because they wrote the rubric | κ suspiciously high, ρ with judge also high | keep raters blind to arm and to each other, rate in randomized order, and state the limitation plainly in the report |

---

## 18. Acceptance criteria and self-checks per phase

Each phase leaves one runnable check behind, in `tests/`, that fails if the logic breaks. These are assertions on small fixtures, not a test framework build-out.

| Phase | Check | Passes when |
|---|---|---|
| 0 | thin end-to-end slice | two profiles go from seed to scored turn with no manual intervention |
| 1 | schema + filter fixtures | a hand-written good session passes the filter, three hand-written bad ones (repetition, role break, profile leakage) are each rejected for the right reason |
| 1 | temporal integrity | for every multi-session profile: gaps positive, sessions ordered, node ids stable, advancement transitions recorded |
| 2 | SFT format round-trip | a built training example parses back into exactly its Analysis / Strategy / response parts; the `w/o thoughts` arm contains no annotation tokens |
| 3 | grounding validator | fabricated spans rejected, off-by-N offsets rejected, exact spans accepted; a draft re-proposing a disconfirmed node is flagged `denied_inference` |
| 3 | conformal monotonicity and selection | the risk curve is non-decreasing in λ; λ̂ is the largest λ with UCB ≤ α; a synthetic calibration set with known violation rate recovers the expected λ̂ |
| 4 | judge separation | scoring refuses to run when the judge model id equals the supporter's or the simulator's; every PRI record's `context_redactions` includes `resistance_level` |
| 4 | counterfactual pairing | factual and control rollouts share seeds pairwise; the control turn contains no question beyond an open invitation and no inference |
| 5 | memory transitions | disconfirm blocks re-proposal; reopen requires the configured evidence; decay only touches hypotheses; the fold over the transition log reproduces the current state exactly |
| 6 | readiness legality and calibration | no feature requires the profile; the reliability curve is monotone and within tolerance of the diagonal |
| 7 | report guards | a cell without n, or with judge == supporter, or an OOD cell labelled "guarantee", fails the build |

Phase-level acceptance, tied to the SOP's stated targets: Phase 1 succeeds when the direction of the published annotation ablation reproduces (`with_thoughts` > `w/o thoughts` on SR) and the fine-tuned supporter beats reactive baselines on SR. Phase 3 succeeds when all four 2×2 cells are reported with paired CIs, whatever the pattern. Phase 4 succeeds when IP and PRI exist for every arm with CIs and an agreement figure attached. Phase 5 succeeds when all five memory arms are reported on both an internal and an external set. Phase 6 succeeds when the (SR, IP) frontier is drawn with the adaptive arm and every baseline arm on it.

---

## 19. Ethics, safety, and honesty constraints

- **No human subjects.** All users are simulated. The only human participation is the two authors rating a subsample of machine-generated turns. Nothing in this project is deployed to a person in distress, and the report says so.
- **Crisis content screen.** Generation can wander into self-harm or abuse content. `filter.py` runs a crisis screen; flagged sessions are quarantined to `data/quarantine/` for review rather than silently dropped (so the rate is known) and excluded from training. If the released corpus retains any such content, the datasheet says so and warns downstream users.
- **No safety theatre in the gate's framing.** The conformal gate bounds a *judge-measured intrusiveness* risk on an in-distribution simulated population. It is not a clinical safety guarantee. The report must state this in the same paragraph as the bound, every time.
- **Simulated-user limit.** IP and PRI measure what a simulated user's turns look like to a judge model. Transfer to human reactance is explicitly out of scope (SOP III-C).
- **Provenance and licensing.** Record dataset licences for EmpatheticDialogues, ESConv, ExTES, ES-MemEval, and model licences for every checkpoint, in `data/raw/PROVENANCE.md`. Release only what those licences permit.
- **Open weights only.** No proprietary API at any stage, in any fallback. There should be no code path capable of one.
- **Attribution.** The upstream prompts and scorers were read; the instruments are re-implemented from their source psychological literature and the item wording is published, so no scoring implementation is inherited silently.

---

## 20. Release artifacts

Mapped to the SOP's five expected outcomes:

1. **Pipeline and corpus** — all of `src/` for Phase 1, all prompts, the profiles, the linked multi-session corpus with per-turn Analysis and Strategy annotations, the filter scores including rejections, the datasheet, and `reports/route_decision.md` naming which construction route produced the release.
2. **Refactored system and ablation** — the multi-agent implementation, the grounding contract with its category table, the calibration artifact, and the 2×2 table with main effects and interaction.
3. **Validated metrics** — the IP and PRI prompt files with verbatim item wording, per-arm scores with CIs, the judge reliability study (κ, α, ρ) on the stratified subsample, and the critic's recovery-versus-cost curve.
4. **Persistent memory** — the need-state memory module, the four baseline memory arms, and results on both this project's sequences and an external set, reporting Success Rate as well as recall.
5. **Adaptive pacing** — the ladder, the readiness estimator with its reliability curve, the privileged teacher, the distilled deployable policy, and the (SR, IP) frontier against fixed-pacing and unconstrained-probing arms.

Plus, per the SOP Appendix: all prompts, model configurations, quantization settings, hyperparameters, and `reports/substitutions.md` documenting every reduced-scale substitution with its rationale.
