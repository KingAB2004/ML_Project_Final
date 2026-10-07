# src/ by category

Files stay flat in `src/` (every module imports the others by bare name, and the lab and the laptop queues call
`src/<file>.py` directly). This page groups them. Status as of 7 Oct 2026.

## 1. Shared infrastructure

| File | What it does | Status |
|---|---|---|
| `common.py` | config, ids, seeds, JSONL IO, spans, transcripts | used everywhere |
| `llm.py` | the only module that loads weights or samples tokens (Ollama, transformers, echo backends; adapter on/off; cache) | used everywhere |

## 2. Dataset creation

| File | What it does | Status |
|---|---|---|
| `seeds.py` | seed situations (EmpatheticDialogues), ESConv problem-type taxonomy | ran: 1,000 seeds |
| `profiles.py` | seed to profile (feeling, 4 past events, hidden 3-level need chain, persona, triggers / blockers), validators, splits | ran: 1,000 profiles (720 train / 100 val / 100 test / 80 calibration) |
| `dialogue.py` | one session: supporter and simulated seeker turn by turn | ran: every corpus session |
| `sessions.py` | links 2-4 sessions per person with 1-56 day gaps and advances the person (resolved / intensified / displaced); `--check` integrity; `--memory` (Option C) and `--replay` | ran: 2,369 sessions. `--memory` not run; `--replay` used by `futurememory.py` |
| `filter.py` | structure, need leaks, safety, active-listening filters | ran: 2,369 kept |
| `annotate.py` | analysis + strategy + ladder rung for every supporter turn | ran: 21,429 turns |
| `convert_corpus.py` | ExTES / ESConv into our session and SFT format | used for the data-quality samples (Table 2 / 8) |
| `external.py` | external evaluation sets mapped to our profile schema | written; cross-dataset arms not run |

## 3. Fine-tuning

| File | What it does | Status |
|---|---|---|
| `build_sft.py` | sessions to SFT examples (input: gap line, memory brief, conversation so far; target: thoughts + reply), per split, both arms | ran: 18,821 train / 2,608 val per arm |
| `train.py` | QLoRA (4-bit NF4, r 32, alpha 64), best checkpoint by validation loss | ran: Qwen2.5 0.5B / 3B / 7B, both arms. Qwen3-4B training now |
| `futurememory.py` | Option B memory-aware SFT data: replay the corpus through the need-state memory, re-annotate sessions 2+ with the brief, build SFT | written and tested 7 Oct; not run (after 9 Oct) |

## 4. Agent pipeline and gate

| File | What it does | Status |
|---|---|---|
| `agents.py` | Analyzer, Strategist, Responder, Critic; revision loop; memory writes (`update_memory`, `observe`) | used in every decomposed arm; tree-memory writes and `observe` added 7 Oct |
| `grounding.py` | typed grounding contract: quoted spans, denied inferences, stale references | used by the Critic in every agent run |
| `pacing.py` | disclosure ladder L0-L3, rung caps | used in every agent run |
| `conformal.py` | conformal risk control for the Critic gate | used by `calibrate.py` and gated arms |
| `calibrate.py` | fits the gate threshold on the calibration profiles | ran: alpha 0.05 / 0.10 / 0.20 (release 13 / 73 / 100 %) |

## 5. Need-state memory

| File | What it does | Status |
|---|---|---|
| `memory.py` | need-state graph: feeling roots, linked needs, propose / confirm / deny / resolve / reinstate, rival lowering, fade with reinforced half-life, pruning, bounded brief | flat version ran in `mem_needstate` (1000). Tree + fade / prune: lab trial on p000394, fine-tuned 7B memory run (7 Oct), v2 memory arms queued |
| `baselines/memory_none.py` | no cross-session state | queued (v2 memory arms) |
| `baselines/memory_summary.py` | running prose summary | queued (v2 memory arms) |
| `baselines/memory_dense.py` | dense retrieval over past turns | queued (v2 memory arms) |
| `baselines/memory_event.py` | event memory with recency decay | queued (v2 memory arms) |

## 6. Baseline supporter

| File | What it does | Status |
|---|---|---|
| `baselines/listener_mono.py` | monolithic supporter: one call writes thoughts + reply; optional memory writer | ran: base_instruct, reactive, cells A / B, all fine-tuned test runs, fine-tuned 7B multi-session (running); memory writer added 7 Oct |

## 7. Evaluation and metrics

| File | What it does | Status |
|---|---|---|
| `evaluate.py` | arm runner: one arm = supporter, architecture, gate, memory, simulator switches (`configs/arms.yaml`) | ran every arm above |
| `judge.py` | scale rendering, score parsing, judge-separation guard | used by `metrics.py` |
| `metrics.py` | Success, AELS, CRS (Aff / Neg), RAC (Sup / Man), Basic, IP, PRI | ran for every finished arm; data-quality Table 2 queued |
| `counterfactual.py` | counterfactual PRI (reactance against a neutral reflection) | ran for the test runs and baselines |
| `report.py` | tables, bootstrap CIs, guards | ran for the baselines; cells C / D report pending |
