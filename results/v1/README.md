# Phase A — v1 results (partial), problems found, and fixes

This folder is a frozen snapshot of the **first baseline run (v1)** of the COCCON_NEW pipeline, taken on
2 October 2026 from the lab server (`cse@10.50.28.201`, RTX 3060 12 GB, Ollama). v1 was stopped
deliberately before scoring finished, because quality checks on its outputs exposed several bugs that made
some of its numbers unreliable. Everything here is kept as evidence; **the numbers to report come from v2**,
which re-runs Phase A with the fixes described below.

---

## 1. What v1 is

Phase A evaluates **untuned** supporters (Qwen2.5-7B-Instruct, no fine-tuning yet) in six configurations
("arms") against a simulated seeker, then scores each dialogue with an independent judge model.

| Item | Value |
|---|---|
| Supporter / simulated seeker | `qwen2.5:7b-instruct` (Ollama, q4) |
| Judge | `mistral-nemo:12b-instruct-2407-q4_K_M` (Ollama) — a different model family from the supporter |
| Seeds | 400 situations from EmpatheticDialogues (800 screened, 87.4 % passed the sustainability screen) |
| Profiles | 400 hidden-need profiles, 0 rejected; resistance low 100 / medium 200 / high 100 |
| Splits | train 288, val 40, **test 40**, calibration 32 (split by profile, no overlap) |
| Evaluation | 30 test profiles per arm; arms with memory run 2–4 linked sessions per profile |
| Reduced settings | 1 judge sample per item, 2 counterfactual (PRI) rollouts, 4 parallel requests |

### The six arms

| Arm | Supporter design | Gate | Seeker simulator | Sessions |
|---|---|---|---|---|
| `base_instruct` | single model, one prompt | off | proactive | 30 |
| `reactive_baseline` | single model, one prompt | off | **reactive** (help-seeking) | 30 |
| `cellA_mono_ungated` | single model (monolithic listener) + need-state memory | off | proactive | 93 |
| `cellB_mono_gated` | monolithic listener + memory | **conformal critic gate** | proactive | 93 |
| `cellC_dec_ungated` | **multi-agent** (Analyzer → Strategist → Generator → Critic) + memory | off | proactive | 93 |
| `cellD_dec_gated` | multi-agent + memory | **conformal critic gate** | proactive | 93 |

`cellA`–`cellD` form the 2×2 of *decomposition × gate* from the SOP. In v1 they all ran on the **base**
model, because no fine-tuned adapter exists yet (the log warns about this; it is expected for Phase A).

---

## 2. What v1 achieved

* The whole Phase A pipeline ran end to end on open-weight models only: seeds → profiles → 432 evaluation
  dialogues across 6 arms (average 20–21 turns each) → conformal calibration → judge scoring for 3 arms.
* Parallel requests (4 at a time through Ollama) were added and measured at **~2.3×** the throughput of the
  original one-request-at-a-time code, with byte-identical outputs.
* Profile generation was fixed from 81 / 400 usable to **400 / 400**.
* Two judge-parsing bugs were found and fixed; Success Rate for `base_instruct` went from 16 / 30 parsed to
  **30 / 30** (re-parsed from cached judge replies, no new model calls).

### 2.1 Scores (normalised 0–1; higher is better except IP and PRI)

| Arm | Success | AELS | Basic | CRS | RAC | IP ↓ | PRI (observed) ↓ | PRI (counterfactual) |
|---|---|---|---|---|---|---|---|---|
| `base_instruct` | 0.406 (30) | 0.935 | 0.813 | 0.594 | 0.856 (27) | 0.042 (120 turns) | 0.111 | −0.001 [−0.012, 0.010] |
| `reactive_baseline` | 0.422 (30) | 0.929 | 0.817 | 0.592 | 0.853 (29) | 0.045 (120 turns) | 0.123 | −0.003 [−0.017, 0.010] |
| `cellA_mono_ungated` | 0.390 (92) | 0.938 | 0.815 | 0.596 | 0.849 (82) | 0.065 (372 turns) | 0.124 | not run |
| `cellB_mono_gated` | not scored | | | | | | | |
| `cellC_dec_ungated` | not scored | | | | | | | |
| `cellD_dec_gated` | not scored | | | | | | | |

Numbers in brackets are sample sizes where items failed to parse, or the 95 % CI for counterfactual PRI.
`cellB`–`cellD` have dialogues but were not scored before v1 was stopped.

* **Success** – did the supporter put the seeker's hidden underlying need into words (judge sees the ground truth).
* **AELS / CRS / RAC / Basic** – published instruments for active listening, comforting responses,
  conversational competence and six basic qualities.
* **IP** – Intrusiveness Penalty (our metric): did a turn overstep what the seeker was ready for.
* **PRI** – Psychological Reactance Index (our metric). The *counterfactual* version replays the seeker's
  reply to the real turn and to a neutral control turn; a value near 0 means the supporter's turns provoked
  no more reactance than a bland reflection would.

### 2.2 Conformal gate calibration

| alpha | tau | lambda_hat | n (turns) | violations in calibration set |
|---|---|---|---|---|
| 0.10 | 0.50 | **1.000** | 98 of 300 intended | 0 |

`lambda_hat = 1.0` means the gate's threshold releases every turn. Only 98 turns were usable (see bug B7),
and the judge rated none of them intrusive (max IP 0.48 < tau 0.50).

### 2.3 What the v1 numbers do and do not show

* The untuned supporter tends to **reassure and advise** ("You've got this!") rather than explore the hidden
  need — visible in the dialogues and consistent with Success ≈ 0.4.
* Proactive vs reactive and single-model vs memory-augmented differ by **less than the noise** at n = 30.
  This is partly real and partly the judge (see problem P2).
* The 2×2 cannot be read from v1: the gated cells and the critic inside them were broken (bugs B11–B14).

---

## 3. Problems found in v1

Found by inspecting the 432 dialogues and the agent artifacts, not just the summary scores.

| # | Problem | Evidence in v1 | Effect |
|---|---|---|---|
| P1 | **Simulated seeker spoke as the supporter** | ~1 seeker turn in 10 copied an earlier line verbatim, usually the supporter's ("user: You're doing great! … Pomodoro Technique") | Noisy, unrealistic dialogues in every arm |
| P2 | **Judge barely separates arms** | AELS ≈ 0.93–0.94 (ceiling), IP ≈ 0.04–0.07 (floor), counterfactual PRI CIs span 0 | Real differences are hard to detect. Not fixable without a stronger judge, which is not available on this server — reported as a limitation |
| P3 | **Critic's IP prediction was always 0** | Mean critic `ip_pred` = 0.0 over ~2,500 turns; item scores were literally the template's zeros | The gate's score was driven only by rule-based violations |
| P4 | **Gate rewrote 72 % of `cellB`** | 326 fallback + 274 revised of 834 turns; **all 326** "denied inference" flags came from the critic model, none from a recorded denial | `cellB` measured the fallback text, not the gate |
| P5 | **Fallback parroted the seeker** | e.g. "It sounds like it's been okay, just trying to keep up with everything. Thanks fo…" | Awkward, unsupportive turns in gated arms |
| P6 | **Analyzer evidence rejected** | 1,870 of 5,303 evidence quotes invalid; 465 cited a supporter turn number for the seeker's words | Analyzer confidence 0.0 on 661 of 849 turns, so memory and depth decisions were starved |
| P7 | **Reply length confound** | single-model supporters ~85 words per turn, multi-agent ~21 (its prompt said "1 to 3 sentences", the other had no limit) | Length differences confound decomposition with verbosity in the 2×2 |
| P8 | **Repeated supporter lines** | 60 verbatim repeats in `cellC` ("You've got this, and I'm here to help.") | Repetitive dialogues |
| P9 | **Occasional Chinese output** | 1–3 sessions per arm | Language drift from Qwen |
| P11 | **Supporter's analysis leaked into the dialogue** | 7–13 % of single-model supporter turns contained "Analysis: … Strategy: …" text (found in the v2 pre-check, then measured in v1) | The seeker and the judge saw the supporter's private reasoning, which inflates or distorts scores for those arms |
| P10 | **Gate never activates by threshold** | lambda_hat = 1.0 at every alpha | With the judge's low IP ratings and tau = 0.5, nothing counts as a violation. Kept as a finding; tau unchanged |

---

## 4. Bugs fixed (whole project so far)

All fixes are applied both locally and on the server; originals are backed up on the server under
`~/coccon_prefix_backup/`. The component test suite went from 61 / 69 passing to **79 / 79** (new tests
added for each fix).

A short real-model check of the v2 code (gated single-model arm, 2 profiles, 64 turns) showed: critic
parse failures 0, critic IP predictions varied 0.00–0.38 (v1: always 0), no `denied_inference` flags,
0 fallbacks, 0 repeated lines, 0 Chinese turns — and exposed B28 below.

### 4.1 Before and during v1

| # | Bug | Root cause | Fix (file) |
|---|---|---|---|
| B1 | 8 failing component tests | Mix of real bugs (below) and stale tests (6 basic metrics not 4, fixture shorter than `min_turns`, `ollama` not an accepted backend) | Code fixed where wrong, tests updated where stale (`tests/test_all.py`) |
| B2 | Judge scores on one line lost | Parser only read items at the start of a line | Second pass fills missing items anywhere (`src/judge.py`) |
| B3 | Hidden-need leakage never detected | Jaccard similarity dilutes a recited need inside a longer turn | Containment of the need's words (`src/filter.py`) |
| B4 | Smoke test silently produced 0 seeds | Stub backend answered "Acknowledged." to the seed screen; every later stage reported OK on 0 rows | Stub answers the screen; `seeds.py` now fails loudly on 0 seeds (`src/llm.py`, `src/seeds.py`) |
| B5 | ExTES conversion produced 0 dialogues | Speaker key `AI` never matched after lower-casing | Exact `ai` match (`src/convert_corpus.py`) |
| B6 | Human-rating sample could come from one arm | Sorted bucket order exhausted the first arm | Round-robin across arms (`human_eval/sample.py`) |
| B7 | 319 / 400 profiles rejected | Judge role capped at 256 output tokens; a profile JSON needs 400–600, so replies were cut mid-JSON | `max_tokens` 1024 for profiles and session advancement, 512 for annotation |
| B8 | Evaluation crashed (`'str' object has no attribute 'get'`) | Analyzer sometimes returned an evidence quote as a plain string | Bare or malformed spans are validated or rejected, never crash (`src/grounding.py`) |
| B9 | Test profiles would leak into fine-tuning data | `build_sft.py` re-shuffled whichever profiles survived filtering instead of using the saved split | Uses the split files `profiles.py` wrote; unknown profiles are skipped, not put in train (`src/build_sft.py`) |
| B10 | Two-thirds of judge replies discarded | Judge often scored only some questions; with 1 sample there was no second chance (calibration used 98 of 300 turns) | One explicit retry for every item; partial answers still rejected, never filled in (`src/judge.py`) |
| B11 | Success Rate failed for 14 / 30 dialogues | Judge wrote `Question 1: Score - 4`; parser did not expect the labels | Parser accepts `Question`/`Item` and `Score` labels (`src/judge.py`) |

### 4.2 Root causes found after v1 (fixed for v2)

| # | Problem | Root cause | Fix (file) |
|---|---|---|---|
| B12 | P1 seeker copied the supporter | The simulator was never told *which* speaker it is in the flat `supporter:` / `seeker:` transcript | Per-turn request names it the seeker and asks for English; a turn that repeats an earlier line is regenerated once (`seeker_prompt` in `src/dialogue.py`, `fresh_turn` in `src/common.py`; used in both corpus generation and evaluation) |
| B13 | P3 critic IP always 0 | Critic prompt showed `"item_scores": {"1": 0, …}` — off the 1–7 scale — and the 7B model copied it | Prompt describes the format without example values; scores must be 1–7 for every item or the critic is asked once more (`prompts/agent_critic.md`, `valid_item_scores` in `src/agents.py`) |
| B14 | P4 denied-inference flood | Prompt listed `"denied_inference|…"` as the example violation; the model echoed it, and it is a hard veto | Model-reported `denied_inference` is accepted only when a denial was actually recorded in memory (`src/agents.py`) |
| B15 | Monolithic critic used the wrong settings | `cellA`/`cellB` built the critic on the supporter's model handle (temperature 0.7) instead of the critic role (0.2) | Critic uses its own role (`src/baselines/listener_mono.py`) |
| B16 | P5 fallback parroted | Fallback pasted the whole last seeker turn into a template | Reflects the first clause in the second person ("you were exhausted"), max 18 words (`src/agents.py`) |
| B17 | P6 evidence rejected | Analyzer template showed `turn_index: 0, quote: ""`; the model also cited wrong turn numbers | Template without example values and "cite only seeker lines"; a verbatim seeker quote under the wrong turn number is relocated to the seeker turn that contains it (supporter words still never count) (`prompts/agent_analyzer.md`, `src/grounding.py`) |
| B18 | Same template-copy hazard elsewhere | `"requested_rung": "L0|L1|L2|L3"` (an invalid value was silently mapped to the *deepest* allowed rung) and `"user_disclosure_depth": 0` | Value descriptions instead of values; a copied option list is not accepted as a rung (`prompts/agent_strategist.md`, `prompts/annotate_turn.md`, `src/annotate.py`) |
| B19 | P7 length confound | Only the multi-agent generator had a length limit | One spec for every supporter: 1–3 sentences of plain spoken English (`SUPPORTER_SYSTEM` in `src/build_sft.py`, prompts) |
| B20 | P8 repeated supporter lines | Nothing stopped a stock line being reused | Same one-retry repeat guard for every supporter arm (`src/agents.py`, `src/baselines/listener_mono.py`) |
| B21 | P9 Chinese output | Prompts never asked for English | "English only" in seeker and supporter prompts |
| B28 | Supporter's private analysis shown to the seeker | The single-model system prompt asked for "your analysis and strategy, then the reply" without saying how to mark them; the code only extracts a reply inside `<response>` tags, so untagged "Analysis: … Reply: …" went into the dialogue verbatim (v1: 7–13 % of turns in `base_instruct`, `reactive_baseline`, `cellA`, `cellB`; 0 % in the multi-agent arm) | System prompt gives the exact `<analysis>/<strategy>/<response>` format (the same one the fine-tuning targets use); the parser also strips untagged "Analysis:/Strategy:" sections and keeps only the reply (`src/build_sft.py`) |

### 4.3 Problems found in Phases B and C before they ran (fixed)

| # | Bug | Root cause | Fix (file) |
|---|---|---|---|
| B22 | Adapter would be trained on a format it never sees | `train.py` built prompts as `<|system|>…<|user|>…`, but inference uses Qwen's real chat template | Training prompts go through `tokenizer.apply_chat_template`, exactly like inference (`src/train.py`) |
| B23 | Possible NaN loss / malformed examples | Over-long examples were cut from the end, dropping the target | Examples that do not fit `seq_len` are dropped and counted (`src/train.py`) |
| B24 | Thoughts ablation compared different data | `with_thoughts` skipped unannotated turns, `wo_thoughts` did not | Both arms train on the same turns (`src/build_sft.py`) |
| B25 | Elapsed-time line missing at inference | Training examples included "It has been about N days…", the monolithic listener did not | One shared `gap_statement()` for both (`src/build_sft.py`, `src/baselines/listener_mono.py`) |
| B26 | Fine-tuned model would load at full size / truncate | Supporter role said `quantization: awq` (ignored by transformers, so 15 GB unquantised) and `max_tokens: 256` (too short for analysis + strategy + response) | `nf4` (the base it is trained on) and 512 tokens (`configs/models.yaml`) |
| B27 | One malformed model field could kill a stage | `int()` / `float()` on model JSON ("2 - intermediate", a list instead of a string) | Tolerant `to_int` / `to_float` / text coercion (`src/common.py`, `src/annotate.py`, `src/sessions.py`, `src/agents.py`) |

### 4.4 Infrastructure changes

* **Parallel requests** (`llm.pmap`, `parallel.workers: 4`) with `OLLAMA_NUM_PARALLEL=4`,
  `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` on the server: ~2.3× faster, identical outputs.
* **Full corpus size** set to 1000 profiles (~3,000 sessions generated, ~2,000 kept) to fit the GPU budget.
* Phase scripts on the server: `phaseA.sh` (v2), `phaseB.sh`, `phaseC.sh`, chained by `run_v2.sh`.

---

## 5. Known limitations that remain in v2

* **Judge strength (P2).** Only open models on the server are available, so Mistral-Nemo 12B remains the
  judge. Expect compressed scores; differences between arms may stay within noise.
* **Small n.** 30 test profiles per arm; confidence intervals are wide. Report results as preliminary.
* **Gate threshold (P10).** If the judge still rates almost nothing above tau = 0.5, the calibrated gate
  will again release everything, and only hard rule-based violations will trigger it.
* **No human agreement yet.** IP and PRI headlines are marked "judge-human agreement NOT YET MEASURED"
  until `human_eval/` is completed.
* **External sets.** ExTES failed to download (connection reset); ES-MemEval mapped only 18 profiles with
  no annotated need. The out-of-distribution arms are not part of the 7-day plan.
* **Comparability with the COCOON paper.** Different supporter (Qwen vs Llama-3-8B), simulator and judge
  (open models vs GPT-4o), language (English) and n — compare trends, not absolute numbers.

---

## 6. What is in this folder

```
results/v1/
├── README.md                     this file
├── runs/week1_v1/
│   ├── results_v1.md             the report generated from v1 (partial)
│   ├── phaseA_v1.log             the run log
│   ├── dialogues/<arm>.jsonl     all 432 evaluation dialogues, with agent artifacts per turn
│   ├── scores/<arm>/             judge scores (base_instruct, reactive_baseline, cellA_mono_ungated)
│   ├── conformal/                calibration.json, calibration_rows.json, alpha_sweep.json
│   ├── memory/                   need-state memory logs per profile and arm
│   ├── arm_summary_<arm>.json    per-arm run summary
│   └── calls.jsonl               one line per model call (role, sizes, no content)
├── data/seeds/, data/profiles/   the exact seeds, profiles and splits v1 (and v2) use
├── reports/seed_stats.md         seed screen statistics
└── configs/                      the configuration v1 ran with
```

Reading a dialogue: each line of `dialogues/<arm>.jsonl` is one session with `turns` (role, text, phase,
ladder rung) and, for the agent arms, `agent_artifacts` (Analyzer state, Strategist plan, Critic verdict
and gate path per supporter turn).

---

## 7. Next steps

1. **v2 Phase A** – rerun all six arms with the fixes above on the same 30 test profiles → `runs/v2/results_v2.md`.
2. **Phase B** – generate the training corpus for the 328 train + val profiles with the fixed simulator
   (~600 filtered, annotated sessions).
3. **Phase C** – QLoRA fine-tune `with_thoughts`, evaluate it on the same test profiles and judge, and add it
   to the v2 table.

`run_v2.sh` on the server runs all three in sequence.
