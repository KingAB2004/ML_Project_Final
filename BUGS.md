# Bugs found and fixed (v2 → v3 and the full-run preparation)

This file lists every pipeline bug fixed on 2–3 October 2026, after the v1 baseline run: what went wrong, why it
mattered, how it was fixed, and how the fix was checked. Bugs B1–B28, found during and right after v1, are
documented in `results/v1/README.md` (section 4); numbering continues from there.

All fixes live in this local copy, which is the version used for the full run on the new server. The lab server
(`cse@10.50.28.201`) received the fixes needed for its own 50-profile v3 run (B29–B34); the later ones are local
only. The component test suite went from 79 to **82 checks, all passing**, with a regression check for each code
fix that can be tested without a GPU.

Status legend: **verified** = confirmed on real model output; **tested** = covered by a unit check with the
stub backend; **pending** = fix in place, real-model confirmation still to come.

---

## Evaluation and inference

### B29 — The supporter's private reasoning still reached the dialogue (verified)
- **Problem.** The single-model supporter (base_instruct, reactive_baseline, cellA, cellB) is asked for
  `<analysis>…</analysis> <strategy>…</strategy> <response>…</response>`; only the response may be shown. The
  base model often wrote the analysis and strategy tags but no `<response>` tag, and copied the label
  `what you say to them:` from the system prompt's placeholder. `parse_target` found no `<response>` tag and
  returned the whole generation, so the analysis went into the dialogue.
- **Why it mattered.** The simulated seeker and the judge saw the supporter's hidden reasoning, which distorts
  every score for those arms. The v2 pre-check found 2 leaking turns and refused to start the v2 run.
- **Fix.** `parse_target` (`src/build_sft.py`) now drops thought blocks (including an unclosed one), strips a
  copied reply label, and returns an empty response when the generation holds only thoughts.
  `MonolithicListener` (`src/baselines/listener_mono.py`) regenerates once when the response is empty, then
  uses the templated reflective fallback; the raw generation is never shown. The system prompt's placeholder is
  now `<response>your reply</response>`, with the length rule stated separately.
- **Check.** Real-model probe: 0 leaking turns, 0 fallbacks, replies about 27 words (matching the multi-agent
  arm, which also removes the v1 length confound). v3 run: 0 leaking turns in every arm. Unit checks cover the
  tagged, labelled, unclosed and thoughts-only cases.

### B30 — `reactive_baseline` never used the reactive simulator (tested)
- **Problem.** The arm sets `simulator: reactive`, but `evaluate.run_session` built the simulator prompt without
  that flag, so it always used the proactive (uninvited) seeker.
- **Why it mattered.** v1's "proactive vs reactive" comparison was really `base_instruct` with different
  openers; the reactive baseline did not exist.
- **Fix.** `run_session` takes `reactive` and `run_arm` passes `spec["simulator"] == "reactive"`
  (`src/evaluate.py`).
- **Check.** Unit check that the reactive arm's simulator gets the reactive system prompt; the v3 probe showed
  help-seeking seeker turns.

### B31 — The conformal gate was calibrated on the profiles it is scored on (verified)
- **Problem.** The gate threshold was fitted on cell C dialogues of the same 30–40 test profiles the gated arms
  are evaluated on. The calibration split (`profiles_calibration.jsonl`) existed but was never used.
- **Why it mattered.** Conformal risk control holds only for held-out data exchangeable with the test set;
  calibrating on the test profiles voids the guarantee and makes the gate look better than it is.
- **Fix.** New arm `calib_dec_ungated` (cell C's setup on the `calibration` eval set) in `configs/arms.yaml`;
  the phase scripts and `scripts/run_all.py` calibrate on it.
- **Check.** v3: the gate was fitted on 81 sessions of calibration profiles (300 turns), separate from the 40 test
  profiles.

### B32 — The gate could never fire at tau = 0.5 (verified)
- **Problem.** A turn counts as a violation when the judge rates its intrusiveness (IP) above tau. With tau 0.5,
  0 of 300 calibration turns qualified (maximum 0.357), so lambda_hat was 1.0 and the gate released everything.
- **Why it mattered.** The gated cells (B, D) would be copies of the ungated ones (A, C), so the 2×2 could not
  test the gate.
- **Fix.** `conformal.tau: 0.20` (`configs/default.yaml`), chosen by the project lead: 12.7% of calibration turns
  are violations, above the alpha = 0.10 budget, so the gate has to act. The calibration was re-fitted from the
  saved rows, with no new model calls.
- **Check.** lambda_hat 0.125, measured risk 5.7% against the 10% budget. In cellB: 49% of turns released, 39%
  revised, 13% fallback (v1's broken gate: 72% rewritten or replaced).

---

## Fine-tuning

### B33 — `train.py` crashed on transformers 5.x (verified)
- **Problem.** `TrainingArguments(warmup_ratio=…)`: transformers 5 removed the argument
  (`unexpected keyword argument 'warmup_ratio'`).
- **Why it mattered.** No adapter could be trained on the server's software stack.
- **Fix.** `src/train.py` checks the signature: `warmup_ratio` on 4.x, `warmup_steps` given as a ratio on 5.x.
- **Check.** Smoke fine-tune on Qwen2.5-3B: trained, saved, reloaded the adapter, generated with the adapter on and
  off.

### B34 — QLoRA ran out of memory on a 12 GB GPU (pending)
- **Problem.** `prepare_model_for_kbit_training` converts every non-quantized weight to fp32. For Qwen2.5-7B the
  152k-vocabulary embedding and output layer alone add about 4.4 GB, and training died at step 2 with
  `CUDA out of memory … 11.45 GiB in use`.
- **Why it mattered.** No fine-tuned model, so no Phase C result.
- **Fix.** `src/train.py` loads the model in bf16 and only enables gradient checkpointing and input gradients
  (what checkpointing needs). The phase script also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Check.** Pending: the 7B fine-tune runs in Phase C on the lab server after Phase A. Look for
  `adapter saved` in `phaseC50.log`.

### B35 — Base-model download failed silently (verified)
- **Problem.** Downloading Qwen2.5-7B-Instruct from the Hub hit connection resets on the xet endpoint
  (`Errno 104`, then `httpx.ReadError`), leaving an incomplete cache. A filter mistake then fetched the weights
  without the config and tokenizer files.
- **Why it mattered.** Fine-tuning cannot start without the complete base model.
- **Fix.** Retry loop with `HF_HUB_DISABLE_XET=1` and longer timeouts, downloading the whole repository
  (`scripts/dl_model.sh`, `scripts/full_run.sh`). `hf` is used (`huggingface-cli` no longer works in
  huggingface_hub 1.x), with a fallback for older installs. `full_run.sh` loads the config to confirm the download.
- **Check.** All four weight shards (14.9 GB) plus config and tokenizer present; config and tokenizer load.

---

## Orchestration (`scripts/run_all.py`, `scripts/full_run.sh`)

### B36 — The end-to-end runner had five gaps (tested)
- **Problems.**
  1. It calibrated the gate on cell C dialogues of the test profiles (the B31 problem again).
  2. It generated corpus sessions for every profile, including test and calibration ones (wasted compute, and
     those sessions sat next to the training data).
  3. It trained the adapters but never used them: the fine-tuned arms were evaluated on the base model, and on
     Ollama, which cannot load an adapter.
  4. By default it ran all 27 arms, including external datasets that are not available.
  5. It loaded EmpatheticDialogues through `datasets`, which in version 4.x refuses that dataset
     (`Dataset scripts are no longer supported`).
- **Why it mattered.** A full run would have produced invalid calibration, a fine-tuned arm that was really the
  base model, and failures on missing data.
- **Fixes.** Calibration uses `calib_dec_ungated`; the corpus is generated for train + val profiles only;
  `evaluate.py --adapter` points the fine-tuned arms at the adapters the run trained, on the transformers
  backend; the default arm set is the six baselines plus the two fine-tuned arms (`--arms all` for everything);
  the original EmpatheticDialogues release is fetched by `scripts/fetch_data.py --ed`, and
  `data/raw/empatheticdialogues.jsonl` ships with the repo. The order is now dataset → fine-tuning → test the
  fine-tuned supporters → calibration + baselines → report.
- **Check.** Stub-backend runs of the whole plan (locally and on the lab server) pass every stage except
  training, which needs a GPU.

### B37 — Phase C's adapter mapping would have leaked into the baseline phase (prevented)
- **Problem.** Cells A–D default to `supporter: sft_with_thoughts`. Writing the adapter path into
  `configs/arms.yaml` for Phase C would have left it there, so the next baseline run would label cells A–D as
  fine-tuned while Ollama silently ran the base model.
- **Fix.** `scripts/phaseC50.sh` restores `arms.yaml` on exit; the full run passes the adapter per arm with
  `--adapter` and never edits the config.

---

## Dataset generation

### B38 — Resistance level was tied to problem type (tested)
- **Problem.** Seeds come out round-robin over 12–14 problem types, and the 1:2:1 resistance plan was assigned by
  position. Because those cycle lengths share a factor of 2, each problem type received only two of the three
  levels. v1: appearance anxiety had 4 high-resistance profiles against 37 medium; breakup with partner 4 low
  against 27 high; conflict with parents 4 high.
- **Why it mattered.** Any result by resistance level would be confounded with the type of problem.
- **Fix.** `src/profiles.py` shuffles the plan with a fixed seed; the exact 1:2:1 counts are kept.
- **Check.** On v1's seed order every problem type now gets all three levels at roughly 25/50/25.

### B39 — Follow-up sessions could lose the terminal need (tested)
- **Problem.** When a profile is advanced between sessions, `advance_profile` accepted whatever chain the model
  returned. In v3, 5 of 103 follow-ups came back with 1 or 2 nodes, so the "terminal need" became a shallower
  node (e.g. "I feel like my time and effort were wasted"), and later sessions inherited it.
- **Why it mattered.** The terminal need is the ground truth for Success Rate and for the annotator.
- **Fix.** `src/sessions.py` requires exactly three non-empty nodes, retries once, and otherwise keeps the previous
  chain, recording `chain_kept_from_previous`.
- **Check.** Unit check with a model that returns a one-node chain.

### B40 — Unusual model JSON could crash a dataset stage (tested)
- **Problem.** Profile building used `int()` on node depths and assumed nodes, `persona_hidden` and `memory`
  had the expected types. The seeker prompt joined a disclosure-trigger string letter by letter
  ("p, a, t, i, e, n, c, e").
- **Why it mattered.** One malformed reply raises inside the parallel map and stops the stage, hours into an
  overnight run; the garbled prompt degrades the simulator silently. Neither happened in 400 + 103 profiles,
  but the chance grows with 1000.
- **Fix.** Coercion in `profiles.build_profile`, `dialogue.as_list` / `memory_lines` / `simulator_system`, and
  `sessions.advance_profile`; a malformed chain becomes a rejected profile with a reason.
- **Check.** Unit check with malformed chain, persona and memory values.

### B41 — `annotations.jsonl` duplicated on every rerun (tested)
- **Problem.** `annotate.py` appended records, so a rerun or resume doubled the file.
- **Fix.** The file is rewritten at the end of the stage; reruns replay from the LLM cache at no cost.

### B42 — Reply length differed between the dataset and the fine-tuning prompt (tested)
- **Problem.** Suggestion-phase replies allowed 1–4 sentences, but the fine-tuning system prompt and the
  listening phase say 1–3 (11 of 1280 training replies had more than 3).
- **Fix.** `prompts/suggest_stage.md`: 1–3 sentences, no lists, no labels.

### B43 — Training targets carried profile and taxonomy jargon (pending)
- **Problem.** The annotator sees the hidden profile and the ladder definitions. In v3, 54 of 1280 analyses said
  "terminal need", 17 said "resistance", and strategies said "terminal need" (11) and "intermediate need" (17).
- **Why it mattered.** These phrases become fine-tuning targets, so the model would learn to reason in jargon
  that a real conversation never reveals.
- **Fix.** `prompts/annotate_turn.md`: both phrases must read as inferences from the conversation, never quoting
  the profile or using the prompt's terms.
- **Check.** Pending (needs the judge model). After the dataset step:
  `grep -ci "terminal need" data/corpus/annotations.jsonl` should be near 0 (v3: 54).

### B44 — An echoed template was read as an answer (tested)
- **Problem.** The seed screen asks for `VERDICT: yes|no`; a model that echoed the template was parsed as "yes".
  3 of 702 cached v1 replies did this and were counted as passes. The chain check's `RESTATEMENT: yes|no` had
  the same hazard (never triggered).
- **Fix.** The parsers in `src/seeds.py` and `src/profiles.py` ignore `yes` followed by `|`.
- **Check.** Unit checks for echoed and genuine answers.

### B45 — Missing problem-type taxonomy file (fixed)
- **Problem.** The local repo had no `data/raw/problem_types.json`, so a new run would have written a slightly
  different taxonomy from the one v1 and v3 used.
- **Fix.** Copied from the lab server (`.gitignore` already allows committing it).

---

## Configuration and hardware

### B46 — Judge sampled three times at temperature 0 (fixed)
- **Problem.** `judge.samples_per_item: 3` with `judge.temperature: 0.0`: greedy decoding gives the same answer
  each time, so every judge call (filter, calibration, all scoring) cost three times as much for nothing.
- **Fix.** `samples_per_item: 1`, matching v1 and v3 so results stay comparable.

### B47 — Ollama context length left to the server default (fixed)
- **Problem.** Four Qwen roles (analyzer, strategist, critic, supporter) set no context length. Newer Ollama
  versions size the default by available VRAM; if it differs from the generator's 4096, Ollama reloads the model
  at every role switch.
- **Why it mattered.** The multi-agent arms switch roles several times per turn and would slow down sharply. (The
  lab server's default happened to be 4096: logs showed constant 4096-token slots, no truncation, reloads only
  between arms.)
- **Fix.** `max_model_len: 4096` on every Qwen role in `configs/models.yaml`.

### B48 — Training assumed bf16 (fixed)
- **Problem.** bf16 was hard-coded; GPUs older than Ampere (V100, T4) do not support it.
- **Fix.** `src/train.py` and the transformers backend fall back to fp16 when bf16 is unavailable; Ampere and
  newer (including the RTX 5090) keep bf16.

### B49 — RTX 5090 (Blackwell) readiness (partly verified)
- **Problem.** Blackwell (sm_120) needs PyTorch 2.7+ built for CUDA 12.8+, bitsandbytes 0.46+ and a driver of
  version 570 or newer. A build without kernels for the GPU would surface only at fine-tuning, hours into the run.
  Ollama also keeps the judge resident for minutes, which training then competes with.
- **Fix.** `requirements.txt`: `torch>=2.7`, `bitsandbytes>=0.46`. `scripts/full_run.sh` runs a bf16 matrix
  multiply and a 4-bit layer on the GPU right after installation, and on failure reinstalls the CUDA 12.8 torch
  build (for drivers 570–579) before stopping with a clear message. Micro-batch is sized by VRAM (4 × 4 at 20 GB
  or more, 1 × 16 below; effective batch 16 either way). `train.py` unloads Ollama's models before loading.
- **Check.** The GPU check passes on the lab server, and the current torch build (2.13 + CUDA 13.0) lists `sm_120`
  kernels. On the 5090 itself: run `./scripts/full_run.sh smoke` first. Micro-batch 4 memory (estimated
  15–16 GB) is untested.

### B50 — Evaluation batch left at the default of 8 (fixed)
- **Problem.** `TrainingArguments` set the training micro-batch but not the evaluation one, which defaults to 8.
  On the lab server (RTX 3060, 12 GB) the 50-profile fine-tune trained epoch 1 normally (loss 1.11 to 0.74,
  72/144 steps) and then crashed with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.02 GiB`
  in the end-of-epoch validation pass. Eight sequences of 152k-vocabulary fp32 logits do not fit beside the
  model. No adapter was saved, because saving happens after that evaluation.
- **Why it mattered.** Every training run with a validation file would die at the end of its first epoch on any
  card, including the 5090 with micro-batch 4, so `full_run.sh` would never reach testing.
- **Fix.** `per_device_eval_batch_size` equals `micro_batch_size`, which training has already shown fits.
- **Check.** Verified on the lab server on 2026-10-03: the phase C rerun passed both validation passes (eval loss
  0.761 after epoch 1, 0.770 after epoch 2) and saved the best adapter (epoch 1, checkpoint-72).

---

## Known limitations (not fixed)

- **Judge ceiling.** The open judge rates AELS near the top (median 6.0 of 7) and IP near the bottom, so
  differences between arms may stay within noise. The AELS filter therefore rejects almost nothing; raising its
  threshold would mostly filter on judge noise.
- **No memory in the corpus.** In follow-up sessions the dataset's supporter sees `[memory] none.`, so the
  fine-tuned model does not learn to refer back to earlier sessions.
- **Privileged annotator.** The annotator still reads the hidden need to judge each turn (B43 limits only the
  wording).
- **Formulaic supporter.** About 8% of dataset turns cheer the seeker on, and openings repeat. An A/B test of a
  stricter prompt (60 real contexts) removed the cheering but cut distinct openings from 41 to 16, so the prompt
  was left unchanged.
- **Unused setting.** `filter.max_regen_attempts` is not read by any code.
