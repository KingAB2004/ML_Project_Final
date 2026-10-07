# v3 results: the 1000-profile corpus, fine-tuning and the scaling study

This folder holds every result of the full-size run: the 1000-profile dataset built on the lab server, the
fine-tuning runs, and the analyses run on top of them (GPU and CPU). Nothing here is hand-edited: it is
copied from `runs/` (which git ignores) by `scripts/collect_results.sh`, and refreshed by rerunning that
script as each pending step finishes. Model weights are not included (too large for git).

**Status on 7 October 2026, 14:20 IST.** Dataset: done. All six fine-tunes (0.5B / 3B / 7B x both arms): done.
LoRA geometry and rank truncation: done for all sizes. PVI: done for all three sizes (the 7B rerun finished
on the lab on 6 Oct 20:06). Test-profile evaluation (section 6): 0.5B and 7B, both arms, done. The 3B was
not evaluated on test profiles (we chose to skip it). SOP evaluations on the lab RTX 3060:
- **Our need-state memory (`mem_needstate`, section 8):** done, 6 Oct 20:06 to 7 Oct 02:28.
- **Baselines and the 2 x 2 (section 7):** gate calibration, `base_instruct`, `reactive_baseline`, cell A
  and cell B (judge scores) are done. Cell B's counterfactual PRI is running; cells C and D come next.
- **Still queued:** the other memory arms and the memory metrics, ES-MemEval, and the alpha sweep (section 9).

| Folder | What | Where it ran |
|---|---|---|
| `dataset/` | corpus statistics, the generation log, the pipeline state | lab server, RTX 3060 (Ollama) + CPU |
| `training/` | loss curves, best checkpoints, adapter configs, cleaned training log | laptop RTX 4060 (0.5B, 3B); lab RTX 3060 (7B) |
| `analyses/pvi/` | usable information in the thoughts | laptop RTX 4060 (forward passes) |
| `analyses/lora_geometry/` | SVD geometry of the LoRA updates, rank truncation | CPU (SVD) + laptop RTX 4060 (truncation) |
| `analyses/scaling/` | scaling-law table and training curves | CPU |
| `test_eval/` | Success / IP / PRI of the fine-tuned 0.5B and 7B on the 100 test profiles | lab RTX 3060 |
| `baselines_1000/` | Outcomes 1-2: gate calibration, baselines, the 2 x 2 (decomposition x gate), later the alpha sweep | lab RTX 3060 (Ollama) |
| `memory_1000/` | Outcome 4: multi-session memory arms, need-state memory files, memory metrics | lab RTX 3060 (Ollama) + CPU |

Hardware and software: laptop RTX 4060 Laptop 8 GB, torch 2.14 (CUDA 13), transformers 5.18, peft 0.21,
bitsandbytes 0.50; lab server RTX 3060 12 GB with the same library versions, Ollama for generation and judging.

---

## 1. The dataset (`dataset/`)

Built by `scripts/run_all.py --scale full --only check,corpus,sft` on the lab server, 3 Oct 14:31 to 4 Oct
08:15 (about 17.7 h). The step-by-step log is `dataset/full_dataset.log`; `dataset/dataset_stats.json` holds
every number below (`python extra/dataset_stats.py`).

| Stage | Result | Time |
|---|---|---|
| Seeds (EmpatheticDialogues) | 1,000 released of 2,000 screened (85.5 % passed the three-step need-chain screen) | 11 min |
| Profiles | 1,000; splits 720 train / 100 val / 100 test / 80 calibration; 0 rejected | 70 min |
| Session 1 (train + val profiles) | 820 sessions | 86 min |
| Linked follow-up sessions | 1,673 (sessions 2-4); temporal integrity check passed | 5.0 h |
| Filter | 2,369 kept, 66 rejected, 58 later sessions of a rejected sequence dropped | 49 min |
| Annotation (Analysis + Strategy per supporter turn) | 21,429 turns | 9.1 h (incl. ~0.7 h label-diversity check) |
| SFT files | 18,821 train / 2,608 val examples per arm (`with_thoughts`, `wo_thoughts`) | seconds |

Corpus shape:

| Item | Value |
|---|---|
| Profiles with dialogues | 801 (of the 820 train + val profiles; 19 lost their first session in the filter) |
| Sessions per profile | 1: 21, 2: 250, 3: 272, 4: 258 |
| Sessions by position | session 1: 801, 2: 780, 3: 530, 4: 258 |
| Turns per session | mean 20.1 (16-24), 47,596 in total, about 1.15 M words |
| Gap before a follow-up session | 1.0 to 56 days, median 7.6 days (log-uniform by design) |
| Resistance mix (all 1,000 profiles) | low 250 / medium 500 / high 250 (the 1:2:1 design) |
| Problem types | 13; largest: appearance anxiety 150, job crisis 149, other 149, breakup 147, academic pressure 131 |
| Strategy labels | 21,354 distinct phrases in 21,429 turns, 19,078 similarity clusters; top-3 clusters cover 1 % (no label collapse) |
| AELS of kept sessions | mean 6.07 / 7, minimum 5.1 (threshold 5.0) |

**Why these numbers look the way they do**

- **Why 66 rejections, and which.** 50 were *non-English content* (Qwen2.5 occasionally drifts into Chinese
  characters, its main pre-training language besides English; the ASCII-ratio screen catches them), 13 were
  the seeker *reciting its hidden need too early* (the leakage guard: a seeker that says its terminal need in
  its first two turns makes Success Rate trivial), and only 3 failed the AELS listening-quality threshold.
  The judge rates the generator's supporter highly (mean 6.07 / 7), so the quality filter rarely binds; the
  structural checks do the real filtering, as in the 50-profile run.
- **Why 58 extra sessions were dropped.** Filtering is per profile: if session 2 fails, sessions 3 and 4 of
  that profile are a broken time sequence and are dropped with it. That is also why later positions are
  thinner (801 -> 780 -> 530 -> 258): the session count is drawn from 2-4 per profile, and each rejection
  removes the tail.
- **Why 21,354 different strategy phrases.** The labels are free-form content-level phrases (COCOON's design,
  not a fixed taxonomy). A degenerate annotator would repeat a few phrases; here almost every label is
  distinct, so the fine-tuning signal did not collapse.
- **Why the problem types are uneven.** Seeds are stratified by type, but the ESConv types are not equally
  common among EmpatheticDialogues situations; rare types (children, friends) simply have few seeds that
  pass the screen.
- **How it compares with COCOON.** COCOON reports 3.5k single-session dialogues at 18.8 turns; ours is 2,369
  sessions at 20.1 turns, but organised as 801 linked multi-session sequences with time gaps, in English, with
  supporter-side annotation: the multi-session structure is what COCOON's corpus does not have.

---

## 2. Fine-tuning (`training/`)

Same QLoRA recipe for every size (`configs/default.yaml` -> `train:`): 4-bit NF4 base, LoRA r = 32,
alpha = 64, dropout 0.05 on all attention and MLP projections, sequence length 2048, micro-batch 1 x
accumulation 16, lr 2e-4 cosine, 3 epochs, loss on target tokens only, best-validation checkpoint kept.
QLoRA because the 7B does not fit a 12 GB card in 16-bit; the smaller sizes use the identical recipe so that
size is the only variable in the scaling study.

### Qwen2.5-0.5B-Instruct (laptop RTX 4060, 4 Oct 08:30-14:54)

| Arm | Trainable params | Time | Mean train loss, epoch 1 / 2 / 3 | Validation loss, epoch 1 / 2 / 3 | Kept |
|---|---|---|---|---|---|
| `with_thoughts` | 17.6 M (3.4 %) | 3 h 15 m | 0.979 / 0.731 / 0.553 | 0.893 / **0.842** / 0.876 | epoch 2 |
| `wo_thoughts` | 17.6 M (3.4 %) | 3 h 08 m | 1.026 / 0.719 / 0.454 | 0.918 / **0.876** / 0.957 | epoch 2 |

3,531 optimizer steps per arm at 3.2 s/step, about 3.3 GB of VRAM. No example was dropped for length.
Plot: `training/0.5B/training_curves_0.5B.png`. Files: `training/0.5B/train_summary_*.json`,
`trainer_state_*.json` (full loss history), `adapter_config_*.json`; log: `training/scale_local_clean.log`.

**Why these numbers**

- **Overfitting in epoch 3.** Training loss drops in a step at the start of each epoch (visible in the plot):
  the model sees the same 18,821 examples again and partly memorises them. Validation loss improves from
  epoch 1 to 2 and then rises, so the third epoch hurts generalisation. The trainer keeps the best epoch
  automatically, so the saved adapters are the epoch-2 ones. The same pattern appeared in the 50-profile 7B
  run (best = epoch 1 of 2). Two epochs would have been enough.
- **wo_thoughts overfits harder** (train 0.454 vs validation 0.957 at epoch 3). Its targets are only the short
  reply, which is easier to memorise; the with_thoughts target also contains the formulaic Analysis and
  Strategy text, which regularises it.
- **The two arms' validation losses are not comparable.** with_thoughts is averaged over thoughts + reply
  tokens, wo_thoughts over reply tokens only. The lower number for with_thoughts (0.842 vs 0.876) does not
  mean it is better; the fair comparison on the same reply tokens is in the PVI analysis (section 3).

### Qwen2.5-3B-Instruct (laptop RTX 4060): both arms done

| Arm | Trainable params | Time | Mean train loss, epoch 1 / 2 / 3 | Validation loss, epoch 1 / 2 / 3 | Kept |
|---|---|---|---|---|---|
| `with_thoughts` | 59.9 M (1.9 %) | 11 h 39 m | 0.771 / 0.574 / 0.391 | 0.715 / **0.687** / 0.752 | epoch 2 |
| `wo_thoughts` | 59.9 M (1.9 %) | 10 h 20 m | 0.781 / 0.541 / 0.290 | 0.718 / **0.696** / 0.828 | epoch 2 |

3,531 optimizer steps per arm, about 10.5 s/step and 4.8 GB of VRAM. The first `wo_thoughts` attempt crashed
at step 452 (5 Oct 19:28) while another program held about 3 GB of the 8 GB card; the rerun from scratch
(5 Oct 21:49 to 6 Oct 08:09) is the one reported. Plot: `training/3B/training_curves_3B.png`. Files:
`training/3B/train_summary_*.json`, `trainer_state_*.json`, `adapter_config_*.json`; log:
`training/scale_3b_clean.log`.

As at 0.5B, `wo_thoughts` overfits harder in epoch 3 (validation +0.132, against +0.065 for `with_thoughts`;
train 0.290 vs 0.391): the short reply alone is easier to memorise.

### Qwen2.5-7B-Instruct (lab RTX 3060): both arms done

| Arm | Trainable params | Time | Mean train loss, epoch 1 / 2 / 3 | Validation loss, epoch 1 / 2 / 3 | Kept |
|---|---|---|---|---|---|
| `with_thoughts` | 80.7 M (1.1 %) | 26 h 23 m | 0.666 / 0.473 / 0.277 | 0.629 / **0.612** / 0.704 | epoch 2 |
| `wo_thoughts` | 80.7 M (1.1 %) | 23 h 22 m | 0.532 / 0.316 / 0.118 | 0.514 / **0.489** / 0.592 | epoch 2 |

1,177 steps per epoch at about 27 s/step (7.6 B parameters, micro-batch 1 x 16, gradient checkpointing,
4-bit de-quantisation on every matmul). Plot: `training/7B/training_curves_7B.png`. Files:
`training/7B/trainer_state_with_thoughts.json` (the loss history; this run started before `train.py` began
writing it into the summary), `train_summary_with_thoughts.json`, `adapter_config_with_thoughts.json`; log:
`training/full_train_clean.log`. The adapter weights (309 MB each) are in `runs/full_1000/adapter_*/` (not
tracked by git).

**Why the 7B `wo_thoughts` loss is so low (0.489, against 0.696 at 3B).** Training code and data are identical
to the other sizes (checked: same SFT files, same `train.py` logic). The reason is the data: every supporter
reply in the corpus was *written by Qwen2.5-7B-Instruct*, so the 7B is scoring its own outputs. Even untuned,
the 7B gives the replies a loss of 0.99, against 1.64 for the untuned 3B (rank-0 column of the truncation table,
section 4). The `with_thoughts` target also holds the Analysis and Strategy, written by Mistral-Nemo, which the
7B cannot predict as well (0.612). So the 7B point of the `wo_thoughts` scaling curve is inflated by
self-generated data, a confound to state in the report; the 0.5B and 3B never produced the data.

**Why these numbers (all sizes)**

- **Every size is best at epoch 2 and overfits in epoch 3**, and the gap grows with size (validation rises
  by 0.034 for 0.5B, 0.065 for 3B, 0.092 for 7B): a bigger model memorises the repeated examples faster.
  Train loss at epoch 3 (0.553 / 0.391 / 0.277) shows that memorisation directly.
- **Bigger is better on held-out data**: best validation loss 0.842 -> 0.687 -> 0.612 for `with_thoughts`.
  The fit over the three sizes is L = 7.14 N^-0.108 (section 5); with only 3 points (1 residual degree of
  freedom) the exponent is a description, not a reliable law.
- **Diminishing returns**: 0.5B -> 3B (7.7x parameters) cuts the loss by 0.155; 3B -> 7B (2.4x) by 0.075.

---

## 3. Usable information in the thoughts (`analyses/pvi/`, GPU)

Question: how much do the annotated Analysis and Strategy tell the model about the reply?
PVI per validation turn = log2 p_with(reply | context, thoughts) - log2 p_wo(reply | context), using the
with_thoughts adapter (teacher-forced through the gold thoughts) and the wo_thoughts adapter (Ethayarajh et
al. 2022). Both score the identical `<response>...</response>` tokens. Mean PVI = usable information in bits.
2,608 validation turns, 97 profiles. Run time: about 5 minutes on the RTX 4060 (0.5B, 3B), 53 minutes on the
lab RTX 3060 (7B). Script: `extra/pvi.py`.

| Size | V-information (bits / turn) [95 % CI] | bits / token | PVI < 0 | reply NLL with thoughts | reply NLL without | Spearman(thought length, PVI) |
|---|---|---|---|---|---|---|
| 0.5B | **8.91 [8.49, 9.34]** | 0.24 | 17.9 % | 0.727 | 0.891 | 0.16 |
| 3B | **7.36 [7.02, 7.72]** | 0.20 | 17.3 % | 0.571 | 0.707 | 0.13 |
| 7B | **3.31 [3.01, 3.62]** | 0.09 | 32.9 % | 0.434 | 0.495 | 0.10 |

By ladder rung: 0.5B L1 (gentle) 7.87 bits (n = 1,160), L2 (deeper) 9.76 bits (n = 1,444), L0 -0.22 (n = 4);
3B L1 6.43, L2 8.12, L0 0.96; 7B L1 3.10, L2 3.49, L0 -1.80.

**Why**

- **The thoughts carry a lot of information about the reply (about 9 bits, 18 % lower reply loss).** The
  annotator wrote the Strategy *after* seeing the supporter's turn ("acknowledging the frustration, then
  gently asking how they feel"), so the strategy describes the reply's content and the reply becomes much
  more predictable. This is information in the **gold** thoughts. At inference the model writes its own
  thoughts, so the benefit there can be smaller; the test-profile comparison of the two arms answers that.
- **Deeper turns gain more (L2 > L1).** A deeper move names a specific feeling or need, which the Analysis
  states explicitly; a gentle reflective turn is predictable from the context alone.
- **17.9 % of turns have negative PVI.** For these, the thoughts made the gold reply *less* likely: the label
  describes something the reply does not do. They are the first annotations to audit
  (`per_example_0.5B.jsonl`, sort by `pvi_bits`).
- **Longer thoughts help only a little** (Spearman 0.16 / 0.13): content matters more than length.
- **The bigger model gets less out of the thoughts (8.9 -> 7.4 bits, CIs do not overlap).** The 3B predicts the
  reply much better from the conversation alone (reply loss without thoughts 0.891 -> 0.707), so there is less
  left for the thoughts to explain. They still cut the 3B's reply loss by 19 % (0.707 -> 0.571), about the
  same relative drop as at 0.5B (18 %).
- **At 7B the information falls by more than half (7.4 to 3.3 bits), and one turn in three has negative PVI.**
  The thoughts now cut the reply loss by only 12 % (0.495 to 0.434), against 18-19 % at the smaller sizes.
  This is mostly the self-generated-data effect from section 2. The corpus replies were written by
  Qwen2.5-7B, so the 7B `wo_thoughts` adapter predicts them very well from the conversation alone (reply loss
  0.495, against 0.707 at 3B), which leaves little for the thoughts to add. Read the 7B point as "the thoughts
  matter less to a model that already knows how the replies were written". It is not evidence that larger
  models in general gain less from thoughts. A fair test needs replies written by a different model.

Figures: `pvi_hist.png` (one histogram per size; the 7B is narrower and sits closer to 0), and
`pvi_by_size.png` (V-information per size with 95 % CIs: a gentle drop from 0.5B to 3B, then a steep drop at 7B).

---

## 4. Geometry of the LoRA updates (`analyses/lora_geometry/`, CPU + GPU)

For every adapted matrix dW = (alpha/r) B A, the singular values are computed from the 32 x 32 core (QR of B
and A^T, then SVD), on the CPU in seconds. `--truncate` replaces every dW by its best rank-k approximation
(Eckart-Young) and measures validation loss on 300 examples (GPU, about 4 minutes). Script:
`extra/lora_geometry.py`.

| Size | Arm | mean norm ||dW|| | stable rank | effective rank (of 32) | energy in top 4 directions | update share early / mid / late layers |
|---|---|---|---|---|---|---|
| 0.5B | with_thoughts | 2.19 | 6.84 | 28.2 | 43 % | 26 / 34 / 40 % |
| 0.5B | wo_thoughts | 1.98 | 6.52 | 28.1 | 44 % | 27 / 35 / 38 % |
| 3B | with_thoughts | 4.20 | 3.62 | 24.9 | 62 % | 38 / 28 / 33 % |
| 3B | wo_thoughts | 3.78 | 3.63 | 25.3 | 61 % | 43 / 27 / 30 % |
| 7B | with_thoughts | 5.57 | 3.51 | 25.1 | 62 % | 34 / 30 / 36 % |
| 7B | wo_thoughts | 4.30 | 4.53 | 26.5 | 55 % | 38 / 29 / 33 % |

Overlap of the two arms' top directions phi(k) (1 = same subspace): 0.5B k=1 0.074, k=4 0.066, k=8 0.077,
against 0.003 / 0.011 / 0.022 for random subspaces; 3B 0.054 / 0.055 / 0.059 against 0.001 / 0.005 / 0.011;
7B 0.028 / 0.030 / 0.033 against 0.0007 / 0.003 / 0.006. The norm ratio with / wo grows with size: 1.10, 1.11,
1.30.

| Rank kept k | 0 (base) | 1 | 2 | 4 | 8 | 16 | 32 (full) | k for 95 % of the gain |
|---|---|---|---|---|---|---|---|---|
| with_thoughts val loss | 1.868 | 1.367 | 1.208 | 1.054 | 0.933 | 0.864 | 0.848 | 16 |
| wo_thoughts val loss | 1.874 | 1.350 | 1.183 | 1.048 | 0.951 | 0.896 | 0.895 | 16 |
| 3B with_thoughts | 1.526 | 0.915 | 0.823 | 0.755 | 0.710 | 0.693 | 0.694 | 8 |
| 3B wo_thoughts | 1.635 | 0.938 | 0.822 | 0.769 | 0.727 | 0.717 | 0.727 | 4 |
| 7B with_thoughts | 1.370 | 0.759 | 0.694 | 0.649 | 0.623 | 0.618 | 0.627 | 4 |
| 7B wo_thoughts | 0.990 | 0.547 | 0.494 | 0.470 | 0.460 | 0.467 | 0.499 | 2 |

(The first two rows are 0.5B.) The rank needed for 95 % of the gain falls with size: 16 -> 4-8 -> 2-4. At 7B,
rank 8-16 beats the full rank 32 for both arms: the smallest directions mostly carry the epoch-2 overfitting.

**Why**

- **The updates use most of their rank, but unevenly.** Effective rank 28 of 32 means no direction is
  negligible; stable rank about 7 and 43 % of the energy in the top 4 directions mean a few directions
  dominate. A single direction already recovers about half of the loss drop (1.868 -> 1.367); rank 16
  recovers 95 %. With 18.8k examples of a new format and style, rank 32 is not wasted, but rank 16 would do
  almost as well.
- **Later layers change more** (40 % of the update mass in the last third, rising with depth in the plot).
  Fine-tuning for a new output format and style mostly reshapes the upper layers, which produce the output,
  while the lower layers' general language features are reused; this matches what is commonly reported for
  LoRA fine-tuning.
- **The thoughts arm moves the weights 10 % more** (norm ratio 1.10): its target has more to learn (two extra
  free-text sections).
- **The two arms learn mostly different directions.** Their overlap (0.07) is 3-25x what random subspaces
  give, so they share a component (the supporter's tone and the reply format), but most of each update is
  specific to its objective: learning to write the analysis first changes the network differently from
  learning to reply directly.

- **The 3B update is more concentrated** (stable rank 3.6 against 6.8, 62 % of the energy in the top 4
  directions against 43 %), and rank 4-8 already recovers 95 % of the gain. The bigger model already has most
  of what the task needs, so a few directions steer it; rank 32 is more than the 3B needs. Unlike the 0.5B,
  the 3B puts the largest share of its update in the early layers (38-43 %).
- **At 3B rank 16 is slightly better than the full rank 32** on these 300 examples (0.717 vs 0.727 for
  `wo_thoughts`): dropping the smallest directions removes a little of the epoch-2 overfitting.

Figures: `effective_rank.png` (layer x module heatmap), `norm_and_similarity_by_depth.png`,
`rank_truncation.png`. Per-matrix numbers: `per_matrix.jsonl`.

---

## 5. Scaling study (`analyses/scaling/`, CPU)

`extra/scaling.py` combines the training logs, the PVI reply losses and (later) the test-profile scores of
every size into one table, fits L(N) = a N^-b over the sizes present, and measures the thoughts gain per size.

Now: `with_thoughts` is trained at all three sizes, so its fit exists: **best validation loss = 7.14 N^-0.108**
(N = non-embedding parameters 0.36 B / 2.77 B / 6.53 B). `wo_thoughts`: 0.876 -> 0.696 -> 0.489, fit
34.8 N^-0.185, steeper only because the 7B point is inflated by self-generated data (section 2). On the identical
reply tokens (PVI losses, now all three sizes), the exponents are 0.167 [0.162, 0.172] with thoughts and 0.187
[0.182, 0.191] without. With only 0.5B and 3B they were 0.118 and 0.113. The 7B steepens both exponents, and the
no-thoughts arm more, for the same self-generated-data reason. The test columns come from section 6.
Thoughts gain on the test profiles (with minus wo): Success 0.002 at 0.5B and 0.023 [0.000, 0.047] at 7B;
IP 0.021 at 0.5B and 0.007 at 7B. So at 7B the thoughts start to help Success and stop adding intrusiveness.
Both trends sit at the edge of their CIs.

Figures: `scaling_loss.png` (validation loss vs size, log-log, with the fit), `training_curves.png` (every
trained run side by side), `training_curves_<size>.png` (one per size; also copied into `training/<size>/`),
`scaling_test.png` (Success / IP / PRI vs size), `thoughts_gain.png` (with minus wo thoughts per size).

---

## 6. Test-profile evaluation (`test_eval/`, lab RTX 3060)

Each fine-tuned supporter talks to the same simulated seeker (Qwen2.5-7B) on the 100 held-out test profiles,
one session each, monolithic, no gate and no memory; the Mistral-Nemo judge scores every dialogue
(`scripts/scaling_eval.sh`). 95 % CIs: bootstrap over profiles.

| Size | Arm | Success | IP (lower better) | PRI observed | PRI counterfactual [95 % CI] |
|---|---|---|---|---|---|
| 0.5B | with_thoughts | 0.397 | 0.103 | 0.246 | 0.010 [-0.000, 0.021] |
| 0.5B | wo_thoughts | 0.395 | 0.081 | 0.233 | 0.006 [-0.004, 0.016] |
| 3B | both | not evaluated (skipped) | | | |
| 7B | with_thoughts | **0.415** | 0.091 | 0.243 | **0.018 [0.008, 0.029]** |
| 7B | wo_thoughts | 0.392 | 0.083 | 0.225 | 0.010 [0.001, 0.020] |

- **Success barely moves with size or thoughts** (0.392-0.415, CIs overlap): the judge's Success scale is
  coarse (mostly 3 or 4 of 7, as in v2), so small real gains may be invisible. At 7B the thoughts arm is
  ahead by 0.023 [0.000, 0.047], the only size where the gain reaches the edge of significance.
- **At 0.5B the thoughts make the supporter more intrusive** (IP +0.021 [0.010, 0.032]), with no Success gain:
  a small model that writes an analysis first acts on it too directly.
- **The 7B `with_thoughts` supporter provokes measurable reactance** (counterfactual PRI 0.018, CI above 0):
  its turns draw more pushback than a neutral reflection would. The fine-tuned 7B in v2 already looked pushier.
  The 7B `wo_thoughts` supporter does too, but less (0.010, CI just above 0).

Compare the untuned base model in section 7: Success 0.383, IP 0.065, PRI 0.192, counterfactual PRI -0.001.
Fine-tuning moves Success by at most +0.03, and it raises intrusiveness (+0.02 to +0.04) and reactance
(observed PRI +0.03 to +0.05, counterfactual +0.01 to +0.02). The SFT corpus teaches the supporter to name
needs more directly, which the seeker experiences as pushier.

---

## 7. Baselines and the 2 x 2: Outcomes 1-2 (`baselines_1000/`, lab RTX 3060, partial)

`scripts/run_all.py --scale full --run-dir runs/baselines_1000 --only calib,eval,score,report`, started 7 Oct
02:28. It has its own run directory, so the trained adapters in `runs/full_1000` are never touched. Every arm
runs the untuned Qwen2.5-7B-Instruct through Ollama as the supporter. The lab has no adapter mapped in
`configs/arms.yaml`, so the 2 x 2 cells test the architecture and the gate, not the fine-tune. The seeker is
Qwen2.5-7B and the judge is Mistral-Nemo-12B. 95 % CIs: bootstrap over profiles.

**Done so far:**
- the gate calibration;
- `base_instruct` and `reactive_baseline`, 100 test profiles, one session each;
- cell A (monolithic, ungated), 310 sessions: 100 test profiles, 2-4 linked sessions each;
- cell B (monolithic, conformal gate at alpha = 0.10), 310 sessions: judge scores done.

Cell B's counterfactual PRI is still being scored.

### 7.1 Conformal gate calibration (`conformal/`)

Cell C (decomposed, ungated) is rolled out on the **calibration** profiles, never on the test profiles. Each
supporter turn gets a nonconformity score s = 0.7 x critic-predicted IP + 0.3 x grounding-violation mass. Conformal
risk control (Hoeffding-Bentkus bound, delta = 0.1) then picks the largest threshold lambda-hat at which the
rate of intrusive turns (IP > tau = 0.20) among the released turns stays at or below alpha, with
probability 1 - delta.

| alpha | lambda-hat | turns released | intrusive-turn rate among released | upper bound |
|---|---|---|---|---|
| 0.05 | 0.0 | 13 % | 2.0 % | 4.1 % |
| **0.10 (used)** | **0.2** | **73 %** | **6.7 %** | **9.8 %** |
| 0.20 | 1.0 (gate never fires) | 100 % | 10.7 % | 14.4 % |

300 calibration turns. The bound is not vacuous at any alpha.

- **alpha = 0.10 holds back about a quarter of the turns and keeps the bound under 10 %.** The ungated rate is
  10.7 %, so a gate is needed to get under 10 %.
- **alpha = 0.20 never fires.** The ungated supporter is already inside 20 %, so the planned alpha = 0.20 arm
  will behave like the ungated cell. **alpha = 0.05 releases only 13 %** of turns: it buys safety by almost
  never speaking freely, and the gated turns fall back to a reflection. This is the alpha trade-off the
  alpha sweep (section 9) measures on the test profiles.

Figure: `extra/baseline_plots/gate_calibration.png`.

### 7.2 Scores so far (`scores/`, `extra/baseline_plots/`)

| Arm | Sessions | Success | AELS listening | CRS comforting | RAC competence | IP (lower better) | PRI observed | PRI counterfactual [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| `base_instruct` | 100 | 0.383 | 0.836 | 0.587 | 0.774 | 0.065 | 0.192 | -0.001 [-0.009, 0.007] |
| `reactive_baseline` | 100 | **0.415** | **0.868** | **0.603** | **0.812** | 0.089 | 0.190 | 0.008 [0.000, 0.016] |
| cell A mono ungated | 310 | 0.382 | 0.859 | 0.589 | 0.778 | 0.068 | 0.184 | -0.003 [-0.007, 0.002] |
| cell B mono gated | 310 | 0.375 | 0.833 | **0.593** | 0.763 | **0.061** | 0.187 | running |

Share of judged turns with IP > tau = 0.20 (the quantity the gate bounds): base 4.0 %, cell A 4.0 %,
**cell B 2.3 %**.

Basic qualities: 0.80-0.81 in every arm. No leaks of the hidden need (0 % in every arm). Median reply length:
26-30 words.

- **The reactive seeker makes the supporter look better on every quality scale.** That simulator answers
  each turn directly instead of withholding, so the supporter has more to work with: Success +0.03 and RAC
  +0.04, both outside the CIs. It also gets the most intrusive score (IP 0.089, against 0.065). A
  cooperative seeker lets the supporter push further, and the judge marks that as intrusive. This is the
  reason the main arms use the withholding simulator: the reactive one flatters every system.
- **Base, untuned: Success 0.383, IP 0.065, counterfactual PRI about 0.** This is the reference every later
  cell is compared against. A counterfactual PRI of 0 means its turns draw no more pushback than a neutral
  reflection would.
- **Cell A matches the base model** (0.382 vs 0.383 Success, 0.068 vs 0.065 IP). It is the same untuned
  monolithic supporter run over 2-4 linked sessions instead of one, so this shows that the extra sessions
  alone do not change the scores, and its counterfactual PRI is zero too (-0.003 [-0.007, 0.002]). Cells B-D
  change one factor each against this cell.
- **Cell B (adding the gate) cuts intrusiveness at a small cost in listening.**
  - Intrusive turns (IP > 0.20) drop from 4.0 % to 2.3 %, and mean IP from 0.068 to 0.061 (CIs [0.063, 0.072]
    and [0.057, 0.064] barely touch).
  - Success is unchanged (0.375 vs 0.382, CIs overlap).
  - Active listening falls (AELS 0.859 to 0.833, outside the CIs) and RAC slightly (0.778 to 0.763); comforting
    (CRS) is unchanged or slightly up (0.593 vs 0.589).
  - Replies are shorter (median 25 words, against 30), and part of the AELS drop may be that, because a
    reworded or fallback reply is plainer.
- **The gate fires much more on the test profiles than calibration predicted.** Only 53.5 % of turns pass
  as drafted; 36.8 % are revised and 9.6 % fall back to a plain reflection. Calibration expected about 73 %
  released at alpha = 0.10. The gate was calibrated on cell C (the decomposed pipeline) and is applied here to
  the monolithic one, whose drafts score as riskier. So the guarantee (at most 10 % intrusive released turns)
  holds by a wide margin (2.3 % measured), but it costs more revisions than planned. Cell D (decomposed +
  gate) is the matched setting.

Figures (`extra/baseline_plots/`):
- `scores_by_arm.png`: one panel per judge scale.
- `score_distributions.png`: per-dialogue spread.
- `ip_per_turn.png`: intrusiveness by turn position.
- `pri_counterfactual.png`: factual vs neutral control.
- `dialogue_diagnostics.png`: length, leaks, gate paths.

`two_by_two.png` appears once cells A-D are all scored.

---

## 8. Our need-state memory: Outcome 4, first arm (`memory_1000/`, lab RTX 3060)

`mem_needstate`: untuned Qwen2.5-7B supporter, decomposed (Analyzer, then Strategist, then Responder), no gate, with
**our need-state memory**:
- a graph of hypothesised needs per profile, with a status (hypothesis / confirmed / disconfirmed), a
  confidence, and the quoted turns that support or contradict each one;
- the graph is carried across 2-4 linked sessions;
- the seeker profile advances between sessions (needs get resolved, intensify or are displaced);
- the simulator is the corrective one (it pushes back on wrong guesses).

100 test profiles, 310 sessions. Run time: 5.5 h rollout plus 0.8 h judging.

| Metric | mem_needstate |
|---|---|
| Success (all sessions) | 0.381 |
| IP (lower better) | 0.095 |
| PRI observed | 0.213 |
| Memory nodes per profile | 6.3 (3-23) |
| Node status at the end | 347 confirmed, 286 hypothesis, **0 disconfirmed** |
| Nodes with any contradicting quote | **0** |

Success by time since the previous session (`extra/memory_eval/`):

| session 1 | <= 7 days | 7-21 days | > 21 days |
|---|---|---|---|
| 0.382 [0.362, 0.405] (n=100) | 0.387 [0.367, 0.410] (n=114) | 0.383 [0.359, 0.411] (n=47) | 0.361 [0.344, 0.380] (n=49) |

**What this shows, and what it cannot show yet:**
- **Success holds across sessions and drops a little only after gaps of more than 3 weeks** (0.361 against
  0.382-0.387; the CIs touch). We cannot yet say whether the memory causes this. The comparison that decides
  it is `mem_none` (same setup, no memory), which runs next. The other arms (base, cell A) use a different
  simulator and no profile advance, so their numbers are a context, not a control.
- **The memory never disconfirms anything.** Not one of 633 nodes was contradicted or disconfirmed. The
  lexical denial detector also found 0 explicit seeker denials in 310 sessions. So the key feature of
  need-state memory (block re-proposing a need the seeker rejected) **was never triggered** in this run, and
  the re-proposal metric is 0 / 0 by default, not a win.
- **Many nodes are generic, but the true need is often in memory and ranked below a generic one.** The most
  common nodes are "validation" (35), "need for validation" (29), "support" (25) and "need for support" (22).
  In two 4-session profiles traced turn by turn, the hidden need is stored and confirmed:
  - p000008: "I want to feel more secure and stable" is stored as "Need for security", confirmed in session 3.
  - p000394: "I need to feel competent and capable" is stored as "Self-efficacy", confirmed in session 3.

  But the brief handed to the next session still names the generic "Validation":
  - p000394: the Analyzer labelled "Self-efficacy" depth 3, and `best_terminal` only looks at depth-2 nodes.
  - p000008: two confirmed nodes tie at 0.85, and the tie goes to whichever node came first.

  The judge-labelled match from the lab's memory-calibration step (section 9) gives the rate over all
  profiles.
- **Why nothing is disconfirmed or resolved** (`src/memory.py`, `AgentPipeline.update_memory`):
  - A disconfirmation needs two things. The Analyzer must mark a need `disconfirmed`: 5 of 7,863 need
    outputs. And that need's text must match a stored node (Jaccard of words with 4+ letters >= 0.6): none did.
  - `resolve`, `reinstate` and `reopen` exist in `memory.py`, but the pipeline never calls them. So the
    profile's resolved, intensified and displaced transitions never change the memory.
  - Nodes are never linked (`parent_id` is empty on all 633). The stored "chain" is a set of depth-labelled
    needs, not a path.
- **IP 0.095 and PRI 0.213 are higher than the base model's 0.065 / 0.192.** Part of this is the corrective
  simulator, which pushes back more often by design. Whether the memory itself adds intrusiveness again
  depends on `mem_none`.

Figure: `extra/memory_eval/success_by_gap.png` (Success per gap bucket with 95 % CIs; the later memory arms
are added to the same plot).

---

## 9. Still running or queued on the lab RTX 3060

The queue is `scripts/sop_queue.sh`; each step resumes where it stopped. Estimates use this run's actual
pace, about 2x faster than the earlier estimate.

| Step | What it adds | Expected |
|---|---|---|
| Rest of section 7: cell B counterfactual PRI, cells C, D, report | the 2 x 2: does decomposition or the gate lower IP / PRI without costing Success? | about 7 Oct night to 8 Oct morning |
| `mem_none`, `mem_summary`, `mem_dense`, `mem_event` | the controls for section 8: does need-state memory beat no memory and the three standard memories? | about 6 h each |
| `memory_eval --judge`, `memory_calibration` | transition recall, abstention on ambiguous profiles, calibration of the memory's confidence (Brier, ECE, AUROC) | after the memory arms |
| ES-MemEval QA | memory question answering on real multi-session data, per memory type | about 6 h |
| alpha sweep (Outcome 3) | gate at alpha 0.05 and 0.20 on the test profiles, next to cell D at 0.10 | about 7-8 h |

The memory-calibration table is produced only by the lab run (the Ollama judge). A local run without
`--backend` uses the echo stub and is meaningless; `scripts/collect_results.sh` leaves it out.

---

## 10. Reproduce or refresh

```
./scripts/local_analyses.sh      # laptop: lora_geometry --truncate, pvi, scaling for every complete size
./scripts/collect_results.sh     # copy everything new from runs/ into this folder
python extra/test_extra.py       # 25 checks of the analysis code, no GPU
```
All numbers come from simulated seekers and an LLM judge; the judge-human agreement study is still open.
