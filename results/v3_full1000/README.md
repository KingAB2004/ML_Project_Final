# v3 results: the 1000-profile corpus, fine-tuning and the scaling study

This folder holds every result of the full-size run: the 1000-profile dataset built on the lab server, the
fine-tuning runs, and the analyses run on top of them (GPU and CPU). Nothing here is hand-edited: it is
copied from `runs/` (which git ignores) by `scripts/collect_results.sh`, and refreshed by rerunning that
script as each pending step finishes. Model weights are not included (too large for git).

**Status on 4 October 2026, 16:00 IST.** Dataset: done. Qwen2.5-0.5B fine-tune (both arms) and its analyses:
done. Qwen2.5-7B fine-tune: running on the lab server. Qwen2.5-3B fine-tune: not started (run
`scripts/scale_3b.sh`). Test-profile evaluation of the fine-tuned models: queued on the lab server.

| Folder | What | Where it ran |
|---|---|---|
| `dataset/` | corpus statistics, the generation log, the pipeline state | lab server, RTX 3060 (Ollama) + CPU |
| `training/` | loss curves, best checkpoints, adapter configs, cleaned training log | laptop RTX 4060 (0.5B); lab RTX 3060 (7B, pending) |
| `analyses/pvi/` | usable information in the thoughts | laptop RTX 4060 (forward passes) |
| `analyses/lora_geometry/` | SVD geometry of the LoRA updates, rank truncation | CPU (SVD) + laptop RTX 4060 (truncation) |
| `analyses/scaling/` | scaling-law table and training curves | CPU |
| `test_eval/` (appears later) | Success / IP / PRI of every fine-tuned size on the 100 test profiles | lab RTX 3060 |

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
Plot: `analyses/scaling/training_curves.png`. Files: `training/0.5B/train_summary_*.json`,
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

### Qwen2.5-7B-Instruct (lab RTX 3060): running
`with_thoughts` started 4 Oct 08:16 (about 25 h, ETA Mon 5 Oct ~10:00), then `wo_thoughts` (ETA Tue 6 Oct
~08:00). 1,177 steps per epoch at about 24.7 s/step. Results will appear in `training/7B/`.

### Qwen2.5-3B-Instruct (laptop): not started
`scripts/scale_3b.sh`, about one day for both arms. Results will appear in `training/3B/`.

---

## 3. Usable information in the thoughts (`analyses/pvi/`, GPU)

Question: how much do the annotated Analysis and Strategy tell the model about the reply?
PVI per validation turn = log2 p_with(reply | context, thoughts) - log2 p_wo(reply | context), using the
with_thoughts adapter (teacher-forced through the gold thoughts) and the wo_thoughts adapter (Ethayarajh et
al. 2022). Both score the identical `<response>...</response>` tokens. Mean PVI = usable information in bits.
2,608 validation turns, 97 profiles, about 5 minutes on the RTX 4060. Script: `extra/pvi.py`.

| Size | V-information (bits / turn) [95 % CI] | bits / token | PVI < 0 | reply NLL with thoughts | reply NLL without | Spearman(thought length, PVI) |
|---|---|---|---|---|---|---|
| 0.5B | **8.91 [8.49, 9.34]** | 0.24 | 17.9 % | 0.727 | 0.891 | 0.16 |

By ladder rung: L1 (gentle) 7.87 bits (n = 1,160), L2 (deeper) 9.76 bits (n = 1,444), L0 -0.22 (n = 4).

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
- **Longer thoughts help only a little** (Spearman 0.16): content matters more than length.

Figures: `pvi_hist.png` (distribution, mostly positive with a long right tail), `pvi_by_size.png`.

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

Overlap of the two arms' top directions phi(k) (1 = same subspace): k=1 0.074, k=4 0.066, k=8 0.077, against
0.003 / 0.011 / 0.022 for random subspaces.

| Rank kept k | 0 (base) | 1 | 2 | 4 | 8 | 16 | 32 (full) | k for 95 % of the gain |
|---|---|---|---|---|---|---|---|---|
| with_thoughts val loss | 1.868 | 1.367 | 1.208 | 1.054 | 0.933 | 0.864 | 0.848 | 16 |
| wo_thoughts val loss | 1.874 | 1.350 | 1.183 | 1.048 | 0.951 | 0.896 | 0.895 | 16 |

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

Figures: `effective_rank.png` (layer x module heatmap), `norm_and_similarity_by_depth.png`,
`rank_truncation.png`. Per-matrix numbers: `per_matrix.jsonl`.

---

## 5. Scaling study (`analyses/scaling/`, CPU)

`extra/scaling.py` combines the training logs, the PVI reply losses and (later) the test-profile scores of
every size into one table, fits L(N) = a N^-b over the sizes present, and measures the thoughts gain per size.
Right now only 0.5B is complete, so the table has one row per arm and no fit yet; it fills in automatically
as 7B (Tuesday) and 3B (when run) finish, and the test columns (Success, IP, PRI, counterfactual PRI) fill in
from the lab-server evaluation. Figures: `training_curves.png`, `scaling_loss.png` (more once more sizes exist).

---

## 6. Reproduce or refresh

```
./scripts/local_analyses.sh      # laptop: lora_geometry --truncate, pvi, scaling for every complete size
./scripts/collect_results.sh     # copy everything new from runs/ into this folder
python extra/test_extra.py       # 24 checks of the analysis code, no GPU
```
All numbers come from simulated seekers and an LLM judge; the judge-human agreement study is still open.
