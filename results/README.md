# Results

Every result of the project, one folder per run. Each folder has its own `README.md` with the setup, the
numbers, and why they came out as they did.

| Folder | Run | Corpus | What it contains | Status |
|---|---|---|---|---|
| [`v1/`](v1/README.md) | first baseline run (Phase A), 2 Oct 2026 | none (untuned base model) | evidence of the bugs only (broken dialogues, gate calibration, run log, two diagnostic plots) and the list of fixes; no results | superseded by v2 |
| [`v2/`](v2/README.md) | 50-profile end-to-end run (Phases A, B, C), 3 Oct 2026 | 50 profiles, 140 sessions | all baselines, the 2 x 2 (decomposition x gate), conformal gate calibration, 7B QLoRA fine-tune and its test | complete |
| [`v3_full1000/`](v3_full1000/README.md) | full-size run, 3-6 Oct 2026 | 1,000 profiles, 2,369 sessions | dataset statistics, fine-tuning curves (all 6 runs done), PVI (0.5B, 3B), LoRA geometry, test-profile scores (partial), PVI, LoRA geometry, scaling study | in progress; refresh with `scripts/collect_results.sh` |

## v1 or v2: which to use

**Use v2.** v1 was stopped on purpose because its dialogues were broken (the supporter's private analysis was
visible in 28-42 % of single-model turns, the critic invented violations so the gate rewrote 72 % of cellB, the
seeker copied the supporter, reply lengths differed 4x between architectures). v2 re-ran everything with those
bugs fixed (through B50), scored every arm, calibrated a gate that actually fires, and added the corpus and the
fine-tuned model.

`v1/` now holds only the evidence of those bugs: the 432 broken dialogues, its gate calibration (critic IP 0 on
all 98 turns, lambda_hat = 1), the run log, and `dialogue_diagnostics.png` / `gate_calibration.png`. Its scores,
memory logs, call log, configs and data copies were removed (git history keeps them; the seeds and profiles are
identical to `v2/data/`).

**The baseline plots to report: [`v2/runs/v3_50/extra/baseline_plots/`](v2/runs/v3_50/extra/baseline_plots/)**
(`python extra/baseline_plots.py --run <run>`, profile-bootstrap 95 % CIs).

What v2 does not have either:
- `sft_wo_thoughts`: not trained in v2; trained in `v3_full1000/`.
- The alpha-sweep arms (gate at alpha 0.05 / 0.2): v2 has the calibration only; the evaluated arms run in the
  1000-profile queue.
- ExTES distribution shift and the memory-arm comparison: queued for the 1000-profile run.

Headline findings so far:
- **v2:** the pipeline works end to end (0 hidden-need leaks in 6,449 supporter turns). The conformal gate cuts
  intrusiveness by 25-35 % for about 0.02 Success; decomposition alone adds nothing measurable. The fine-tuned
  7B matches the base model on Success but is pushier (IP 0.103 vs 0.082). n = 40 test profiles, so most
  differences sit inside their CIs.
- **v3:** a 2,369-session English multi-session corpus with 21,429 annotated supporter turns. On 0.5B the gold
  thoughts carry about 8.9 bits of usable information per reply, the LoRA updates concentrate in the later
  layers and need about rank 16 for 95 % of their gain, and a third epoch overfits.
