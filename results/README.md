# Results

Every result of the project, one folder per run. Each folder has its own `README.md` with the setup, the
numbers, and why they came out as they did.

| Folder | Run | Corpus | What it contains | Status |
|---|---|---|---|---|
| [`v1/`](v1/README.md) | first baseline run (Phase A), 2 Oct 2026 | none (untuned base model) | partial baseline scores, the bugs they exposed and the fixes | superseded by v2; kept as evidence |
| [`v2/`](v2/README.md) | 50-profile end-to-end run (Phases A, B, C), 3 Oct 2026 | 50 profiles, 140 sessions | all baselines, the 2 x 2 (decomposition x gate), conformal gate calibration, 7B QLoRA fine-tune and its test | complete |
| [`v3_full1000/`](v3_full1000/README.md) | full-size run, 3-6 Oct 2026 | 1,000 profiles, 2,369 sessions | dataset statistics, fine-tuning curves (0.5B done; 3B and 7B with_thoughts done, wo_thoughts running), PVI, LoRA geometry, scaling study | in progress; refresh with `scripts/collect_results.sh` |

Headline findings so far:
- **v2:** the pipeline works end to end (0 hidden-need leaks in 6,449 supporter turns). The conformal gate cuts
  intrusiveness by 25-35 % for about 0.02 Success; decomposition alone adds nothing measurable. The fine-tuned
  7B matches the base model on Success but is pushier (IP 0.103 vs 0.082). n = 40 test profiles, so most
  differences sit inside their CIs.
- **v3:** a 2,369-session English multi-session corpus with 21,429 annotated supporter turns. On 0.5B the gold
  thoughts carry about 8.9 bits of usable information per reply, the LoRA updates concentrate in the later
  layers and need about rank 16 for 95 % of their gain, and a third epoch overfits.
