# Memory metrics without a model (v2 run, CPU)

Two of the SOP's cross-session memory measures (Expected Outcome 4, PLAN Sec. 12.4), computed on the v2 run
(50-profile corpus, 40 test profiles) with `python extra/memory_eval.py --run results/v2/runs/v3_50`. CPU only,
about one second; no model call. The full-size memory arms (`mem_none`, `mem_summary`, `mem_dense`,
`mem_event`, `mem_needstate`) have not been run yet, so the only multi-session arms here are the 2 x 2 cells,
which all use the need-state memory.

| File | What |
|---|---|
| `report.md` | both tables below |
| `summary.json` | every number, per arm |
| `success_by_gap.png` | Success by gap bucket for the multi-session arms, with 95 % CIs |
| `denials.jsonl` | every detected denial with its claim and any re-proposals (empty in this run, see below) |

## 1. Success Rate by time since the previous session

| arm | session 1 | <= 7 days | 7-21 days | > 21 days |
|---|---|---|---|---|
| cellA_mono_ungated | 0.396 (n=40) | 0.368 (n=39) | 0.377 (n=27) | 0.367 (n=15) |
| cellB_mono_gated | 0.379 (n=40) | 0.343 (n=36) | 0.352 (n=27) | 0.352 (n=18) |
| cellC_dec_ungated | 0.379 (n=40) | 0.378 (n=37) | 0.393 (n=25) | 0.360 (n=19) |
| cellD_dec_gated | 0.371 (n=40) | 0.342 (n=40) | 0.377 (n=19) | 0.364 (n=22) |

(95 % CIs, bootstrap over profiles, are in `report.md`; the single-session arms have only the first column.)

**What it shows, and why.**
- **No measurable effect of the gap.** Follow-up sessions score about the same as first sessions at every gap;
  every CI overlaps. Memory neither visibly helps nor decays with time at this sample size.
- **The judge's Success scale has almost no resolution here.** 0.333 is a rating of 3 of 7, and many lower CI
  bounds sit exactly on it: the judge gives 3 to most sessions, so small real differences cannot show. This is
  the "judge ceiling/floor" limitation in `BUGS.md`; Success Rate needs more profiles (the 1000-profile run)
  or a finer instrument to separate arms.
- **Cells A and B have no memory in practice.** Their memory files hold 0 nodes: the monolithic listener reads
  the memory brief but has no Analyzer to write beliefs, so `memory: needstate` is inert for them. Only the
  decomposed cells C and D built a memory (239 and 242 nodes). Their follow-up sessions are not better than
  A/B's either.

## 2. Re-proposal of denied inferences

| arm | memory | denials detected | re-proposed | memory nodes | disconfirmed nodes |
|---|---|---|---|---|---|
| base_instruct | none | 0 | 0 | 0 | 0 |
| reactive_baseline | none | 0 | 0 | 0 | 0 |
| sft_with_thoughts | none | 0 | 0 | 0 | 0 |
| cellA_mono_ungated | needstate | 0 | 0 | 0 | 0 |
| cellB_mono_gated | needstate | 0 | 0 | 0 | 0 |
| cellC_dec_ungated | needstate | 0 | 0 | 239 | 0 |
| cellD_dec_gated | needstate | 0 | 0 | 242 | 0 |

**What it shows, and why.**
- **The probe is never triggered: the simulated seekers do not deny inferences.** Out of 6,051 seeker turns,
  only 11 open with a denial ("not really", "it's not that", "that's not really an option"), and inspecting all
  11, every one answers a question ("do you feel like talking?") or rejects *advice* ("it's not that simple").
  None rejects an inference about the seeker's feelings or needs, although supporters made 101 such inferences.
  The seekers resist by deflecting and staying vague, not by saying "that's not it".
- **So the memory never records a disconfirmation** (0 disconfirmed nodes in 481), and the mechanism the SOP
  argues for (block a denied inference from coming back) is never exercised. Re-proposal is 0 for every arm,
  with or without memory, which says nothing about the memory.
- PLAN Sec. 12.4 anticipated exactly this: "if the baselines also score zero the probe was too easy and needs
  harder denial scripting". To test the mechanism, the seeker prompt (`prompts/user_proactive.md`) needs a
  scripted chance to reject a wrong interpretation explicitly (e.g. profiles whose terminal need contradicts the
  obvious reading), then the memory arms must be rerun.

The detector is lexical: an inference sentence (hedge or state attribution introducing a want, need or cause,
`grounding`'s cue lists) followed by a seeker reply that opens with a denial cue; a later supporter turn holding
60 % of the claim's content words is a re-proposal. It misses paraphrased denials, so it is a lower bound. The
11 candidate denials were checked by hand (listed above), so the zero here is not a detector artefact.
