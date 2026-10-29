# Need-State Observatory (demo)

A web demo of the system: a conversation, the need-state graph changing turn by turn beside it, an
inspector for each supporter turn (Analyzer, depth ladder, Strategist, Generator, Critic and gate), and the
project's results.

```
python3 demo/build_data.py                      # once: results/v3_full1000 -> demo/static/data/replay.json (CPU, ~15 s)
python3 demo/server.py                          # replay only, any machine -> http://127.0.0.1:8765
python3 demo/server.py --live --backend ollama  # + live chat on a GPU machine with Ollama
python3 demo/server.py --live --backend transformers \
    --adapter runs/full_1000/adapter_with_thoughts \
    --calibration results/v3_full1000/sft_agents/conformal/calibration.json   # live step 12 (fine-tuned Generator)
```

Standard library only (`pyyaml` and `numpy` as for the rest of `src/`). The page loads d3 and Google Fonts
from their CDNs.

## Views

- **Conversation, Replay.** Recorded test conversations from two systems: the full system (step 12: four
  agents, fine-tuned Generator, conformal gate, need-state memory) and the untuned four agents with
  need-state memory (`mem_needstate`). 20 people per system, picked for graphs that change, plus p000394.
  Play, step, scrub, or click a message. Seeker turns show the judge's PRI, supporter turns the judge's IP
  where it was sampled, the Critic's predicted IP, the rung, revisions or fallback, and how many beliefs
  changed.
- **Conversation, Live.** You type as the seeker, or pick a test person and let the simulated seeker
  talk. "Next session" applies the memory's decay over the gap you set, then opens a follow-up session.
  No judge runs live (it would have to share the GPU with the generator), so live turns show the Critic's
  prediction only.
- **Results.** Every arm's Success / IP / PRI / counterfactual PRI / AELS with bootstrap CIs over
  people, computed from the score files. Also: IP against Success, gate calibration, recovery against cost,
  the 42 Holm-corrected contrasts, the corpus, and a table of every number with its source folder.
- **How it works.** The pipeline, the ladder, the IP items, the memory rules, and the caveats.

## Where the graph comes from

`build_data.py` does not copy summaries. For each supporter turn it feeds the stored Analyzer output back
through `AgentPipeline.update_memory` on a fresh `NeedStateMemory`, and applies `decay` between sessions.
The page therefore shows the graph the system held at that turn. The final rebuilt graphs match the
memory files the runs saved: 725/725 nodes (step 12) and 785/785 (`mem_needstate`). The build prints
this check.

## Files

| File | What |
|---|---|
| `build_data.py` | replay data and arm statistics -> `static/data/replay.json` |
| `server.py` | static server plus `/api/status` and `/api/live/{reset,turn,session}` |
| `static/app.html`, `static/app.js` | the page; `app.html` is a fragment that `server.py` wraps in a document |
