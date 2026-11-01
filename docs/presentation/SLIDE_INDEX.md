# COCCON_NEW presentation: slide index

`COCCON_NEW_presentation.pptx` (16:9, 15 slides, editable shapes, native charts and tables, source notes on every slide).
`COCCON_NEW_presentation.pdf` is a LibreOffice-rendered preview.

| # | Title | What the slide shows |
|---|---|---|
| 1 | COCCON_NEW: Proactive Emotional Support Across Sessions | Title, subtitle, authors and IDs (from the SOP), illustrative stock photo |
| 2 | What is the real problem? | Saved test excerpt (p000852): what was said, the hidden need, the risk of probing too hard |
| 3 | Earlier work and the original COCOON method | Reactive support (ESConv, ExTES) next to the COCOON pipeline, plus our English, open-weight setting |
| 4 | Gaps we targeted and our four additions | Three gaps linked to SOP enhancements E1–E4; supporting work listed separately |
| 5 | The corpus we built | Seeds, profiles, generated and kept sessions, annotated turns, sessions per position, profile splits |
| 6 | Architecture of our solution | Offline build and train, inference with four roles of one model, offline evaluation, model roles |
| 7 | How one supporter reply is produced | Analyzer, depth ladder, Strategist, Generator, Critic, calibrated gate, revise and fallback paths, memory update |
| 8 | Architecture of need-state memory | Evidence enters, stored belief fields, brief read by the agents; hidden profile kept separate |
| 9 | How a need belief changes across sessions | Belief lifecycle (implemented rules, schematic) and a session timeline |
| 10 | How we evaluated the system | Test setup, judge, Success / IP / PRI definitions and directions, uncertainty conventions |
| 11 | Ablation study: what does each component add? | 2 × 2 design (A–D) and the annotation and memory comparisons (design only) |
| 12 | Results: architecture and full system | Success, IP, observed PRI for A–D and the full system; gate violation rate with α and fallback shares |
| 13 | Results: fine-tuning and annotations | With vs without thoughts at 0.5B and 7B (3B not dialogue-tested), untuned 7B reference, PVI by size |
| 14 | Results: need-state memory | Untuned memory baselines, fine-tuned 7B with vs without memory, ES-MemEval QA accuracy |
| 15 | Results: comparison methods | Steering vs control, DPO vs SFT, ExTES-trained vs ours-trained, with Holm markers |

Slides 12–15 carry data and provenance notes only. Slides 1–11 carry optional speaking notes and sources.
