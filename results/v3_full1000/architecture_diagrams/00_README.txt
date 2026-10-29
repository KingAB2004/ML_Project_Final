COCCON overall architecture (one diagram)

COCCON_architecture.png   2x resolution, for slides
COCCON_architecture.svg   vector, scales / editable

Sections: 1 dataset creation (with the planned memory-aware data path: replay through memory, record the brief,
re-annotate later sessions), 2 fine-tuning, 3 supporter at inference (Analyzer, pacing cap, Strategist around the
fine-tuned Responder, Critic, calibrated gate, one-call mode), 4 need-state memory (structure, node lifecycle,
across sessions). Solid = built, dashed = planned. No numbers by design.
Regenerate: cd source && python architecture.py ..   (needs google-chrome for the PNG)
