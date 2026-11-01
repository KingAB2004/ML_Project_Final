"""Derived analysis for the project report (Appendix B.6): a fraction-based Success Rate.

The judge rates need identification on a 1-7 scale (prompts/scale_success.md) and the pipeline reports the mean
normalised to 0-1. The SOP described Success Rate as a fraction of dialogues that identify the need. This script
reads the saved judge ratings (it re-runs no model) and reports, per arm, the share of sessions rated >= 5
("articulated the underlying need, partially or hedged" or better), with a 95 % bootstrap interval over people.

    python3 docs/report/derived_success_rate.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "v3_full1000"
ARMS = [
    ("untuned 7B, one call, proactive seeker (base_instruct)", "baselines_1000/scores/base_instruct"),
    ("untuned 7B, one call, reactive seeker (reactive_baseline)", "baselines_1000/scores/reactive_baseline"),
    ("cell A: one call, gate off", "baselines_1000/scores/cellA_mono_ungated"),
    ("cell B: one call, gate on", "baselines_1000/scores/cellB_mono_gated"),
    ("cell C: four agents, gate off", "baselines_1000/scores/cellC_dec_ungated"),
    ("cell D: four agents, gate on", "baselines_1000/scores/cellD_dec_gated"),
    ("step 12: cell D + fine-tuned Generator + memory", "sft_agents/scores/sft_dec_gated_mem"),
    ("fine-tuned 7B, with thoughts (single session)", "test_eval/qwen2.5_7B/scores/sft_with_thoughts"),
    ("fine-tuned 7B, without thoughts (single session)", "test_eval/qwen2.5_7B/scores/sft_wo_thoughts"),
    ("fine-tuned 7B + need-state memory (multi-session)", "sft7b_multi/scores/sft7b_multi_mem"),
    ("fine-tuned 7B, no memory (multi-session)", "sft7b_multi/scores/sft7b_multi_nomem"),
    ("memory: none", "memory_v2/scores/mem_none"),
    ("memory: flat summary", "memory_v2/scores/mem_summary"),
    ("memory: dense retrieval", "memory_v2/scores/mem_dense"),
    ("memory: need-state", "memory_v2/scores/mem_needstate"),
    ("steering control (L12, c0)", "steering/scores/steer_mono_L12_c0"),
    ("activation steering (L12, c0.2)", "steering/scores/steer_mono_L12_c0.2"),
    ("SFT control", "preference/scores/pref_mono_sft"),
    ("DPO", "preference/scores/pref_mono_dpo"),
]
THRESHOLD = 5
REPS = 2000


def person(target: str) -> str:
    return target.split("-")[0]


def main() -> None:
    rng = random.Random(0)
    print(f"Share of sessions with judge Success rating >= {THRESHOLD} of 7 (95 % CI, {REPS} bootstrap resamples over people)")
    print(f"{'arm':58s} {'sessions':>8s} {'mean rating':>11s} {'share >= 5':>10s}  95% CI")
    for label, rel in ARMS:
        path = R / rel / "success.jsonl"
        rows = [json.loads(line) for line in path.open() if line.strip()]
        rows = [r for r in rows if not r.get("parse_failed")]
        by: dict[str, list[int]] = {}
        for r in rows:
            by.setdefault(person(r["target"]), []).append(int(r["mean"] >= THRESHOLD))
        hits = [h for hs in by.values() for h in hs]
        point = sum(hits) / len(hits)
        groups = list(by.values())
        boots = []
        for _ in range(REPS):
            sample = [h for g in (rng.choice(groups) for _ in groups) for h in g]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        mean_rating = sum(r["mean"] for r in rows) / len(rows)
        print(f"{label:58s} {len(rows):8d} {mean_rating:11.2f} {point:10.3f}  [{boots[int(.025 * REPS)]:.3f}, "
              f"{boots[int(.975 * REPS)]:.3f}]")


if __name__ == "__main__":
    main()
