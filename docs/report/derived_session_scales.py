"""Derived analysis for the project report (Appendix B.8): basic metrics, CRS and RAC per evaluated system.

These session scales were scored by the judge for the evaluated arms, but the result summaries report only
Success, AELS, IP and PRI. This script aggregates the saved judge ratings with the project's own grouping code
(src/metrics.py: basic_breakdown, crs_dimensions, rac_dimensions); it re-runs no model. Arms without a saved
score file are skipped.

    python3 docs/report/derived_session_scales.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from metrics import basic_breakdown, crs_dimensions, rac_dimensions  # noqa: E402

R = ROOT / "results" / "v3_full1000"
ARMS = [
    ("untuned, one call, proactive seeker", "baselines_1000/scores/base_instruct"),
    ("untuned, one call, reactive seeker", "baselines_1000/scores/reactive_baseline"),
    ("cell A", "baselines_1000/scores/cellA_mono_ungated"),
    ("cell B", "baselines_1000/scores/cellB_mono_gated"),
    ("cell C", "baselines_1000/scores/cellC_dec_ungated"),
    ("cell D", "baselines_1000/scores/cellD_dec_gated"),
    ("step 12", "sft_agents/scores/sft_dec_gated_mem"),
    ("fine-tuned 7B with thoughts (single)", "test_eval/qwen2.5_7B/scores/sft_with_thoughts"),
    ("fine-tuned 7B without thoughts (single)", "test_eval/qwen2.5_7B/scores/sft_wo_thoughts"),
    ("fine-tuned 7B + memory (multi)", "sft7b_multi/scores/sft7b_multi_mem"),
    ("fine-tuned 7B, no memory (multi)", "sft7b_multi/scores/sft7b_multi_nomem"),
    ("memory: none", "memory_v2/scores/mem_none"),
    ("memory: flat summary", "memory_v2/scores/mem_summary"),
    ("memory: dense", "memory_v2/scores/mem_dense"),
    ("memory: need-state", "memory_v2/scores/mem_needstate"),
    ("steering control", "steering/scores/steer_mono_L12_c0"),
    ("steered", "steering/scores/steer_mono_L12_c0.2"),
    ("SFT control", "preference/scores/pref_mono_sft"),
    ("DPO", "preference/scores/pref_mono_dpo"),
]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [r for r in (json.loads(line) for line in path.open() if line.strip()) if not r.get("parse_failed")]


def fmt(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> None:
    print("Session scales per system (judge Mistral-Nemo-12B; basic metrics rescaled to 0-100; CRS / RAC on 1-7;")
    print("Neg lower is better). Grouping: src/metrics.py.")
    print(f"{'system':42s} {'n':>4s} {'Basic Avg':>9s} {'Aff':>5s} {'Neg':>5s} {'Sup':>5s} {'Man':>5s}")
    for label, rel in ARMS:
        basic, crs, rac = (load(R / rel / f"{s}.jsonl") for s in ("basic", "crs", "rac"))
        if not (basic or crs or rac):
            print(f"{label:42s}  (no saved basic / CRS / RAC scores)")
            continue
        b = basic_breakdown(basic) if basic else {}
        c = crs_dimensions(crs) if crs else {}
        r = rac_dimensions(rac) if rac else {}
        n = max(len(basic), len(crs), len(rac))
        print(f"{label:42s} {n:4d} {fmt(b.get('basic_avg_100'), 1):>9s} {fmt(c.get('affective_improvement')):>5s} "
              f"{fmt(c.get('negative_helper')):>5s} {fmt(r.get('supportiveness')):>5s} {fmt(r.get('management')):>5s}")


if __name__ == "__main__":
    main()
