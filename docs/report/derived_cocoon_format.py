"""Derived analysis for the project paper: model results in the format of COCOON's Tables 3 and 4.

Success plus the six basic metrics (0-100) and the CRS / RAC dimensions (1-7) per model, aggregated from the saved
judge ratings with the project's own grouping code (src/metrics.py). Re-runs no model. Each table compares only
systems scored on the same judge machine.

    python3 docs/report/derived_cocoon_format.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from metrics import basic_breakdown, crs_dimensions, rac_dimensions  # noqa: E402

R = ROOT / "results" / "v3_full1000"
TABLES = {
    "our test people (COCOON Table 3 format)": [
        ("Base (untuned 7B)", "baselines_1000/scores/base_instruct"),
        ("FT-7B, with thoughts", "test_eval/qwen2.5_7B/scores/sft_with_thoughts"),
        ("FT-7B, without thoughts", "test_eval/qwen2.5_7B/scores/sft_wo_thoughts"),
    ],
    "ExTES people (COCOON Table 4 format; Success scored without a reference need)": [
        ("Ours-trained 7B", "corpus_extes/scores/corpus_ours_extes_profiles"),
        ("ExTES-trained 7B", "corpus_extes/scores/corpus_extes_extes_profiles"),
        ("Cell D (four agents + gate)", "baselines_1000/scores/cellD_extes"),
    ],
}


def load(path: Path) -> list[dict]:
    return [r for r in (json.loads(line) for line in path.open() if line.strip()) if not r.get("parse_failed")]


def main() -> None:
    for title, arms in TABLES.items():
        print(title)
        print(f"{'system':30s} {'SR':>6s} {'Flu':>5s} {'Div':>5s} {'Emp':>5s} {'Inf':>5s} {'Hum':>5s} {'Skil':>5s} {'Avg':>5s} "
              f"{'Aff':>5s} {'Neg':>5s} {'Sup':>5s} {'Man':>5s}")
        for label, rel in arms:
            d = R / rel
            b = basic_breakdown(load(d / "basic.jsonl"))
            c = crs_dimensions(load(d / "crs.jsonl"))
            r = rac_dimensions(load(d / "rac.jsonl"))
            s = load(d / "success.jsonl")
            sr = sum(x["normalized"] for x in s) / len(s)
            vals = [b[f"{k}_100"] for k in ("fluency", "diversity", "empathy", "information", "humanoid", "skillfulness")]
            print(f"{label:30s} {sr:6.3f} " + " ".join(f"{v:5.1f}" for v in vals) + f" {b['basic_avg_100']:5.1f} "
                  f"{c['affective_improvement']:5.2f} {c['negative_helper']:5.2f} {r['supportiveness']:5.2f} {r['management']:5.2f}")
        print()


if __name__ == "__main__":
    main()
