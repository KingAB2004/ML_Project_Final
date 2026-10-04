"""Corpus statistics of the generated dataset (CPU): counts, session structure, turns, gaps, resistance mix.

  python extra/dataset_stats.py [--out results/v3_full1000/dataset/dataset_stats.json]
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from _shared import ROOT, read_jsonl, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Statistics of data/ (profiles, corpus, SFT files).")
    ap.add_argument("--out", default="results/v3_full1000/dataset/dataset_stats.json")
    args = ap.parse_args()
    d = ROOT / "data"
    profiles = read_jsonl(d / "profiles" / "profiles.jsonl")
    split = {s: len(read_jsonl(d / "profiles" / f"profiles_{s}.jsonl")) for s in ("train", "val", "test", "calibration")}
    generated = read_jsonl(d / "corpus" / "sessions.jsonl")
    kept = read_jsonl(d / "corpus" / "sessions_annotated.jsonl")
    rejected = read_jsonl(d / "corpus" / "sessions_rejected.jsonl")
    per_profile = Counter(s["profile_id"] for s in kept)
    turns = [len(s["turns"]) for s in kept]
    gaps = [s["gap_days_from_prev"] for s in kept if s.get("gap_days_from_prev")]
    strategies = [a.get("strategy", "") for a in read_jsonl(d / "corpus" / "annotations.jsonl")]
    sft = {f.stem: sum(1 for _ in open(f, encoding="utf-8")) for f in sorted((d / "sft").glob("*.jsonl"))}
    stats = {
        "seeds": len(read_jsonl(d / "seeds" / "seeds.jsonl")),
        "profiles": len(profiles), "profile_splits": split,
        "resistance": dict(Counter(p.get("persona_hidden", {}).get("resistance_level") for p in profiles)),
        "problem_types": dict(Counter(p.get("problem_type") for p in profiles).most_common()),
        "sessions_generated": len(generated), "sessions_rejected": len(rejected), "sessions_kept": len(kept),
        "profiles_with_dialogues": len(per_profile),
        "sessions_per_profile": dict(sorted(Counter(per_profile.values()).items())),
        "sessions_by_index": dict(sorted(Counter(s.get("session_index") for s in kept).items())),
        "turns_per_session": {"mean": statistics.fmean(turns), "min": min(turns), "max": max(turns), "total": sum(turns)},
        "words_in_dialogues": sum(len(t["text"].split()) for s in kept for t in s["turns"]),
        "gap_days": {"n": len(gaps), "median": statistics.median(gaps), "min": min(gaps), "max": max(gaps)},
        "annotations": len(strategies), "unique_strategy_phrases": len({s.strip().lower() for s in strategies}),
        "sft_examples": sft,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, stats)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
