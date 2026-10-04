"""Phase 1.7 - content-level annotation (PLAN Sec. 8.7).

Free-form Analysis and Strategy phrases, NOT a fixed atomic taxonomy: the content level is the baseline
paper's claimed advantage over ESConv and ExTES, so it is inherited deliberately rather than replaced.

The consistency pass matters as much as the annotation: if three phrases cover most Strategy labels the
fine-tuning signal is gone, and no aggregate metric will say so.

  python src/annotate.py --backend echo
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common import (DATA, fill, read_prompt, read_jsonl, render_transcript, to_int,
                    write_jsonl)
from llm import LLM, pmap
from memory import similarity

FILTERED_PATH = DATA / "corpus" / "sessions_filtered.jsonl"
ANNOTATED_PATH = DATA / "corpus" / "sessions_annotated.jsonl"
ANNOTATIONS_PATH = DATA / "corpus" / "annotations.jsonl"


def profile_block(profile: dict) -> str:
    keys = ("emotion", "feeling", "terminal_need", "resistance_level")
    return json.dumps({k: profile.get(k) for k in keys}, ensure_ascii=False)


def text_field(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v).strip() for v in value if v)
    return json.dumps(value, ensure_ascii=False) if value else ""


def annotate_session(llm: LLM, session: dict) -> list[dict]:
    template = read_prompt("annotate_turn.md")
    profile = session.get("profile_snapshot", {})
    records = []
    turns = session["turns"]
    for turn in turns:
        if turn.get("role") != "supporter" or turn.get("meta", {}).get("source") == "opener_pool":
            continue
        prefix = [t for t in turns if t["turn_index"] <= turn["turn_index"]]
        out = llm.structured(
            fill(template, profile_block=profile_block(profile),
                 history=render_transcript(prefix), turn_text=turn["text"]),
            required=("analysis", "strategy"),
            max_tokens=512,
        )
        rungs = set(re.findall(r"L[0-3]", str(out.get("ladder_rung", "")).upper()))
        rung = rungs.pop() if len(rungs) == 1 else ""      # "L0|L1|L2|L3" copied back is no answer
        rec = {
            "session_id": session["session_id"],
            "turn_index": turn["turn_index"],
            # Model JSON is untrusted: a list or dict where a phrase belongs becomes text, not a crash.
            "analysis": text_field(out.get("analysis")),
            "strategy": text_field(out.get("strategy")),
            "ladder_rung": rung or turn.get("ladder_rung", "L0"),
            "user_disclosure_depth": to_int(out.get("user_disclosure_depth"), 0, 0, 3),
            "annotator": llm.spec["model"],
            "parse_failed": bool(out.get("parse_failed")),
        }
        records.append(rec)
    return records


def apply_annotations(session: dict, records: list[dict]) -> dict:
    by_index = {r["turn_index"]: r for r in records}
    depth_so_far = 0
    for turn in session["turns"]:
        rec = by_index.get(turn["turn_index"])
        if rec and turn.get("role") == "supporter":
            turn["analysis"] = rec["analysis"]
            turn["strategy"] = rec["strategy"]
            turn["ladder_rung"] = rec["ladder_rung"]
        if rec:
            depth_so_far = max(depth_so_far, rec["user_disclosure_depth"])
        if turn.get("role") == "user":
            turn["disclosure_depth"] = depth_so_far
    session["annotations_complete"] = bool(records) and all(not r["parse_failed"] for r in records)
    return session


def vocabulary_report(records: list[dict], top_n: int = 20) -> dict:
    """Degeneracy check: how much of the label space the largest clusters cover."""
    strategies = [r["strategy"].strip().lower() for r in records if r.get("strategy")]
    counts = Counter(strategies)
    clusters: list[list[str]] = []
    for phrase, _ in counts.most_common():
        for cluster in clusters:
            if similarity(cluster[0], phrase) >= 0.6:
                cluster.append(phrase)
                break
        else:
            clusters.append([phrase])
    total = sum(counts.values()) or 1
    sizes = sorted((sum(counts[p] for p in c) for c in clusters), reverse=True)
    return {
        "n_labels": total,
        "n_unique": len(counts),
        "n_clusters": len(clusters),
        "top3_share": sum(sizes[:3]) / total,
        "top_clusters": [{"size": s} for s in sizes[:top_n]],
        "degenerate": (sum(sizes[:3]) / total) > 0.8,
    }


def run(sessions_path: Path = FILTERED_PATH, out_path: Path = ANNOTATED_PATH,
        backend: str | None = None, limit: int | None = None) -> dict:
    sessions = read_jsonl(sessions_path)
    if limit:
        sessions = sessions[:limit]
    llm = LLM("judge", backend=backend)
    all_records, out_sessions = [], []
    try:
        for session, records in zip(sessions, pmap(lambda s: annotate_session(llm, s), sessions, llm)):
            all_records.extend(records)
            out_sessions.append(apply_annotations(session, records))
    finally:
        llm.release()
    # Rewritten, not appended: a rerun replays from the LLM cache and must not duplicate every record.
    write_jsonl(ANNOTATIONS_PATH, all_records)
    write_jsonl(out_path, out_sessions)
    vocab = vocabulary_report(all_records)
    return {"sessions": len(out_sessions), "turns_annotated": len(all_records), "vocabulary": vocab}


def main() -> None:
    ap = argparse.ArgumentParser(description="Content-level Analysis/Strategy annotation.")
    ap.add_argument("--sessions", default=str(FILTERED_PATH))
    ap.add_argument("--out", default=str(ANNOTATED_PATH))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    stats = run(Path(args.sessions), Path(args.out), args.backend, args.limit)
    v = stats["vocabulary"]
    print(f"annotated {stats['turns_annotated']} turns in {stats['sessions']} sessions; "
          f"{v['n_unique']} unique strategy phrases in {v['n_clusters']} clusters, "
          f"top-3 share {v['top3_share']:.0%}"
          + ("  [DEGENERATE - diversify the few-shot examples and re-run]" if v["degenerate"] else ""))


if __name__ == "__main__":
    main()
