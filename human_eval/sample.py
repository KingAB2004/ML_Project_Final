"""Stratified subsample for human rating (PLAN Sec. 11.4 control 3).

~100 turns, stratified by arm, scripted resistance level, and judge IP decile, so the agreement figure is
not dominated by easy turns. Raters are blind to arm and to each other: the sheet carries the dialogue
prefix, the target turn, and nothing else.

  python human_eval/sample.py --run runs/<run_id> --n 100
"""
from __future__ import annotations

import argparse
import sys
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import read_jsonl, render_transcript, rng_for, write_json, write_jsonl  # noqa: E402


def decile(value: float) -> int:
    return min(9, max(0, int(float(value) * 10)))


def collect(run_dir: Path, metric: str = "ip") -> list[dict]:
    rows = []
    sessions = {}
    for path in (run_dir / "dialogues").glob("*.jsonl"):
        for s in read_jsonl(path):
            sessions[s["session_id"]] = s
    for path in (run_dir / "scores").rglob(f"{metric}.jsonl"):
        for r in read_jsonl(path):
            if r.get("parse_failed"):
                continue
            target = str(r.get("target", ""))
            sid, _, idx = target.partition("#")
            session = sessions.get(sid)
            if session is None or not idx:
                continue
            turn_index = int(idx)
            turns = session["turns"]
            prefix = [t for t in turns if t["turn_index"] < turn_index]
            target_turn = next((t for t in turns if t["turn_index"] == turn_index), None)
            if target_turn is None:
                continue
            rows.append({
                "item_id": f"{sid}#{turn_index}",
                "arm": session.get("arm", path.parent.name),
                "resistance_level": (session.get("profile_snapshot", {}) or {}).get("resistance_level",
                                                                                   "unknown"),
                "judge_score": float(r.get("normalized", 0.0)),
                "judge_decile": decile(r.get("normalized", 0.0)),
                "dialogue_prefix": render_transcript(prefix),
                "target_turn": target_turn.get("text", ""),
                "metric": metric,
            })
    return rows


def stratify(rows: list[dict], n: int) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        buckets.setdefault((r["arm"], r["resistance_level"], r["judge_decile"]), []).append(r)
    rng = rng_for("human_eval", "sample")
    per_arm: dict[str, list[tuple]] = {}
    for k in sorted(buckets):
        per_arm.setdefault(k[0], []).append(k)
    for group in per_arm.values():
        rng.shuffle(group)
    # Round-robin across arms so a small n still covers every arm.
    keys = [k for tier in zip_longest(*per_arm.values()) for k in tier if k is not None]
    picked: list[dict] = []
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < n:
                picked.append(buckets[k].pop(rng.randrange(len(buckets[k]))))
    return picked


def write_sheets(rows: list[dict], out_dir: Path, raters: tuple[str, ...] = ("rater_a", "rater_b")) -> None:
    """One blinded sheet per rater, in a different random order each, with the arm stripped out."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_orders: list[list[str]] = []
    for rater in raters:
        rng = rng_for(rater, "order")
        order = list(rows)
        rng.shuffle(order)
        # Two fixed seeds can coincide; rotate rather than ship two identically ordered sheets.
        while [r["item_id"] for r in order] in seen_orders and len(order) > 1:
            order.append(order.pop(0))
        seen_orders.append([r["item_id"] for r in order])
        sheet = [{"item_id": r["item_id"], "metric": r["metric"],
                  "dialogue_prefix": r["dialogue_prefix"], "target_turn": r["target_turn"],
                  "scores": {str(i): None for i in range(1, 8)}} for r in order]
        write_json(out_dir / f"{rater}.json", sheet)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw the blinded human-rating subsample.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default="ip", choices=["ip", "pri"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run)
    rows = stratify(collect(run_dir, args.metric), args.n)
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "forms" / args.metric
    write_jsonl(out_dir / "sampled.jsonl", rows)
    write_sheets(rows, out_dir)
    print(f"{len(rows)} items sampled for {args.metric}; blinded sheets in {out_dir}")


if __name__ == "__main__":
    main()
