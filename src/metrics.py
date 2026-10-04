"""Enhancement 3 - turn-level psychological metrics (PLAN Sec. 11).

IP scores a SUPPORTER turn for overstepping readiness. PRI scores the SEEKER's next turn for resistance
aroused by the preceding supporter move; `counterfactual.py` supplies its attribution.

The three validity controls are enforced here and in judge.py, not left to discipline:
  1. the scripted resistance level is never in a judge prompt (judge.ALLOWED_INFO / NEVER_SHOWN);
  2. the judge model must differ from the supporter and the simulator (judge.assert_judge_separate);
  3. a stratified subsample goes to human raters (human_eval/) and its agreement travels with every number.

  python src/metrics.py --dialogues runs/<id>/dialogues/arm.jsonl --scale ip --backend echo
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Sequence

from common import cfg, read_jsonl, render_transcript, supporter_turns, user_turns, write_jsonl
from judge import Judge
from llm import pmap


def sample_turn_indices(session: dict, positions: Sequence[str] | None = None) -> list[int]:
    """Stratified by position - one early, two middle, one late - because exhaustive scoring is unaffordable."""
    sup = [t["turn_index"] for t in supporter_turns(session.get("turns", []))
           if t.get("meta", {}).get("source") != "opener_pool"]
    if not sup:
        return []
    wanted = list(positions or cfg("eval.pri_turn_positions", default=["early", "middle", "middle", "late"]))
    n = len(sup)
    slots = {"early": 0, "middle": n // 2, "late": n - 1}
    out: list[int] = []
    for w in wanted:
        idx = slots.get(w, n // 2)
        while idx < n and sup[idx] in out:
            idx += 1
        if idx < n:
            out.append(sup[idx])
    return sorted(out)


def score_ip(judge: Judge, session: dict, profile: dict, turn_indices: Sequence[int] | None = None) -> list[dict]:
    """IP per sampled supporter turn."""
    rows = []
    for idx in (turn_indices if turn_indices is not None else sample_turn_indices(session)):
        res = judge.score_turn("ip", session, idx, profile)
        rows.append(res.to_dict())
    return rows


def score_pri_observational(judge: Judge, session: dict, profile: dict) -> list[dict]:
    """Reactance of each seeker reply, as observed. Attribution needs counterfactual.py; this is the raw read."""
    rows = []
    turns = session.get("turns", [])
    for turn in user_turns(turns):
        prev = [t for t in turns if t["turn_index"] < turn["turn_index"]]
        if not prev or prev[-1].get("role") != "supporter":
            continue
        res = judge.score("pri", f"{session['session_id']}#{turn['turn_index']}",
                          render_transcript(prev), profile, target_text=turn.get("text", ""))
        row = res.to_dict()
        row["supporter_turn_index"] = prev[-1]["turn_index"]
        rows.append(row)
    return rows


def score_session_scales(judge: Judge, session: dict, profile: dict,
                         scales: Sequence[str] = ("aels", "crs", "rac", "basic", "success")) -> list[dict]:
    return [judge.score_session(s, session, profile).to_dict() for s in scales]


def ip_components(rows: Sequence[dict]) -> dict:
    """Item 3 - naming a need the seeker never acknowledged - is the sharpest premature-naming signal."""
    item3 = [float(r["items"].get("3", 0)) for r in rows if r.get("items")]
    return {"item3_mean": statistics.fmean(item3) if item3 else None}


def pri_components(rows: Sequence[dict]) -> dict:
    """Reactance is a conjunction: anger alone and counter-arguing alone are different phenomena."""
    anger, cognition = [], []
    for r in rows:
        items = r.get("items") or {}
        if not items:
            continue
        anger.append(statistics.fmean(float(items[k]) for k in ("1", "2") if k in items))
        cognition.append(statistics.fmean(float(items[k]) for k in ("3", "4", "5") if k in items))
    return {"anger_mean": statistics.fmean(anger) if anger else None,
            "negative_cognition_mean": statistics.fmean(cognition) if cognition else None}


# --------------------------------------------------------------------------- published dimensions
#
# The baseline paper reports six basic metrics plus four scale dimensions (their Tables 3 and 4). The item
# groupings below are OURS - the instruments publish items, not groupings - so they are written out here and
# printed in the report, which is the commitment the SOP makes about instrument transparency.

BASIC_NAMES = {1: "fluency", 2: "diversity", 3: "empathy", 4: "information", 5: "humanoid",
               6: "skillfulness"}

# Comforting Responses Scale: items 1-5 are affective improvement; 6, 8 and 9 are the negative
# helper-evaluation items (7 and 10 are positively worded and are NOT counted as negatives).
CRS_AFFECTIVE = (1, 2, 3, 4, 5)
CRS_NEGATIVE = (6, 8, 9)
CRS_NEGATIVE_ALT = (6, 7, 8, 9, 10)          # reported too, with 7 and 10 reverse-scored

# RAC: supportiveness is empathy / compassion / validation / overall support; management is the broader
# conversational-competence half.
RAC_SUPPORTIVENESS = (1, 2, 3, 5, 6, 7, 10, 11)
RAC_MANAGEMENT = (8, 9, 12, 13, 14, 15, 16)
RAC_REVERSE = (4,)                            # "ignored my feelings"


def _mean_items(rows: Sequence[dict], items: Sequence[int], reverse: Sequence[int] = ()) -> float | None:
    vals = []
    for r in rows:
        got = r.get("items") or {}
        picked = [(7.0 + 1.0 - float(got[str(i)])) if i in reverse else float(got[str(i)])
                  for i in items if str(i) in got]
        if picked:
            vals.append(statistics.fmean(picked))
    return statistics.fmean(vals) if vals else None


def basic_breakdown(rows: Sequence[dict]) -> dict:
    """Per-metric means plus `basic_avg`, rescaled to 0-100 the way the baseline table reports them."""
    out: dict = {}
    for num, name in BASIC_NAMES.items():
        raw = _mean_items(rows, (num,))
        out[name] = raw
        out[f"{name}_100"] = None if raw is None else round((raw - 1) / 6 * 100, 1)
    present = [out[f"{n}_100"] for n in BASIC_NAMES.values() if out.get(f"{n}_100") is not None]
    out["basic_avg_100"] = round(statistics.fmean(present), 1) if present else None
    return out


def crs_dimensions(rows: Sequence[dict]) -> dict:
    """Affective Improvement (higher is better) and Negative Helper Evaluations (lower is better)."""
    return {
        "affective_improvement": _mean_items(rows, CRS_AFFECTIVE),
        "negative_helper": _mean_items(rows, CRS_NEGATIVE),
        "negative_helper_all_items_reversed": _mean_items(rows, CRS_NEGATIVE_ALT, reverse=(7, 10)),
        "items_used": {"affective": list(CRS_AFFECTIVE), "negative": list(CRS_NEGATIVE)},
    }


def rac_dimensions(rows: Sequence[dict]) -> dict:
    """Supportiveness and Management, with item 4 reverse-scored."""
    return {
        "supportiveness": _mean_items(rows, RAC_SUPPORTIVENESS),
        "management": _mean_items(rows, RAC_MANAGEMENT),
        "items_used": {"supportiveness": list(RAC_SUPPORTIVENESS), "management": list(RAC_MANAGEMENT),
                       "reverse_scored": list(RAC_REVERSE)},
    }


def aggregate(rows: Sequence[dict], scale: str) -> dict:
    vals = [float(r["normalized"]) for r in rows if not r.get("parse_failed")]
    out = {
        "scale": scale,
        "n": len(vals),
        "mean_normalized": statistics.fmean(vals) if vals else None,
        "mean_raw": statistics.fmean([float(r["mean"]) for r in rows if not r.get("parse_failed")])
        if vals else None,
        "parse_failures": sum(1 for r in rows if r.get("parse_failed")),
        "judge_model": rows[0]["judge_model"] if rows else "",
        "redactions": rows[0]["context_redactions"] if rows else [],
    }
    if scale == "ip":
        out.update(ip_components(rows))
    if scale == "pri":
        out.update(pri_components(rows))
    if scale == "basic":
        out.update(basic_breakdown(rows))
    if scale == "crs":
        out.update(crs_dimensions(rows))
    if scale == "rac":
        out.update(rac_dimensions(rows))
    return out


def run(dialogues_path: Path, out_dir: Path, scales: Sequence[str], backend: str | None = None) -> dict:
    sessions = read_jsonl(dialogues_path)
    judge = Judge(backend=backend)
    summary = {}
    try:
        for scale in scales:

            def score(session: dict, scale: str = scale) -> list[dict]:
                profile = session.get("profile_snapshot", {})
                if scale == "ip":
                    return score_ip(judge, session, profile)
                if scale == "pri":
                    return score_pri_observational(judge, session, profile)
                return [judge.score_session(scale, session, profile).to_dict()]

            rows = [row for part in pmap(score, sessions, judge.llm) for row in part]
            write_jsonl(out_dir / f"{scale}.jsonl", rows)
            summary[scale] = aggregate(rows, scale)
    finally:
        judge.llm.release()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Score dialogues on the instruments and the turn-level metrics.")
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--out", default=None, help="defaults to <dialogues dir>/../scores")
    ap.add_argument("--scale", action="append", default=None,
                    choices=["aels", "crs", "rac", "basic", "success", "ip", "pri"])
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    dialogues = Path(args.dialogues)
    out_dir = Path(args.out) if args.out else dialogues.parent.parent / "scores" / dialogues.stem
    scales = args.scale or ["success", "aels", "crs", "rac", "basic", "ip", "pri"]
    summary = run(dialogues, out_dir, scales, args.backend)
    for scale, agg in summary.items():
        print(f"{scale}: n={agg['n']} mean={agg['mean_normalized']} "
              f"failures={agg['parse_failures']} judge={agg['judge_model']}")


if __name__ == "__main__":
    main()
