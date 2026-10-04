"""Label every seeker reply with four judge ratings (disclosed, opening, guarded, pulling back) and a state.

One judge pass over the evaluation dialogues, shared by markov.py and survival.py. The judge sees the
seeker's terminal need (it must, to tell whether the need was disclosed - the same exposure scale_success.md
has) but never the scripted resistance level or the hidden persona. Raw ratings are stored, so the state
rule below can change without a single new model call.

State rule (1-7 ratings; 5 = "Considerably"):
  disclosed  item 1 >= 5                                   the need is in the seeker's words, or confirmed
  withdrawn  item 4 >= 5 and item 4 >= item 2              pulls back more than it opens
  opening    item 2 >= 5                                   new feeling, detail or concern
  guarded    otherwise

  python extra/label_seeker.py --run runs/v3_50                        # every test arm
  python extra/label_seeker.py --run runs/v3_50 --arms cellA_mono_ungated --limit 10
"""
from __future__ import annotations

import argparse

from _shared import fill, load_arm_dialogues, out_dir, rate_items, read_extra_prompt, read_jsonl, run_path, \
    write_jsonl
from common import render_transcript, user_turns
from judge import assert_judge_separate
from llm import LLM, pmap

ITEMS = ("disclosed", "opening", "guarded", "pulling_back")
STATES = ("guarded", "opening", "withdrawn", "disclosed")
THRESHOLD = 5


def state_of(row: dict) -> str:
    if row["disclosed"] >= THRESHOLD:
        return "disclosed"
    if row["pulling_back"] >= THRESHOLD and row["pulling_back"] >= row["opening"]:
        return "withdrawn"
    if row["opening"] >= THRESHOLD:
        return "opening"
    return "guarded"


def label_session(llm, template: str, session: dict) -> list[dict]:
    profile = session.get("profile_snapshot", {})
    need = profile.get("terminal_need") or "(not recorded)"
    turns = session.get("turns", [])
    rows = []
    for k, turn in enumerate(user_turns(turns), start=1):
        prev = [t for t in turns if t["turn_index"] < turn["turn_index"]]
        prompt = fill(template, need=need, diag=render_transcript(prev), target=turn.get("text", ""))
        items = rate_items(llm, prompt, len(ITEMS))
        row = {"session_id": session["session_id"], "profile_id": session.get("profile_id"),
               "turn_index": turn["turn_index"], "seeker_turn": k, "parse_failed": items is None}
        if items:
            row.update({name: items[i + 1] for i, name in enumerate(ITEMS)})
            row["state"] = state_of(row)
        rows.append(row)
    return rows


def load_labels(run_dir, arms=None) -> dict[str, list[dict]]:
    """{arm: label rows}, as written by this script."""
    d = run_dir / "extra" / "seeker_states"
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    return {f.stem: read_jsonl(f) for f in files if not arms or f.stem in arms}


def sequences(rows: list[dict]) -> dict[str, list[str]]:
    """{session_id: [state per seeker turn, in order]}. A turn whose rating failed to parse ends the
    sequence there: a gap would otherwise join two non-adjacent turns into one transition."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["session_id"], []).append(r)
    out = {}
    for sid, rs in by.items():
        seq = []
        for r in sorted(rs, key=lambda x: x["seeker_turn"]):
            if r.get("parse_failed"):
                break
            seq.append(r["state"])
        out[sid] = seq
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge pass: a state for every seeker reply.")
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/v3_50")
    ap.add_argument("--arms", nargs="*", default=None, help="default: every arm except calibration")
    ap.add_argument("--limit", type=int, default=None, help="sessions per arm")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    run_dir = run_path(args.run)
    dialogues = load_arm_dialogues(run_dir, args.arms)
    template = read_extra_prompt("seeker_state.md")
    dest = out_dir(run_dir, "seeker_states")
    assert_judge_separate("judge")
    llm = LLM("judge", backend=args.backend, call_log=run_dir / "calls.jsonl")
    try:
        for arm, sessions in dialogues.items():
            sessions = sessions[: args.limit] if args.limit else sessions
            rows = [r for part in pmap(lambda s: label_session(llm, template, s), sessions, llm) for r in part]
            write_jsonl(dest / f"{arm}.jsonl", rows)
            ok = [r for r in rows if not r["parse_failed"]]
            counts = {s: sum(1 for r in ok if r["state"] == s) for s in STATES}
            print(f"{arm}: {len(rows)} seeker turns, {len(rows) - len(ok)} unparsed, states {counts}")
    finally:
        llm.release()


if __name__ == "__main__":
    main()
