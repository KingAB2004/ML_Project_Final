"""Cross-session memory metrics that need no model (CPU): SOP Expected Outcome 4 / PLAN Sec. 12.4.

1. Success Rate by time gap. Every session's judge Success score (scores/<arm>/success.jsonl) grouped by the
   gap since the previous session: session 1 (no gap), <= 7 days, 7-21 days, > 21 days. The corpus draws gaps
   log-uniform on 1-56 days, so the three gap buckets hold about 48 / 27 / 24 % of follow-up sessions. Does
   success hold up after a long gap? 95 % CIs: bootstrap over profiles.

2. Re-proposal of denied inferences, measured two ways:
   a. recorded: memory nodes the system marked `disconfirmed`, and later supporter turns that still overlap
      them (grounding._overlaps, 60 % of the node's content words). This is what the memory blocks.
   b. dialogue-level, for every arm with or without memory: a supporter turn that makes an inference
      (a hedge or state attribution introducing a want, need or cause, grounding's cue lists), answered by a
      seeker turn that OPENS with an explicit denial cue, is a denial; the inference sentence is the denied claim. A later
      supporter turn of the same profile (rest of the session, or later sessions) that contains 60 % of the
      claim's content words re-proposes it. A lexical heuristic: it misses paraphrased re-proposals and
      denials worded without a cue, so read it as a lower bound and compare arms with it, not as an exact
      rate. A judge-based version is the planned GPU step.

  python extra/memory_eval.py --run results/v2/runs/v3_50
Writes runs/<id>/extra/memory_eval/: summary.json, report.md, denials.jsonl, success_by_gap.png.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics

from _shared import cluster_bootstrap, fmt, load_arm_dialogues, markdown_table, out_dir, pyplot, read_jsonl, \
    run_path, write_json, write_jsonl
from grounding import ASSERTION_CUES, HEDGES, INFERENCE_MARKERS, _overlaps

BUCKETS = ("session 1", "<= 7 d", "7-21 d", "> 21 d")
DENIAL_CUES = ("that's not it", "that's not what", "that's not really", "that's not the", "it's not that",
               "it isn't that", "not really", "not exactly", "i don't think that's", "i don't think so",
               "i wouldn't say", "you're wrong", "that isn't", "no, it's", "no, i", "no it's", "nah")
MIN_CLAIM_WORDS = 3


def bucket(session: dict) -> str:
    gap = session.get("gap_days_from_prev")
    if session.get("session_index", 1) == 1 or not gap:
        return BUCKETS[0]
    return BUCKETS[1] if gap <= 7 else BUCKETS[2] if gap <= 21 else BUCKETS[3]


def success_by_gap(run_dir, arm: str, sessions: list[dict], reps: int) -> dict | None:
    scores = {r["target"]: r["normalized"] for r in read_jsonl(run_dir / "scores" / arm / "success.jsonl")
              if not r.get("parse_failed")}
    if not scores:
        return None
    out = {}
    for b in BUCKETS:
        by_profile: dict[str, list[float]] = {}
        for s in sessions:
            if bucket(s) == b and s["session_id"] in scores:
                by_profile.setdefault(s["profile_id"], []).append(scores[s["session_id"]])
        vals = [v for vs in by_profile.values() for v in vs]
        if vals:
            out[b] = {"n_sessions": len(vals), "n_profiles": len(by_profile), "success": statistics.fmean(vals),
                      "ci": cluster_bootstrap(list(by_profile.values()),
                                              lambda g: statistics.fmean(v for vs in g for v in vs), reps,
                                              f"memeval:{arm}:{b}")}
    return out


def is_denial(text: str) -> bool:
    """A reply that OPENS by rejecting what was said. A cue later in the reply ("it's not really my fault")
    is about something else, so only the start counts (after a filler like "oh," or "well,")."""
    low = re.sub(r"^\W*(oh|hmm+|um+|well|honestly|i mean)\W+", "", text.lower().replace("\u2019", "'").strip())
    return any(low.startswith(cue) for cue in DENIAL_CUES)


def inference_sentence(text: str) -> str | None:
    """The sentence of a supporter turn that reads into the seeker: a hedge or state attribution introducing a
    want, need or cause (the same cue lists grounding.classify_rung uses)."""
    for sent in re.split(r"(?<=[.?!])\s+", text):
        low = sent.lower()
        if (any(c in low for c in HEDGES + ASSERTION_CUES) and any(m in low for m in INFERENCE_MARKERS)
                and len(re.findall(r"\b\w{4,}\b", sent)) >= MIN_CLAIM_WORDS):
            return sent
    return None


def denials_and_reproposals(sessions: list[dict]) -> list[dict]:
    """Every detected denial of one profile's supporter inference, with the later turns that re-propose it."""
    by_profile: dict[str, list[dict]] = {}
    for s in sessions:
        by_profile.setdefault(s["profile_id"], []).append(s)
    events = []
    for pid, ss in by_profile.items():
        stream = [(s["session_index"], t) for s in sorted(ss, key=lambda x: x.get("session_index", 1))
                  for t in s["turns"]]
        for i, (sidx, turn) in enumerate(stream[:-1]):
            nxt_sidx, nxt = stream[i + 1]
            if (turn["role"] != "supporter" or turn.get("meta", {}).get("source") == "opener_pool"
                    or nxt["role"] != "user" or nxt_sidx != sidx or not is_denial(nxt["text"])):
                continue
            claim = inference_sentence(turn["text"])
            if not claim:
                continue
            later = [(j_sidx, t) for j_sidx, t in stream[i + 2:]
                     if t["role"] == "supporter" and _overlaps(t["text"].lower(), claim)]
            events.append({"profile_id": pid, "session_index": sidx, "turn_index": turn["turn_index"],
                           "claim": claim, "denial": nxt["text"][:300],
                           "reproposals": [{"session_index": j, "turn_index": t["turn_index"], "text": t["text"][:300]}
                                           for j, t in later]})
    return events


def recorded_reproposals(run_dir, arm: str, sessions: list[dict]) -> dict:
    """Disconfirmed memory nodes, and later supporter turns that overlap them."""
    nodes = []
    for f in sorted((run_dir / "memory").glob(f"*_{arm}.jsonl")):
        nodes += [(f.stem.split("_", 1)[0], n) for n in read_jsonl(f)]
    disconfirmed = [(pid, n) for pid, n in nodes if n.get("status") == "disconfirmed"]
    hits = 0
    for pid, n in disconfirmed:
        span = (n.get("contradicting_spans") or [{}])[-1]
        after = False
        for s in sorted((s for s in sessions if s["profile_id"] == pid), key=lambda x: x["session_index"]):
            for t in s["turns"]:
                if s["session_id"] == span.get("session_id") and t["turn_index"] == span.get("turn_index"):
                    after = True
                elif after and t["role"] == "supporter" and _overlaps(t["text"].lower(), n.get("text", "")):
                    hits += 1
    return {"memory_nodes": len(nodes), "by_status": dict(sorted(
        {st: sum(1 for _, n in nodes if n.get("status") == st) for st in {n.get("status") for _, n in nodes}}.items())),
        "disconfirmed": len(disconfirmed), "reproposals_after_disconfirmation": hits}


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU memory metrics: success by gap, re-proposal of denials.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    run_dir = run_path(args.run)
    dialogues = load_arm_dialogues(run_dir, args.arms)
    results, all_events = {}, []
    for arm, sessions in dialogues.items():
        events = denials_and_reproposals(sessions)
        all_events += [{"arm": arm, **e} for e in events]
        n_sup = sum(1 for s in sessions for t in s["turns"] if t["role"] == "supporter")
        results[arm] = {
            "sessions": len(sessions), "profiles": len({s["profile_id"] for s in sessions}),
            "memory": (sessions[0].get("arm_spec") or {}).get("memory"),
            "multi_session": any(s.get("session_index", 1) > 1 for s in sessions),
            "success_by_gap": success_by_gap(run_dir, arm, sessions, args.reps),
            "denials_detected": len(events), "supporter_turns": n_sup,
            "denials_reproposed": sum(1 for e in events if e["reproposals"]),
            "reproposal_turns": sum(len(e["reproposals"]) for e in events),
            "reproposal_turns_cross_session": sum(1 for e in events for r in e["reproposals"]
                                                  if r["session_index"] > e["session_index"]),
            "recorded": recorded_reproposals(run_dir, arm, sessions),
        }

    dest = out_dir(run_dir, "memory_eval")
    write_json(dest / "summary.json", {"buckets": BUCKETS, "denial_cues": DENIAL_CUES, "arms": results})
    write_jsonl(dest / "denials.jsonl", all_events)

    gap_rows = []
    for arm, r in results.items():
        row = {"arm": arm, "memory": r["memory"] or "-"}
        for b in BUCKETS:
            g = (r["success_by_gap"] or {}).get(b)
            row[b] = (f"{fmt(g['success'])} [{fmt(g['ci'][0])}, {fmt(g['ci'][1])}] (n={g['n_sessions']})"
                      if g else "-")
        gap_rows.append(row)
    rep_rows = [{"arm": arm, "memory": r["memory"] or "-", "sessions": r["sessions"],
                 "denials detected": r["denials_detected"],
                 "denials re-proposed": r["denials_reproposed"],
                 "re-proposing turns (cross-session)": f"{r['reproposal_turns']} ({r['reproposal_turns_cross_session']})",
                 "memory nodes": r["recorded"]["memory_nodes"],
                 "disconfirmed nodes": r["recorded"]["disconfirmed"],
                 "re-proposed after disconfirmation": r["recorded"]["reproposals_after_disconfirmation"]}
                for arm, r in results.items()]
    lines = [f"# Memory metrics without a model - {run_dir.name}", "",
             "## 1. Success Rate by time since the previous session", "",
             "Judge Success per session (0-1), grouped by the gap before it. 95 % CI: bootstrap over profiles.", "",
             markdown_table(gap_rows, list(gap_rows[0])), "",
             "## 2. Re-proposal of denied inferences", "",
             "Dialogue level: a supporter inference (>= L2) answered by an explicit seeker denial, and later supporter "
             "turns of the same profile containing 60 % of the denied sentence's content words. Lexical heuristic: a "
             "lower bound, for comparing arms. Recorded: nodes the need-state memory marked disconfirmed.", "",
             markdown_table(rep_rows, list(rep_rows[0])), "",
             "Every detected denial, with its claim and any re-proposals, is in `denials.jsonl`."]
    plt = pyplot()
    multi = {a: r for a, r in results.items() if r["multi_session"] and r["success_by_gap"]}
    if plt is not None and multi:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for k, (arm, r) in enumerate(multi.items()):
            xs = [i + (k - len(multi) / 2) * 0.08 for i, b in enumerate(BUCKETS) if b in r["success_by_gap"]]
            gs = [r["success_by_gap"][b] for b in BUCKETS if b in r["success_by_gap"]]
            ax.errorbar(xs, [g["success"] for g in gs],
                        yerr=[[g["success"] - g["ci"][0] for g in gs], [g["ci"][1] - g["success"] for g in gs]],
                        marker="o", capsize=3, label=arm)
        ax.set_xticks(range(len(BUCKETS)), BUCKETS)
        ax.set_ylabel("Success (0-1)")
        ax.set_title("Success by time since the previous session", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "success_by_gap.png", dpi=150)
        plt.close(fig)
        lines += ["", "![success_by_gap.png](success_by_gap.png)"]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({a: {k: r[k] for k in ("denials_detected", "denials_reproposed")} for a, r in results.items()}))
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
