"""Phase 1.5 - multi-session linking with sampled time gaps (PLAN Sec. 8.5).

This is the part that makes Limitation 3 (context collapse across sessions) addressable, so the rules are
explicit rather than free drift:

  - gaps are log-uniform over 1 day to 8 weeks, so "next day" and "six weeks later" both occur and
    recency-versus-relevance becomes a real tension;
  - the profile is advanced under a named transition (resolved / intensified / displaced), and the applied
    transition is recorded as the ground truth E4's memory recall is scored against;
  - the elapsed interval is stated in BOTH contexts - a memory system cannot be evaluated on time it was
    never told about;
  - chain node ids stay stable across sessions; only a displaced concern introduces new ids.

  python src/sessions.py --limit 20 --backend echo
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import (
    DATA,
    append_jsonl,
    cfg,
    fill,
    read_prompt,
    read_jsonl,
    render_transcript,
    rng_for,
    to_int,
)
from dialogue import generate_session
from llm import LLM, pmap

PROFILES_PATH = DATA / "profiles" / "profiles.jsonl"
SESSIONS_PATH = DATA / "corpus" / "sessions.jsonl"
TRANSITIONS = ("resolved", "intensified", "displaced")


def sample_gap_days(key: str) -> float:
    lo, hi = cfg("corpus.gap_days_range", default=[1.0, 56.0])
    r = rng_for(key, "gap").random()
    return float(math.exp(math.log(lo) + r * (math.log(hi) - math.log(lo))))


def sample_session_count(profile_id: str) -> int:
    lo, hi = cfg("corpus.sessions_per_profile", default=[2, 4])
    return rng_for(profile_id, "session_count").randint(int(lo), int(hi))


def pick_transition(key: str, index: int) -> str:
    rng = rng_for(f"{key}-{index}", "transition")
    weights = {"resolved": 0.35, "intensified": 0.45, "displaced": 0.20}
    roll, acc = rng.random(), 0.0
    for name, w in weights.items():
        acc += w
        if roll <= acc:
            return name
    return "intensified"


def gap_phrase(days: float) -> str:
    if days < 2:
        return "since yesterday"
    if days < 10:
        return f"in the {round(days)} days since we last talked"
    if days < 45:
        return f"in the {max(1, round(days / 7))} weeks since we last talked"
    return "in the couple of months since we last talked"


def valid_chain(chain) -> bool:
    depth = int(cfg("corpus.need_chain_depth", default=3))
    return (isinstance(chain, list) and len(chain) == depth
            and all(isinstance(n, dict) and str(n.get("text") or "").strip() for n in chain))


def advance_profile(llm: LLM, profile: dict, prev_session: dict, gap_days: float,
                    transition: str) -> dict:
    prompt = fill(read_prompt("profile_advance.md"),
                  gap_days=f"{gap_days:.1f}",
                  profile_json=json.dumps({k: profile.get(k) for k in
                                           ("emotion", "feeling", "need_chain", "memory", "terminal_need",
                                            "resistance_level")}, ensure_ascii=False),
                  transcript=render_transcript(prev_session["turns"]),
                  transition=transition)
    # max_tokens: the advanced profile is as long as the original one
    out = llm.structured(prompt, required=("need_chain",), max_tokens=1024)
    if not valid_chain(out.get("need_chain")):
        # v3: 5 of 103 follow-ups came back with a 1- or 2-node chain, so the "terminal need" (Success's
        # ground truth) silently became a shallower node, and later sessions inherited it. Ask once more.
        out = llm.structured(f"{prompt}\n\nYour need_chain must have exactly three nodes (depth 0, 1, 2), "
                             f"each with non-empty text.", required=("need_chain",), max_tokens=1024)
    chain_ok = valid_chain(out.get("need_chain"))
    advanced = dict(profile)
    # Still malformed: keep the previous chain (recorded below) rather than a truncated one.
    chain_in = out["need_chain"] if chain_ok else profile["need_chain"]
    old = {n["depth"]: n for n in profile.get("need_chain", [])}
    chain = []
    for i, node in enumerate(chain_in[:3]):
        node = node if isinstance(node, dict) else {"text": str(node)}
        depth = to_int(node.get("depth"), i, 0, 9)
        text = (node.get("text") or "").strip()
        keep_id = (transition != "displaced" and depth in old)
        node_id = old[depth]["node_id"] if keep_id else f"nd{depth:03d}x{prev_session['session_index']}"
        chain.append({"node_id": node_id, "depth": depth, "text": text,
                      "parent_id": None if i == 0 else chain[i - 1]["node_id"]})
    advanced.update({
        "emotion": out.get("emotion", profile["emotion"]),
        "feeling": out.get("feeling", profile["feeling"]),
        "need_chain": chain,
        "terminal_need": chain[-1]["text"] if chain else profile["terminal_need"],
        "memory": out["memory"] if isinstance(out.get("memory"), list) else profile.get("memory", []),
        "resistance_level": (out.get("resistance_level") if out.get("resistance_level") in ("low", "medium", "high")
                             else profile.get("resistance_level", "medium")),
    })
    hidden = dict(advanced.get("persona_hidden", {}))
    hidden["terminal_need"] = advanced["terminal_need"]
    hidden["resistance_level"] = advanced["resistance_level"]
    advanced["persona_hidden"] = hidden
    advanced["advancement"] = {"applied": transition, "gap_days": gap_days,
                               "notes": out.get("notes", ""), "parse_failed": out.get("parse_failed", False),
                               "chain_kept_from_previous": not chain_ok}
    return advanced


def build(limit: int | None = None, backend: str | None = None, profiles_path: Path = PROFILES_PATH,
          sessions_path: Path = SESSIONS_PATH) -> dict:
    profiles = {p["profile_id"]: p for p in read_jsonl(profiles_path)}
    first_sessions = {s["profile_id"]: s for s in read_jsonl(sessions_path) if s["session_index"] == 1}
    existing = {s["session_id"]: s for s in read_jsonl(sessions_path)}
    lo, hi = cfg("corpus.turns_per_session", default=[8, 12])
    llm = LLM("generator", backend=backend)
    written = 0
    sim = llm.view("simulator")

    def chain(first: dict) -> list[dict]:
        """Follow-up sessions for one profile. Returns only the newly generated ones."""
        pid = first["profile_id"]
        new: list[dict] = []
        total = sample_session_count(pid)
        prev, current = first, dict(first["profile_snapshot"])
        for index in range(2, total + 1):
            sid = f"{pid}-s{index}"
            gap = sample_gap_days(sid)
            transition = pick_transition(pid, index)
            current = advance_profile(llm, current, prev, gap, transition)
            if sid in existing:
                prev = existing[sid]
                continue
            context = (f"This conversation is a follow-up: it has been {gap:.0f} days "
                       f"({gap_phrase(gap)}) since the previous one. What has changed: "
                       f"{current['advancement'].get('notes', 'some things shifted')}")
            n_turns = rng_for(sid, "turns").randint(int(lo), int(hi))
            session = generate_session(
                current, sid, llm, sim, n_turns, session_context=context,
                opener=f"Hey - it's been a little while. How have things been {gap_phrase(gap)}?")
            session.update({"session_index": index, "prev_session_id": prev["session_id"],
                            "gap_days_from_prev": gap, "profile_snapshot": current})
            new.append(session)
            prev = session
        return new

    firsts = [f for pid, f in list(first_sessions.items())[: limit or len(first_sessions)] if pid in profiles]
    linked_profiles = len(firsts)
    try:
        for new in pmap(chain, firsts, llm):
            for session in new:
                append_jsonl(sessions_path, session)
                written += 1
    finally:
        llm.release()
    return {"written": written, "profiles_linked": linked_profiles}


def integrity_check(sessions: list[dict]) -> list[str]:
    """Temporal integrity: gaps positive, order monotone, links resolvable, transitions recorded."""
    problems: list[str] = []
    by_profile: dict[str, list[dict]] = {}
    for s in sessions:
        by_profile.setdefault(s["profile_id"], []).append(s)
    for pid, group in by_profile.items():
        group.sort(key=lambda s: s["session_index"])
        indices = [s["session_index"] for s in group]
        if indices != list(range(1, len(group) + 1)):
            problems.append(f"{pid}: session indices {indices} are not contiguous from 1")
        for s in group[1:]:
            if not s.get("prev_session_id"):
                problems.append(f"{s['session_id']}: missing prev_session_id")
            gap = s.get("gap_days_from_prev")
            if gap is None or float(gap) <= 0:
                problems.append(f"{s['session_id']}: non-positive gap {gap}")
            if not (s.get("profile_snapshot", {}).get("advancement", {}) or {}).get("applied"):
                problems.append(f"{s['session_id']}: no advancement transition recorded")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Link follow-up sessions with sampled time gaps.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--profiles", default=str(PROFILES_PATH))
    ap.add_argument("--sessions", default=str(SESSIONS_PATH))
    ap.add_argument("--check", action="store_true", help="only run the temporal integrity check")
    args = ap.parse_args()
    if args.check:
        problems = integrity_check(read_jsonl(args.sessions))
        print("\n".join(problems) if problems else "temporal integrity: ok")
        return
    stats = build(limit=args.limit, backend=args.backend, profiles_path=Path(args.profiles),
                  sessions_path=Path(args.sessions))
    print(f"follow-up sessions written {stats['written']} across {stats['profiles_linked']} profiles")


if __name__ == "__main__":
    main()
