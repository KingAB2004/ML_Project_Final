from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import DATA, cfg, read_jsonl, render_transcript, supporter_turns, user_turns, write_jsonl
from judge import Judge
from llm import pmap
from memory import _content_words

SESSIONS_PATH = DATA / "corpus" / "sessions.jsonl"
FILTERED_PATH = DATA / "corpus" / "sessions_filtered.jsonl"
QUARANTINE_PATH = DATA / "quarantine" / "sessions_crisis.jsonl"


def structural_problems(session: dict) -> list[str]:
    turns = session.get("turns", [])
    problems: list[str] = []
    min_turns = int(cfg("filter.min_turns", default=6))
    if len(turns) < min_turns:
        problems.append(f"only {len(turns)} turns, minimum {min_turns}")
    if turns and turns[0].get("role") != "supporter":
        problems.append("the system must open in a proactive setting")
    for a, b in zip(turns, turns[1:]):
        if a.get("role") == b.get("role"):
            problems.append(f"role repeats at turn {b.get('turn_index')}")
            break
    for t in turns:
        if not (t.get("text") or "").strip():
            problems.append(f"empty text at turn {t.get('turn_index')}")
            break
    seen: dict[str, int] = {}
    for t in turns:
        key = re.sub(r"\W+", " ", (t.get("text") or "").lower()).strip()
        if key and key in seen:
            problems.append(f"turn {t.get('turn_index')} repeats turn {seen[key]} verbatim")
            break
        seen[key] = t.get("turn_index", -1)
    problems += leakage_problems(session)
    problems += language_problems(turns)
    return problems


def recites(need: str, text: str, threshold: float = 0.6) -> bool:
    """Share of the need's content words that appear in the turn. Jaccard would dilute a recited need
    inside a longer turn below any threshold, so containment is the right measure here."""
    words = _content_words(need)
    return bool(words) and len(words & _content_words(text)) / len(words) > threshold


def leakage_problems(session: dict) -> list[str]:
    """The simulator must not recite its own hidden need early, or quote its instructions at all."""
    guard = int(cfg("filter.leakage_guard_turns", default=2))
    need = (session.get("profile_snapshot", {}) or {}).get("terminal_need", "")
    out = []
    for i, t in enumerate(user_turns(session.get("turns", []))):
        text = t.get("text", "")
        if i < guard and need and recites(need, text):
            out.append(f"seeker recites the hidden need at user turn {i} (guard is {guard})")
        if "persona_hidden" in text or "disclosure_blockers" in text or "Your Profile" in text:
            out.append(f"prompt leakage in user turn {i}")
    return out


def language_problems(turns: list[dict]) -> list[str]:
    """Per turn, not per transcript: one Chinese turn among ten English ones must still be caught."""
    ratio_min = float(cfg("filter.ascii_ratio_min", default=0.90))
    if not any((t.get("text") or "").strip() for t in turns):
        return ["no text at all"]
    for t in turns:
        text = t.get("text") or ""
        if not text:
            continue
        ratio = sum(1 for ch in text if ord(ch) < 128) / len(text)
        if ratio < ratio_min:
            return [f"non-English content in turn {t.get('turn_index')} (ascii ratio {ratio:.2f})"]
    return []


def crisis_flags(session: dict) -> list[str]:
    terms = cfg("filter.crisis_terms", default=[])
    text = " ".join(t.get("text", "") for t in session.get("turns", [])).lower()
    return [t for t in terms if t in text]


def aels_verdict(result_items: dict, mean: float) -> tuple[bool, str]:
    min_mean = float(cfg("filter.aels_min_mean", default=5.0))
    min_item = float(cfg("filter.aels_min_item", default=3))
    if mean < min_mean:
        return False, f"AELS mean {mean:.2f} < {min_mean}"
    low = [k for k, v in result_items.items() if float(v) < min_item]
    if low:
        return False, f"AELS items below {min_item}: {sorted(low)}"
    return True, "ok"


def run(sessions_path: Path = SESSIONS_PATH, out_path: Path = FILTERED_PATH,
        backend: str | None = None, skip_judge: bool = False) -> dict:
    sessions = read_jsonl(sessions_path)
    quarantined, scored = [], []
    judge = None if skip_judge else Judge(backend=backend)

    def check(session: dict) -> dict:
        qc: dict = {"structural": structural_problems(session)}
        crisis = crisis_flags(session)
        if crisis:
            qc["crisis_terms"] = crisis
        elif qc["structural"]:
            qc["verdict"] = "reject"
            qc["reason"] = qc["structural"][0]
        elif judge is not None:
            res = judge.score_session("aels", session, session.get("profile_snapshot", {}))
            qc["aels"] = res.to_dict()
            ok, why = aels_verdict(res.items, res.mean)
            qc["verdict"] = "keep" if ok and not res.parse_failed else "reject"
            qc["reason"] = why if not ok else ("judge parse failed" if res.parse_failed else "ok")
        else:
            qc["verdict"] = "keep"
            qc["reason"] = "structural only (judge skipped)"
        session["qc"] = qc
        return session

    try:
        for session in pmap(check, sessions, judge.llm if judge else None):
            (quarantined if "crisis_terms" in session["qc"] else scored).append(session)
    finally:
        if judge is not None:
            judge.llm.release()

    kept = prune_broken_tails(scored)
    write_jsonl(out_path, kept)
    write_jsonl(QUARANTINE_PATH, quarantined)
    write_jsonl(out_path.with_name("sessions_rejected.jsonl"),
                [s for s in scored if s["qc"].get("verdict") != "keep"])
    return {
        "input": len(sessions), "kept": len(kept),
        "rejected": len(scored) - len([s for s in scored if s["qc"].get("verdict") == "keep"]),
        "quarantined": len(quarantined),
        "tails_dropped": len([s for s in scored if s["qc"].get("verdict") == "keep"]) - len(kept),
    }


def prune_broken_tails(sessions: list[dict]) -> list[dict]:
    """A profile's sequence is only as long as its first failure. Later sessions are dropped, not kept."""
    by_profile: dict[str, list[dict]] = {}
    for s in sessions:
        by_profile.setdefault(s["profile_id"], []).append(s)
    out = []
    for group in by_profile.values():
        group.sort(key=lambda s: s["session_index"])
        for s in group:
            if s["qc"].get("verdict") != "keep":
                break
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural + AELS filtering.")
    ap.add_argument("--sessions", default=str(SESSIONS_PATH))
    ap.add_argument("--out", default=str(FILTERED_PATH))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--skip-judge", action="store_true", help="structural checks only")
    args = ap.parse_args()
    stats = run(Path(args.sessions), Path(args.out), args.backend, args.skip_judge)
    print(f"filter: kept {stats['kept']}/{stats['input']}, rejected {stats['rejected']}, "
          f"quarantined {stats['quarantined']}, tails dropped {stats['tails_dropped']}")


if __name__ == "__main__":
    main()
