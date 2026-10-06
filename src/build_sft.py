"""
 fine-tuning design is that intermediate reasoning is LEARNED, not prompted: Analysis and
Strategy sit in the *target* sequence, ahead of the response.

"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from common import DATA, cfg, read_jsonl, render_transcript, write_jsonl
from profiles import split_profiles

ANNOTATED_PATH = DATA / "corpus" / "sessions_annotated.jsonl"
SFT_DIR = DATA / "sft"
PROFILES_DIR = DATA / "profiles"

SUPPORTER_SYSTEM = (
    "You are a proactive emotional-support listener. You opened this conversation; the person did not ask "
    "for help. Work from what they actually said. Answer in exactly this format: "
    "<analysis>your reading of their state</analysis> <strategy>the move you are choosing</strategy> "
    "<response>your reply</response>. The reply is 1 to 3 sentences of warm, plain spoken English. "
    "Only the text inside <response> is shown to them."
)

A_OPEN, A_CLOSE = "<analysis>", "</analysis>"
S_OPEN, S_CLOSE = "<strategy>", "</strategy>"
R_OPEN, R_CLOSE = "<response>", "</response>"


def render_input(turns: list[dict], upto_index: int, memory_block: str = "",
                 gap_statement: str = "") -> str:
    prefix = [t for t in turns if t["turn_index"] < upto_index]
    parts = []
    if gap_statement:
        parts.append(f"[elapsed] {gap_statement}")
    parts.append(memory_block or "[memory] none.")
    parts.append("Conversation so far:\n" + render_transcript(prefix))
    parts.append("Write the supporter's next turn.")
    return "\n\n".join(parts)


def render_target(analysis: str, strategy: str, response: str, with_thoughts: bool = True) -> str:
    if not with_thoughts:
        return f"{R_OPEN}{response.strip()}{R_CLOSE}"
    return (f"{A_OPEN}{analysis.strip()}{A_CLOSE}\n{S_OPEN}{strategy.strip()}{S_CLOSE}\n"
            f"{R_OPEN}{response.strip()}{R_CLOSE}")


THOUGHT_BLOCK = re.compile(r"<(analysis|strategy)>.*?(?:</\1>|$)", re.S | re.I)
REPLY_LABEL = r"(?:what you say to them|reply|response|supporter)\s*:\s*"


def parse_target(text: str) -> dict:
    def grab(open_tag: str, close_tag: str) -> str:
        m = re.search(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), text, re.S)
        return m.group(1).strip() if m else ""

    response = grab(R_OPEN, R_CLOSE)
    if not response and R_OPEN in text:
        # a truncated generation still has to yield something usable
        response = text.split(R_OPEN)[-1].replace(R_CLOSE, "").strip()
    if not response:
        # Thoughts without a <response> tag. Tagged blocks are dropped (an unclosed one runs to the end);
        # untagged "Analysis: ... Reply: ..." keeps only what follows the reply label, else nothing.
        rest = THOUGHT_BLOCK.sub("", text)
        if re.search(r"(?im)^\s*(analysis|strategy)\s*:", rest):
            labelled = re.split(rf"(?im)^\s*{REPLY_LABEL}", rest)
            rest = labelled[-1] if len(labelled) > 1 else ""
        response = rest.replace(R_CLOSE, "").strip()
    # a label copied in front of the reply ("what you say to them: ...") is not part of it
    response = re.sub(rf"(?i)^\s*{REPLY_LABEL}", "", response).strip()
    return {"analysis": grab(A_OPEN, A_CLOSE), "strategy": grab(S_OPEN, S_CLOSE), "response": response}


def gap_statement(gap_days: float | None, session_index: int | None) -> str:
    if not gap_days:
        return ""
    return (f"It has been about {round(float(gap_days))} days since the previous session with this person "
            f"(session {session_index}).")


def gap_statement_for(session: dict) -> str:
    return gap_statement(session.get("gap_days_from_prev"), session.get("session_index"))


def examples_from_session(session: dict, with_thoughts: bool = True,
                          memory_block: str = "") -> list[dict]:
    rows = []
    turns = session["turns"]
    gap = gap_statement_for(session)
    for turn in turns:
        if turn.get("role") != "supporter":
            continue
        if turn.get("meta", {}).get("source") == "opener_pool":
            continue
        if not (turn.get("analysis") and turn.get("strategy")):
            continue
        rows.append({
            "session_id": session["session_id"],
            "profile_id": session["profile_id"],
            "turn_index": turn["turn_index"],
            "system": SUPPORTER_SYSTEM,
            # memory-aware corpus: the brief this turn was generated under (sessions.py --memory)
            "input": render_input(turns, turn["turn_index"], memory_block=turn.get("memory_block") or memory_block,
                                  gap_statement=gap),
            "target": render_target(turn.get("analysis", ""), turn.get("strategy", ""), turn["text"],
                                    with_thoughts),
            "ladder_rung": turn.get("ladder_rung", "L0"),
        })
    return rows


def split_assignment(profile_ids: list[str], profiles_dir: Path = PROFILES_DIR) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for name in ("train", "val", "test", "calibration"):
        path = profiles_dir / f"profiles_{name}.jsonl"
        if path.exists():
            assignment.update({p["profile_id"]: name for p in read_jsonl(path)})
    if assignment:
        return assignment
    splits = split_profiles([{"profile_id": pid} for pid in sorted(profile_ids)])
    return {p["profile_id"]: name for name, rows in splits.items() for p in rows}


def build(arm: str = "with_thoughts", sessions_path: Path = ANNOTATED_PATH,
          out_dir: Path = SFT_DIR, profiles_dir: Path = PROFILES_DIR) -> dict:
    sessions = read_jsonl(sessions_path)
    with_thoughts = arm == "with_thoughts"
    assignment = split_assignment([s["profile_id"] for s in sessions], profiles_dir)

    buckets: dict[str, list[dict]] = {name: [] for name in ("train", "val", "test", "calibration")}
    unassigned = 0
    for session in sessions:
        name = assignment.get(session["profile_id"])
        if name is None:  # never guess "train" for a profile of unknown split
            unassigned += 1
            continue
        buckets[name].extend(examples_from_session(session, with_thoughts))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, rows in buckets.items():
        path = out_dir / f"{arm}_{name}.jsonl"
        write_jsonl(path, rows)
        written[name] = len(rows)
    return {"arm": arm, "counts": written, "unassigned_sessions": unassigned,
            "profiles": dict(Counter(assignment[pid] for pid in {s["profile_id"] for s in sessions}
                                     if pid in assignment))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT target sequences.")
    ap.add_argument("--arm", choices=["with_thoughts", "wo_thoughts"], default="with_thoughts")
    ap.add_argument("--sessions", default=str(ANNOTATED_PATH))
    ap.add_argument("--out", default=str(SFT_DIR))
    args = ap.parse_args()
    stats = build(args.arm, Path(args.sessions), Path(args.out))
    print(f"{stats['arm']}: " + ", ".join(f"{k}={v}" for k, v in stats["counts"].items()))


if __name__ == "__main__":
    main()
