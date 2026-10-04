"""Phase 1.4 - two-stage session generation (PLAN Sec. 8.4).

The system opens: this is a proactive setting, the person did not ask for help. Then a listening phase
(reflect, validate, explore, no advice) and a suggestion phase (perspective tied to the inferred need,
still refusable).

Two mirrored histories - what is `assistant` for the simulator is `user` for the supporter - plus one flat
transcript for judges. That two-history shape is the one piece of the upstream chat loop worth keeping.

  python src/dialogue.py --limit 20 --backend echo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from common import (
    DATA,
    cfg,
    fill,
    read_prompt,
    read_jsonl,
    render_transcript,
    rng_for,
    append_jsonl,
    existing_ids,
    fresh_turn,
    word_count,
)
from llm import LLM, pmap

PROFILES_PATH = DATA / "profiles" / "profiles.jsonl"
SESSIONS_PATH = DATA / "corpus" / "sessions.jsonl"


def as_list(value) -> list[str]:
    """A list of phrases out of model JSON: a bare string is one phrase, not one phrase per character."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if str(v).strip()] if isinstance(value, (list, tuple)) else []


def memory_lines(profile: dict) -> str:
    out = []
    events = profile.get("memory")
    for ev in events if isinstance(events, list) else []:
        ev = ev if isinstance(ev, dict) else {"text": str(ev)}
        when = ev.get("when_relative", "recently")
        out.append(f"- {ev.get('text', '')} ({when}, salience {ev.get('salience', 0.5)})")
    return "\n".join(out) or "- nothing specific comes to mind"


def seeker_prompt(turns: Sequence[dict]) -> str:
    """The simulator's per-turn request. It must say WHICH speaker the model is: without that, Qwen-7B
    sometimes continued as the supporter and echoed its last line (v1: about 1 seeker turn in 10)."""
    return (f"{render_transcript(turns)}\n\nYou are the seeker in this conversation; the 'supporter:' lines "
            f"are the other person. Write only your next reply as the seeker, in your own words and in "
            f"English - never repeat or rephrase the supporter.")


def simulator_system(profile: dict, session_context: str = "", reactive: bool = False,
                     corrective: bool = False) -> str:
    """corrective: the evaluation-only seeker that explicitly rejects a wrong reading of its needs
    (prompts/user_corrective.md), so that disconfirmation - and re-proposal - can be measured. The corpus
    seeker resists by deflecting and almost never says "that's not it" (v2: 0 denials of 101 inferences)."""
    if reactive:
        return fill(read_prompt("user_reactive.md"), emotion=profile.get("emotion", ""),
                    feeling=profile.get("feeling", ""), memory=memory_lines(profile))
    hidden = profile.get("persona_hidden") if isinstance(profile.get("persona_hidden"), dict) else {}
    return fill(
        read_prompt("user_proactive.md"),
        emotion=profile.get("emotion", ""),
        feeling=profile.get("feeling", ""),
        terminal_need=profile.get("terminal_need", ""),
        memory=memory_lines(profile),
        persona_surface=json.dumps(profile.get("persona_surface", {}), ensure_ascii=False),
        resistance_level=profile.get("resistance_level", "medium"),
        disclosure_triggers=", ".join(as_list(hidden.get("disclosure_triggers"))),
        disclosure_blockers=", ".join(as_list(hidden.get("disclosure_blockers"))),
        session_context=session_context,
    ) + (read_prompt("user_corrective.md") if corrective else "")


def rung_schedule(turn_number: int, total_turns: int) -> str:
    """Corpus-generation pacing curriculum: shallow early, deeper only late.

    This is a curriculum, not the learned policy - Enhancement 5 replaces it at inference time. It exists so
    the corpus contains rung variety for the annotator and the pacing teacher to learn from.
    """
    frac = turn_number / max(1, total_turns)
    if frac < 0.3:
        return "L0"
    if frac < 0.6:
        return "L1"
    if frac < 0.85:
        return "L2"
    return "L3"


def generate_session(profile: dict, session_id: str, gen: LLM, sim: LLM, n_supporter_turns: int,
                     session_context: str = "", memory_block: str = "", reactive: bool = False,
                     opener: str | None = None) -> dict:
    """One session. `gen` and `sim` may be the same LLM handle (they share a model by design)."""
    rng = rng_for(session_id, "dialogue")
    openers = cfg("corpus.openers", default=["Hey - how are things with you today?"])
    turns: list[dict] = []
    sim_system = simulator_system(profile, session_context, reactive)
    stage_boundary = max(2, int(n_supporter_turns * 0.6))

    first = opener or openers[rng.randrange(len(openers))]
    turns.append({"turn_index": 0, "role": "supporter", "text": first, "phase": "listening",
                  "ladder_rung": "L0", "analysis": None, "strategy": None,
                  "meta": {"source": "opener_pool"}})

    for i in range(n_supporter_turns):
        # seeker replies
        sim_prompt = seeker_prompt(turns)
        reply = fresh_turn(sim.chat, sim_prompt, turns, "seeker", system=sim_system,
                           temperature=float(sim.spec.get("temperature", 0.85)))
        turns.append({"turn_index": len(turns), "role": "user", "text": reply,
                      "phase": "listening" if i < stage_boundary else "suggestion",
                      "disclosure_depth": None,
                      "meta": {"model": sim.spec["model"], "words": word_count(reply)}})

        if i == n_supporter_turns - 1:
            break
        phase = "listening" if i + 1 < stage_boundary else "suggestion"
        template = read_prompt("listen_stage.md" if phase == "listening" else "suggest_stage.md")
        rung = rung_schedule(i + 1, n_supporter_turns)
        sup_prompt = fill(template, permitted_rung=rung, memory_block=memory_block or "[memory] none.",
                          history=render_transcript(turns))
        text = fresh_turn(gen.chat, sup_prompt, turns, "supporter")
        turns.append({"turn_index": len(turns), "role": "supporter", "text": text, "phase": phase,
                      "ladder_rung": rung, "analysis": None, "strategy": None,
                      "meta": {"model": gen.spec["model"], "curriculum_rung": rung}})

    return {
        "session_id": session_id,
        "profile_id": profile["profile_id"],
        "session_index": 1,
        "prev_session_id": None,
        "gap_days_from_prev": None,
        "profile_snapshot": profile,
        "stage_boundary_turn": stage_boundary,
        "turns": turns,
        "qc": {},
        "annotations_complete": False,
    }


def build(limit: int | None = None, backend: str | None = None, reactive: bool = False,
          profiles_path: Path = PROFILES_PATH, out_path: Path = SESSIONS_PATH) -> dict:
    profiles = read_jsonl(profiles_path)
    if limit:
        profiles = profiles[:limit]
    done = existing_ids(out_path, "session_id")
    lo, hi = cfg("corpus.turns_per_session", default=[8, 12])
    llm = LLM("generator", backend=backend)
    written = 0
    try:
        # the simulator is the same resident weights with its own (warmer) sampling
        sim = llm.view("simulator")

        def make(prof: dict) -> dict:
            sid = f"{prof['profile_id']}-s1"
            n_turns = rng_for(sid, "turns").randint(int(lo), int(hi))
            return generate_session(prof, sid, llm, sim, n_turns, reactive=reactive)

        pending = [p for p in profiles if f"{p['profile_id']}-s1" not in done]
        for session in pmap(make, pending, llm):
            append_jsonl(out_path, session)
            written += 1
    finally:
        llm.release()
    return {"written": written, "skipped": len(done), "out": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate first sessions (two-stage, proactive).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--reactive", action="store_true", help="reactive simulator, for the baseline arm")
    ap.add_argument("--profiles", default=str(PROFILES_PATH))
    ap.add_argument("--out", default=str(SESSIONS_PATH))
    args = ap.parse_args()
    stats = build(limit=args.limit, backend=args.backend, reactive=args.reactive,
                  profiles_path=Path(args.profiles), out_path=Path(args.out))
    print(f"sessions written {stats['written']} (skipped {stats['skipped']} already present) "
          f"-> {stats['out']}")


if __name__ == "__main__":
    main()
