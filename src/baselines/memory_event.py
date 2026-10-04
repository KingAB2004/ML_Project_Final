"""Event-level memory in the style of the strong simple baseline the SOP names (reference [9]).

Atomic events with timestamps, retrieved by recency and relevance. Structurally simple on purpose: if this
matches the need-state memory, the honest conclusion is that the inference-chain structure did not pay, and
the report says so.
"""
from __future__ import annotations

import math
from typing import Sequence

from common import truncate_tokens
from baselines.memory_dense import _bow, cosine
from baselines.memory_none import NoMemory


class EventMemory(NoMemory):
    name = "event"

    def __init__(self, profile_id: str = "", llm=None, half_life_days: float = 21.0, top_k: int = 5,
                 **_: object):
        super().__init__(profile_id)
        self.llm = llm
        self.events: list[dict] = []
        self.half_life = half_life_days
        self.top_k = top_k
        self.now_days = 0.0

    def advance_clock(self, gap_days: float) -> None:
        self.now_days += float(gap_days or 0.0)

    def ingest_session(self, session: dict) -> None:
        """Extract atomic events from the seeker's turns. One model call per session, not per turn."""
        text = "\n".join(t["text"] for t in session.get("turns", []) if t.get("role") == "user")
        if self.llm is not None:
            prompt = ("List the concrete events and facts this person reported, one per line, no commentary, "
                      f"at most eight lines.\n\n{text}")
            lines = [l.strip("-* \t") for l in self.llm.chat(prompt, temperature=0.2).splitlines()]
        else:
            lines = [t["text"] for t in session.get("turns", []) if t.get("role") == "user"]
        for line in [l for l in lines if len(l.split()) >= 3]:
            self.events.append({"text": line, "at_days": self.now_days,
                                "session_id": session.get("session_id", "")})

    def retrieve(self, query: str) -> list[dict]:
        qb = _bow(query)
        scored = []
        for ev in self.events:
            recency = 0.5 ** ((self.now_days - ev["at_days"]) / self.half_life)
            scored.append((ev, 0.5 * cosine(_bow(ev["text"]), qb) + 0.5 * recency))
        return [{**ev, "score": round(s, 4)} for ev, s in sorted(scored, key=lambda p: -p[1])[: self.top_k]]

    def render_brief(self, open_questions: Sequence[str] = (), token_budget: int | None = None,
                     query: str = "") -> str:
        hits = self.retrieve(query or (self.events[-1]["text"] if self.events else ""))
        if not hits:
            return "[memory] first session with this person."
        lines = ["[memory - event level, recency plus relevance]"]
        lines += [f"- (day {h['at_days']:.0f}, score {h['score']}) {h['text']}" for h in hits]
        return truncate_tokens("\n".join(lines), int(token_budget or 320))
