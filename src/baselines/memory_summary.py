"""Flat running summary: prose regenerated each session. Isolates whether structure beats prose."""
from __future__ import annotations

from typing import Sequence

from common import render_transcript, truncate_tokens
from baselines.memory_none import NoMemory


class SummaryMemory(NoMemory):
    """One paragraph of natural language, rewritten at each session boundary.

    Deliberately has no status, no spans and no notion of a denied inference - which is the thing the
    comparison is meant to expose.
    """

    name = "summary"

    def __init__(self, profile_id: str = "", llm=None, **_: object):
        super().__init__(profile_id)
        self.llm = llm
        self.summary = ""

    def update_from_session(self, session: dict) -> str:
        transcript = render_transcript(session.get("turns", []))
        prompt = ("Summarize what a supporter should remember about this person for next time, in at most "
                  "five sentences of plain prose.\n\n"
                  f"Previous summary: {self.summary or '(none)'}\n\nSession:\n{transcript}")
        self.summary = self.llm.chat(prompt, temperature=0.3).strip() if self.llm else transcript[:600]
        return self.summary

    def render_brief(self, open_questions: Sequence[str] = (), token_budget: int | None = None) -> str:
        if not self.summary:
            return "[memory] first session with this person."
        return truncate_tokens(f"[memory - running summary]\n{self.summary}", int(token_budget or 320))
