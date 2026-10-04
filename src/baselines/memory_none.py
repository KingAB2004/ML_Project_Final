"""Session-isolated memory: no cross-session state at all. Isolates the value of having any memory."""
from __future__ import annotations

from typing import Sequence


class NoMemory:
    """Same surface as NeedStateMemory, but it forgets everything the moment a session ends."""

    name = "none"

    def __init__(self, profile_id: str = "", **_: object):
        self.profile_id = profile_id
        self.nodes: dict = {}
        self.transitions: list = []

    def brief(self, open_questions: Sequence[str] = ()) -> dict:
        return {"profile_id": self.profile_id, "best_terminal": None, "chain": [], "untested_links": [],
                "open_questions": list(open_questions), "forbidden_inferences": {}, "resolved": [],
                "dormant": []}

    def render_brief(self, open_questions: Sequence[str] = (), token_budget: int | None = None) -> str:
        return "[memory] this arm has no memory: treat this as a first meeting."

    def forbidden_inferences(self) -> list[str]:
        return []

    def stale_references(self) -> list[str]:
        return []

    def find_equivalent(self, text: str, include_closed: bool = True):
        return None

    def propose(self, *_a, **_k):
        return None

    def confirm(self, *_a, **_k):
        return None

    def disconfirm(self, *_a, **_k):
        return None

    def resolve(self, *_a, **_k):
        return None

    def decay(self, *_a, **_k) -> None:
        return None

    def save(self, *_a, **_k) -> None:
        return None

    def status_map(self) -> dict:
        return {}
