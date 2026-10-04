"""Dense retrieval over past turns: top-k by similarity to the latest turn.

The central comparison for E4: similarity retrieval answers "what looks like now", while the need-state
brief answers "what do we still not know". Embeddings are optional - a bag-of-words cosine stands in when
sentence-transformers is unavailable, and the report must say which was used.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from common import truncate_tokens
from baselines.memory_none import NoMemory


def _bow(text: str) -> Counter:
    return Counter(re.findall(r"\b\w{3,}\b", text.lower()))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class DenseMemory(NoMemory):
    name = "dense"

    def __init__(self, profile_id: str = "", encoder=None, top_k: int = 4, **_: object):
        super().__init__(profile_id)
        self.encoder = encoder           # sentence-transformers model, or None for the bag-of-words fallback
        self.turns: list[dict] = []
        self.top_k = top_k
        self.backend = "sentence-transformers" if encoder is not None else "bag-of-words"

    def ingest_session(self, session: dict) -> None:
        for turn in session.get("turns", []):
            if turn.get("role") == "user" and turn.get("text"):
                self.turns.append({"session_id": session["session_id"], "turn_index": turn["turn_index"],
                                   "text": turn["text"]})

    def retrieve(self, query: str) -> list[dict]:
        if not self.turns:
            return []
        if self.encoder is not None:  # pragma: no cover - optional dependency
            import numpy as np

            vecs = self.encoder.encode([t["text"] for t in self.turns] + [query], normalize_embeddings=True)
            q = vecs[-1]
            sims = (np.array(vecs[:-1]) @ q).tolist()
        else:
            qb = _bow(query)
            sims = [cosine(_bow(t["text"]), qb) for t in self.turns]
        ranked = sorted(zip(self.turns, sims), key=lambda p: -p[1])[: self.top_k]
        return [{**t, "score": round(s, 4)} for t, s in ranked]

    def render_brief(self, open_questions: Sequence[str] = (), token_budget: int | None = None,
                     query: str = "") -> str:
        hits = self.retrieve(query or (self.turns[-1]["text"] if self.turns else ""))
        if not hits:
            return "[memory] first session with this person."
        lines = [f"[memory - {self.backend} retrieval, top {len(hits)} past seeker turns]"]
        lines += [f"- ({h['session_id']} turn {h['turn_index']}, sim {h['score']}) {h['text']}" for h in hits]
        return truncate_tokens("\n".join(lines), int(token_budget or 320))
