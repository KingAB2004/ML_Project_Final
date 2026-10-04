"""Enhancement 4: need-state trajectory memory (PLAN Sec. 12).

Not a store of facts and not a graph: the unit of memory is the need chain, kept as a revisable belief.
What the module must carry across sessions is *which links have been tested* - which is why
`disconfirmed` is a first-class status with the user's own denial quoted, and why re-proposal is blocked.

State is a fold over an append-only transition log, so every belief change is explainable afterwards.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

from common import Span, append_jsonl, cfg, read_jsonl, truncate_tokens, utc_stamp, write_jsonl

STATUS_HYPOTHESIS = "hypothesis"
STATUS_CONFIRMED = "confirmed"
STATUS_DISCONFIRMED = "disconfirmed"
STATUS_RESOLVED = "resolved"


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\b\w{4,}\b", (text or "").lower())}


def similarity(a: str, b: str) -> float:
    """Jaccard over content words. Cheap on purpose: no embedding model is loaded for equivalence."""
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 1.0 if (a or "").strip().lower() == (b or "").strip().lower() else 0.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class MemoryNode:
    node_id: str
    text: str
    depth: int = 1
    parent_id: str | None = None
    status: str = STATUS_HYPOTHESIS
    confidence: float = 0.3
    confidence_raw: float = 0.3
    supporting_spans: list[dict] = field(default_factory=list)
    contradicting_spans: list[dict] = field(default_factory=list)
    first_seen_at: str = ""
    last_updated_at: str = ""
    first_seen_session: str = ""
    last_session: str = ""
    blocked_from_reproposal: bool = False
    reinstated_count: int = 0
    dormant: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class NeedStateMemory:
    """One instance per profile. Construct empty, or `load()` a previous run's log."""

    def __init__(self, profile_id: str, log_path: str | Path | None = None):
        self.profile_id = profile_id
        self.nodes: dict[str, MemoryNode] = {}
        self.transitions: list[dict] = []
        self.log_path = Path(log_path) if log_path else None
        self._counter = 0
        self.half_life = float(cfg("memory.decay_half_life_days", default=14.0))
        self.reinstate_factor = float(cfg("memory.reinstate_confidence_factor", default=0.5))
        self.min_confirm = int(cfg("memory.min_spans_to_confirm", default=2))
        self.min_reopen = int(cfg("memory.min_spans_to_reopen_disconfirmed", default=2))
        self.dormant_floor = float(cfg("memory.dormant_confidence_floor", default=0.15))
        self.equiv_min = float(cfg("memory.equivalence_overlap_min", default=0.6))

    # -- transition log ----------------------------------------------------
    def _record(self, node_id: str, frm: str | None, to: str, reason: str, span: dict | None = None,
                session_id: str = "", extra: dict | None = None) -> None:
        entry = {
            "profile_id": self.profile_id, "node_id": node_id, "from": frm, "to": to,
            "reason": reason, "span": span, "session_id": session_id, "at": utc_stamp(),
        }
        if extra:
            entry.update(extra)
        self.transitions.append(entry)
        if self.log_path:
            append_jsonl(self.log_path, entry)

    def new_node_id(self) -> str:
        self._counter += 1
        return f"nd{self._counter:03d}"

    # -- lookup ------------------------------------------------------------
    def find_equivalent(self, text: str, include_closed: bool = True) -> MemoryNode | None:
        best, best_sim = None, 0.0
        for node in self.nodes.values():
            if not include_closed and node.status in (STATUS_DISCONFIRMED, STATUS_RESOLVED):
                continue
            sim = similarity(node.text, text)
            if sim > best_sim:
                best, best_sim = node, sim
        return best if best and best_sim >= self.equiv_min else None

    def forbidden_inferences(self) -> list[str]:
        """Texts the supporter must not re-propose. Feeds grounding's `denied_inference` check."""
        return [n.text for n in self.nodes.values()
                if n.status == STATUS_DISCONFIRMED and n.blocked_from_reproposal]

    def denial_quotes(self) -> dict[str, str]:
        out = {}
        for n in self.nodes.values():
            if n.status == STATUS_DISCONFIRMED and n.contradicting_spans:
                out[n.text] = n.contradicting_spans[-1].get("quote", "")
        return out

    def stale_references(self) -> list[str]:
        return [n.text for n in self.nodes.values()
                if n.dormant or n.status == STATUS_RESOLVED]

    def terminal_candidates(self) -> list[MemoryNode]:
        live = [n for n in self.nodes.values()
                if n.depth == 2 and n.status in (STATUS_HYPOTHESIS, STATUS_CONFIRMED) and not n.dormant]
        return sorted(live, key=lambda n: (n.status != STATUS_CONFIRMED, -n.confidence))

    def best_terminal(self) -> MemoryNode | None:
        cands = self.terminal_candidates()
        return cands[0] if cands else None

    # -- operations --------------------------------------------------------
    def propose(self, text: str, spans: Sequence[dict | Span], confidence: float, depth: int = 1,
                parent_id: str | None = None, session_id: str = "") -> MemoryNode | None:
        """Create a hypothesis. Rejected without a validated span, or if an equivalent node is blocked.

        Callers must pass spans that have already passed grounding.validate_span - this method trusts the
        caller for validity but not for presence (PLAN R4).
        """
        span_dicts = [s.to_dict() if isinstance(s, Span) else dict(s) for s in (spans or [])]
        if not span_dicts:
            self._record("-", None, "rejected", "no grounding span", session_id=session_id,
                         extra={"claim": text})
            return None
        existing = self.find_equivalent(text)
        if existing is not None:
            if existing.blocked_from_reproposal:
                self._record(existing.node_id, existing.status, existing.status,
                             "re-proposal blocked (previously disconfirmed)", session_id=session_id,
                             extra={"claim": text})
                return None
            return self.support(existing.node_id, span_dicts[0], confidence, session_id=session_id)
        node = MemoryNode(
            node_id=self.new_node_id(), text=text, depth=depth, parent_id=parent_id,
            status=STATUS_HYPOTHESIS, confidence=float(confidence), confidence_raw=float(confidence),
            supporting_spans=span_dicts, first_seen_at=utc_stamp(), last_updated_at=utc_stamp(),
            first_seen_session=session_id, last_session=session_id,
        )
        self.nodes[node.node_id] = node
        self._record(node.node_id, None, STATUS_HYPOTHESIS, "proposed", span_dicts[0], session_id)
        return node

    def support(self, node_id: str, span: dict | Span, confidence: float | None = None,
                session_id: str = "") -> MemoryNode:
        """Add evidence without changing status. Promotion is `confirm`'s job."""
        node = self.nodes[node_id]
        sd = span.to_dict() if isinstance(span, Span) else dict(span)
        if sd not in node.supporting_spans:
            node.supporting_spans.append(sd)
        if confidence is not None:
            node.confidence_raw = max(node.confidence_raw, float(confidence))
            node.confidence = node.confidence_raw
        node.dormant = False
        node.last_updated_at = utc_stamp()
        node.last_session = session_id or node.last_session
        self._record(node_id, node.status, node.status, "supporting evidence added", sd, session_id)
        return node

    def confirm(self, node_id: str, span: dict | Span, session_id: str = "",
                explicit_affirmation: bool = True) -> MemoryNode:
        """hypothesis -> confirmed, once enough spans exist and one is an explicit user affirmation."""
        node = self.nodes[node_id]
        self.support(node_id, span, session_id=session_id)
        enough = len(node.supporting_spans) >= self.min_confirm
        if enough and explicit_affirmation and node.status == STATUS_HYPOTHESIS:
            frm, node.status = node.status, STATUS_CONFIRMED
            node.confidence = max(node.confidence, 0.85)
            node.confidence_raw = node.confidence
            node.last_updated_at = utc_stamp()
            self._record(node_id, frm, STATUS_CONFIRMED, "confirmed by explicit affirmation",
                         span if isinstance(span, dict) else span.to_dict(), session_id)
        return node

    def disconfirm(self, node_id: str, span: dict | Span, session_id: str = "") -> MemoryNode:
        """The highest-value operation here: records the denial and blocks re-proposal."""
        node = self.nodes[node_id]
        sd = span.to_dict() if isinstance(span, Span) else dict(span)
        frm = node.status
        node.contradicting_spans.append(sd)
        node.status = STATUS_DISCONFIRMED
        node.blocked_from_reproposal = True
        node.confidence = 0.0
        node.confidence_raw = 0.0
        node.last_updated_at = utc_stamp()
        node.last_session = session_id or node.last_session
        self._record(node_id, frm, STATUS_DISCONFIRMED, "seeker denied the inference", sd, session_id)
        return node

    def resolve(self, node_id: str, session_id: str = "", span: dict | Span | None = None) -> MemoryNode:
        node = self.nodes[node_id]
        frm = node.status
        node.status = STATUS_RESOLVED
        node.last_updated_at = utc_stamp()
        node.last_session = session_id or node.last_session
        sd = None if span is None else (span.to_dict() if isinstance(span, Span) else dict(span))
        if sd:
            node.supporting_spans.append(sd)
        self._record(node_id, frm, STATUS_RESOLVED, "need acted on and reported helpful", sd, session_id)
        return node

    def decay(self, gap_days: float, session_id: str = "") -> None:
        """Run at session start. Only unconfirmed hypotheses decay: a tested link stays tested."""
        if gap_days <= 0:
            return
        factor = 0.5 ** (float(gap_days) / self.half_life)
        for node in self.nodes.values():
            if node.status != STATUS_HYPOTHESIS:
                continue
            before = node.confidence
            node.confidence = node.confidence_raw * factor
            if node.confidence < self.dormant_floor and not node.dormant:
                node.dormant = True
                self._record(node.node_id, STATUS_HYPOTHESIS, STATUS_HYPOTHESIS,
                             f"decayed below floor over {gap_days:.1f} days and went dormant",
                             session_id=session_id, extra={"confidence_before": before,
                                                           "confidence_after": node.confidence})

    def reinstate(self, node_id: str, span: dict | Span, session_id: str = "") -> MemoryNode:
        """A dormant or resolved concern that resurfaces comes back at reduced confidence."""
        node = self.nodes[node_id]
        frm = node.status
        sd = span.to_dict() if isinstance(span, Span) else dict(span)
        node.status = STATUS_HYPOTHESIS
        node.dormant = False
        node.reinstated_count += 1
        node.confidence_raw = max(self.dormant_floor, node.confidence_raw * self.reinstate_factor)
        node.confidence = node.confidence_raw
        node.supporting_spans.append(sd)
        node.last_updated_at = utc_stamp()
        node.last_session = session_id or node.last_session
        self._record(node_id, frm, STATUS_HYPOTHESIS, "resurfaced after a gap, reduced confidence", sd,
                     session_id)
        return node

    def reopen(self, node_id: str, spans: Sequence[dict | Span], session_id: str = "") -> MemoryNode | None:
        """The only exit from `blocked_from_reproposal`, and deliberately hard."""
        node = self.nodes[node_id]
        sds = [s.to_dict() if isinstance(s, Span) else dict(s) for s in spans]
        if len(sds) < self.min_reopen:
            self._record(node_id, node.status, node.status,
                         f"reopen refused: {len(sds)} span(s) < {self.min_reopen} required",
                         session_id=session_id)
            return None
        frm = node.status
        node.status = STATUS_HYPOTHESIS
        node.blocked_from_reproposal = False
        node.confidence_raw = 0.3
        node.confidence = 0.3
        node.supporting_spans.extend(sds)
        node.last_updated_at = utc_stamp()
        self._record(node_id, frm, STATUS_HYPOTHESIS, "reopened on new contradicting evidence", sds[0],
                     session_id)
        return node

    # -- retrieval ---------------------------------------------------------
    def brief(self, open_questions: Sequence[str] = ()) -> dict:
        """Retrieval as a question, not a lookup (PLAN 12.1).

        Ranked by what the system does not yet know: the current best terminal hypothesis, the chain path
        to it, which links are still untested, what is forbidden, and what is already resolved.
        """
        best = self.best_terminal()
        chain = self.chain_path(best.node_id) if best else []
        untested = [
            {"node_id": n.node_id, "text": n.text, "confidence": round(n.confidence, 3),
             "spans": len(n.supporting_spans)}
            for n in sorted(self.nodes.values(), key=lambda n: -n.confidence)
            if n.status == STATUS_HYPOTHESIS and not n.dormant
        ]
        return {
            "profile_id": self.profile_id,
            "best_terminal": None if best is None else {
                "node_id": best.node_id, "text": best.text, "status": best.status,
                "confidence": round(best.confidence, 3),
            },
            "chain": [{"node_id": n.node_id, "depth": n.depth, "text": n.text, "status": n.status}
                      for n in chain],
            "untested_links": untested,
            "open_questions": list(open_questions),
            "forbidden_inferences": self.denial_quotes(),
            "resolved": [n.text for n in self.nodes.values() if n.status == STATUS_RESOLVED],
            "dormant": [n.text for n in self.nodes.values() if n.dormant],
        }

    def chain_path(self, node_id: str) -> list[MemoryNode]:
        path, seen = [], set()
        cur = self.nodes.get(node_id)
        while cur and cur.node_id not in seen:
            seen.add(cur.node_id)
            path.append(cur)
            cur = self.nodes.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(path))

    def render_brief(self, open_questions: Sequence[str] = (), token_budget: int | None = None) -> str:
        """The block injected into agent prompts. Always ALONGSIDE the raw history, never instead of it."""
        b = self.brief(open_questions)
        budget = int(token_budget or cfg("agents.memory_brief_token_budget", default=320))
        if not self.nodes:
            return "[memory] first session with this person: nothing believed yet."
        lines = ["[memory - belief about this person, not a transcript]"]
        if b["best_terminal"]:
            bt = b["best_terminal"]
            lines.append(f"best guess at the underlying need ({bt['status']}, confidence "
                         f"{bt['confidence']}): {bt['text']}")
        for link in b["untested_links"][:3]:
            lines.append(f"still untested: {link['text']} (confidence {link['confidence']})")
        for q in b["open_questions"][:3]:
            lines.append(f"open question: {q}")
        for text, quote in list(b["forbidden_inferences"].items())[:3]:
            lines.append(f"DO NOT re-propose '{text}' - they denied it: \"{quote}\"")
        for r in b["resolved"][:2]:
            lines.append(f"already worked through: {r}")
        return truncate_tokens("\n".join(lines), budget)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        write_jsonl(path, [n.to_dict() for n in self.nodes.values()])

    @staticmethod
    def load(profile_id: str, path: str | Path, log_path: str | Path | None = None) -> "NeedStateMemory":
        mem = NeedStateMemory(profile_id, log_path=log_path)
        for rec in read_jsonl(path):
            node = MemoryNode(**rec)
            mem.nodes[node.node_id] = node
            num = re.sub(r"\D", "", node.node_id)
            if num:
                mem._counter = max(mem._counter, int(num))
        return mem

    @staticmethod
    def fold(transitions: Iterable[dict], profile_id: str = "") -> dict[str, str]:
        """Replay the log to the current status per node. Used to prove the state is auditable."""
        state: dict[str, str] = {}
        for t in transitions:
            if t.get("node_id") in (None, "-"):
                continue
            if t.get("to") in (STATUS_HYPOTHESIS, STATUS_CONFIRMED, STATUS_DISCONFIRMED, STATUS_RESOLVED):
                state[t["node_id"]] = t["to"]
        return state

    def status_map(self) -> dict[str, str]:
        return {n.node_id: n.status for n in self.nodes.values()}
