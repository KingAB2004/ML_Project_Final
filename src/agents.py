"""Enhancement 2: the four agents, the revision loop, and the gated release (PLAN Sec. 10.2, 10.4, 15).

Decomposition is infrastructure; the contract and the gate are the claim. So the interesting code here is
not the four prompts - it is that every boundary writes an inspectable artifact, that no belief is written
without a validated span, and that release is decided by a calibrated risk threshold rather than a
hand-tuned confidence number.

All four roles are the SAME resident base model with different system prompts and temperatures. They run
sequentially: a 12 GB card cannot hold four models, and the report says so rather than implying concurrency.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import grounding
from common import (cfg, fill, fresh_turn, normalize_scale, read_prompt, render_transcript, to_float, to_int,
                    user_turns)
from conformal import Calibration, nonconformity
from memory import NeedStateMemory
from pacing import at_most, permitted_rung

FALLBACK_TEMPLATE = ("It sounds like {echo} That's worth sitting with. I'm here if you want to say more "
                     "about it.")


SECOND_PERSON = {"i": "you", "i'm": "you're", "im": "you're", "i've": "you've", "i'd": "you'd",
                 "i'll": "you'll", "me": "you", "my": "your", "mine": "yours", "myself": "yourself"}


def second_person(text: str) -> str:
    # The verb changes only after "I" ("the week was" stays as it is).
    text = re.sub(r"\bI am\b", "you are", re.sub(r"\bI was\b", "you were", text, flags=re.I), flags=re.I)
    return re.sub(r"[A-Za-z']+", lambda m: SECOND_PERSON.get(m.group().lower(), m.group()), text)


def reflective_fallback(turns: Sequence[dict]) -> str:
    """Templated L0 reflection. No new inference, no question, no model call - so it cannot itself fail.

    Reflects the first clause of the seeker's last turn in the second person. v1 echoed the whole turn
    verbatim ("It sounds like ... Thanks for checking in!"), which read as parroting.
    """
    us = user_turns(turns)
    if not us:
        return "I'm here whenever you feel like talking."
    first = re.split(r"(?<=[.!?])\s+|\s+(?:but|and|so)\s+", us[-1].get("text", "").strip())[0]
    words = first.rstrip(".!?,;").split()[:18]
    if len(words) < 3:
        return "I'm here, and there's no rush. Say as much or as little as you like."
    echo = second_person(" ".join(words))
    return FALLBACK_TEMPLATE.format(echo=echo[0].lower() + echo[1:] + ".")


def valid_item_scores(raw: Any, items: Sequence[int] = range(1, 8)) -> dict[str, float] | None:
    """Critic item scores, or None unless every item has a score on the 1-7 scale."""
    if not isinstance(raw, dict):
        return None
    scores = {str(k): to_float(v, -1.0) for k, v in raw.items()}
    if any(not 1.0 <= scores.get(str(i), -1.0) <= 7.0 for i in items):
        return None
    return {str(i): scores[str(i)] for i in items}


# --------------------------------------------------------------------------- critic + gate


@dataclass
class CriticVerdict:
    ip_pred: float = 0.0
    item_scores: dict = field(default_factory=dict)
    grounding_violations: list[dict] = field(default_factory=list)
    nonconformity: float = 0.0
    decision: str = "release"          # release | revise | fallback
    feedback: str = ""
    revision_index: int = 0
    draft_rung: str = "L0"
    parse_failed: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Critic:
    """Scores a draft on the IP rubric, runs the mechanical grounding checks, and emits the gate score.

    The IP item list lives in exactly one file (prompts/metric_ip.md, mirrored in agent_critic.md) so the
    thing being gated and the thing being measured cannot drift apart.
    """

    def __init__(self, llm, calibration: Calibration | None = None, gate: bool = True):
        self.llm = llm
        self.calibration = calibration
        self.gate = gate
        self.template = read_prompt("agent_critic.md")
        self.max_revisions = int(cfg("agents.max_revisions", default=2))

    def review(self, draft: str, turns: Sequence[dict], analyzer_json: dict, plan: str,
               permitted_rung: str, memory: NeedStateMemory | None = None, phase: str = "listening",
               revision_index: int = 0) -> CriticVerdict:
        forbidden = memory.forbidden_inferences() if memory else []
        prompt = fill(
            self.template,
            history=render_transcript(turns, numbered=True),
            analyzer_json=json.dumps(analyzer_json, ensure_ascii=False)[:4000],
            plan=plan,
            permitted_rung=permitted_rung,
            forbidden="; ".join(forbidden) or "(none recorded)",
            draft=draft,
        )
        out = self.llm.structured(prompt, required=("item_scores",))
        scores = valid_item_scores(out.get("item_scores"))
        if scores is None and not out.get("parse_failed"):
            # Off-scale or missing items (v1: the template's example zeros copied back): ask once more.
            out = self.llm.structured(f"{prompt}\n\nYour previous item_scores were invalid. Give every item "
                                      f"1-7 an integer score from 1 to 7.", required=("item_scores",))
            scores = valid_item_scores(out.get("item_scores"))
        verdict = CriticVerdict(revision_index=revision_index, parse_failed=scores is None)
        scores = scores or {}
        verdict.item_scores = scores
        if scores:
            verdict.ip_pred = normalize_scale(sum(scores.values()) / len(scores))
        else:
            verdict.ip_pred = 1.0  # unparseable critic output is treated as maximally risky
        verdict.draft_rung = out.get("draft_rung", "L0")
        verdict.feedback = out.get("feedback", "") or ""

        report = grounding.check_draft(
            draft,
            turns,
            permitted_rung=permitted_rung,
            forbidden_inferences=forbidden,
            phase=phase,
            stale_references=memory.stale_references() if memory else [],
            terminal_need_terms=[n["text"] for n in (memory.brief().get("untested_links", []) if memory else [])],
            critic_rung=verdict.draft_rung,
        )
        for cat in out.get("violations", []) or []:
            if not (isinstance(cat, str) and cat in grounding.CATEGORY_WEIGHTS) or report.has(cat):
                continue
            if cat == "denied_inference" and not forbidden:
                # A denial that was never recorded cannot be re-proposed. In v1 the critic model reported
                # this hard-veto category 326 times with nothing denied, forcing 72% of cell B to fallback.
                continue
            report.violations.append(grounding.Violation(cat, "reported by the critic model"))
        verdict.grounding_violations = [v.to_dict() for v in report.violations]
        verdict.nonconformity = nonconformity(verdict.ip_pred, report.mass())

        verdict.decision = self._decide(verdict, revision_index, report)
        return verdict

    def _decide(self, verdict: CriticVerdict, revision_index: int, report: grounding.GroundingReport) -> str:
        if not self.gate:
            # Ungated arm: the critic is advisory, everything is released, and that is the point of the cell.
            return "release"
        released = (self.calibration.releases(verdict.nonconformity) if self.calibration
                    else verdict.nonconformity <= 0.5)
        if released and not report.has("denied_inference"):
            return "release"
        return "revise" if revision_index < self.max_revisions else "fallback"


# --------------------------------------------------------------------------- the pipeline


@dataclass
class TurnResult:
    text: str
    permitted_rung: str
    analyzer: dict
    strategist: dict
    critic: dict
    path: str                      # released | revised | fallback
    revisions: int
    memory_writes: list[dict] = field(default_factory=list)
    restricted_reason: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class AgentPipeline:
    """Analyzer -> Strategist -> Generator -> Critic -> gate, with memory wired in."""

    def __init__(self, llm, memory: NeedStateMemory | None = None,
                 calibration: Calibration | None = None, gate: bool = True,
                 views: dict | None = None):
        # Per-role views over ONE resident model, so each agent keeps its own sampling temperature
        # (parsed roles cold, the generator warmer) without loading a second set of weights.
        views = views or {}
        self.llm = llm
        self.analyzer_llm = views.get("analyzer", llm)
        self.strategist_llm = views.get("strategist", llm)
        self.generator_llm = views.get("generator", llm)
        self.memory = memory
        self.critic = Critic(views.get("critic", llm), calibration=calibration, gate=gate)
        self.analyzer_tpl = read_prompt("agent_analyzer.md")
        self.strategist_tpl = read_prompt("agent_strategist.md")
        self.generator_tpl = read_prompt("agent_generator.md")
        self.max_revisions = int(cfg("agents.max_revisions", default=2))

    # -- individual agents -------------------------------------------------
    def analyze(self, turns: Sequence[dict], session_id: str, memory_block: str) -> dict:
        prompt = fill(self.analyzer_tpl, memory_block=memory_block,
                      history=render_transcript(turns, numbered=True))
        out = self.analyzer_llm.structured(prompt, required=("emotional_state",))
        return grounding.zero_unsupported_claims(out, turns, session_id=session_id)

    def strategize(self, turns: Sequence[dict], analyzer: dict, allowed: Sequence[str],
                   memory_block: str, restriction_note: str) -> dict:
        prompt = fill(self.strategist_tpl,
                      analyzer_json=json.dumps(analyzer, ensure_ascii=False)[:4000],
                      memory_block=memory_block,
                      history=render_transcript(turns, numbered=True),
                      allowed_rungs=", ".join(allowed),
                      restriction_note=restriction_note or "")
        out = self.strategist_llm.structured(prompt, required=("plan",))
        if out.get("requested_rung") not in allowed:
            out["requested_rung"] = allowed[-1]
            out["rung_clamped"] = True
        out["allowed_rungs"] = list(allowed)
        out["restricted_reason"] = restriction_note
        return out

    def generate(self, turns: Sequence[dict], plan: str, permitted_rung: str, memory_block: str,
                 revision_note: str = "") -> str:
        prompt = fill(self.generator_tpl, plan=plan, permitted_rung=permitted_rung,
                      memory_block=memory_block, history=render_transcript(turns, numbered=True),
                      revision_note=revision_note)
        return fresh_turn(self.generator_llm.chat, prompt, turns, "supporter", use_cache=not revision_note)

    # -- one full turn -----------------------------------------------------
    def step(self, turns: Sequence[dict], session_id: str = "", phase: str = "listening",
             session_index: int = 1, gap_days: float = 0.0) -> TurnResult:
        brief = self.memory.brief() if self.memory else {}
        memory_block = (self.memory.render_brief(brief.get("open_questions", []))
                        if self.memory else "[memory] disabled for this arm.")

        analyzer = self.analyze(turns, session_id, memory_block)
        top_need = max((n for n in analyzer.get("implicit_needs", []) if isinstance(n, dict)),
                       key=lambda n: to_float(n.get("confidence")), default={})
        analyzer_conf = to_float(top_need.get("confidence"))

        rung, restricted = permitted_rung(analyzer_confidence=analyzer_conf, memory_brief=brief)
        strategist = self.strategize(turns, analyzer, at_most(rung), memory_block, restricted)
        requested = strategist.get("requested_rung", rung)

        draft = self.generate(turns, strategist.get("plan", ""), requested, memory_block)
        verdict = self.critic.review(draft, turns, analyzer, strategist.get("plan", ""), requested,
                                     memory=self.memory, phase=phase, revision_index=0)
        revisions = 0
        while verdict.decision == "revise" and revisions < self.max_revisions:
            revisions += 1
            note = (f"Your previous draft was rejected: {verdict.feedback or 'too intrusive for this point'}. "
                    f"Violations: {[v['category'] for v in verdict.grounding_violations]}. Rewrite it "
                    f"shallower and stay at or below {requested}.")
            draft = self.generate(turns, strategist.get("plan", ""), requested, memory_block,
                                  revision_note=note)
            verdict = self.critic.review(draft, turns, analyzer, strategist.get("plan", ""), requested,
                                         memory=self.memory, phase=phase, revision_index=revisions)

        if verdict.decision == "fallback":
            text, path = reflective_fallback(turns), "fallback"
        else:
            text, path = draft, ("revised" if revisions else "released")

        writes = self.update_memory(analyzer, session_id) if self.memory else []
        return TurnResult(text=text, permitted_rung=rung, analyzer=analyzer, strategist=strategist,
                          critic=verdict.to_dict(), path=path, revisions=revisions,
                          memory_writes=writes, restricted_reason=restricted)

    # -- memory writes -----------------------------------------------------
    def update_memory(self, analyzer: dict, session_id: str) -> list[dict]:
        """Only grounded, non-zero-confidence claims reach the belief state (PLAN R4)."""
        writes: list[dict] = []
        for need in analyzer.get("implicit_needs", []):
            if not isinstance(need, dict):
                continue
            spans = need.get("evidence_spans") or []
            conf = to_float(need.get("confidence"))
            text = (need.get("text") or "").strip()
            if not text or not spans or conf <= 0.0:
                writes.append({"claim": text, "action": "refused", "why": "ungrounded or zero confidence"})
                continue
            hint = need.get("status_hint", "hypothesis")
            depth = to_int(need.get("depth"), 1, 1, 3)
            existing = self.memory.find_equivalent(text)
            if hint == "disconfirmed" and existing is not None:
                self.memory.disconfirm(existing.node_id, spans[-1], session_id=session_id)
                writes.append({"claim": text, "action": "disconfirmed", "node_id": existing.node_id})
                continue
            if hint == "confirmed" and existing is not None:
                self.memory.confirm(existing.node_id, spans[-1], session_id=session_id)
                writes.append({"claim": text, "action": "confirmed", "node_id": existing.node_id})
                continue
            node = self.memory.propose(text, spans, conf, depth=depth, session_id=session_id)
            writes.append({"claim": text, "action": "proposed" if node else "blocked",
                           "node_id": node.node_id if node else None})
        return writes
