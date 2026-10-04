"""The typed grounding contract (PLAN Sec. 10.1). Built before the agents, on purpose.

Two rules from the SOP:
  (1) structured output augments, never replaces, raw dialogue history - enforced by the agents' prompts
      and checked here only insofar as spans must point at real user turns;
  (2) nothing enters belief state ungrounded - enforced here, and memory.py refuses any write whose claim
      did not pass.

The decisive category is `denied_inference`: re-proposing an inference the user has already denied. It has
no counterpart in factual grounding and carries the highest gate weight.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from common import Span, locate_span

CATEGORY_WEIGHTS: dict[str, float] = {
    "denied_inference": 1.0,
    "unsupported_state_attribution": 0.7,
    "fabricated_fact": 0.7,
    "overreaching_depth": 0.45,
    "unsolicited_advice": 0.4,
    "stale_reference": 0.15,
}

RUNG_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}

# Cheap lexical cues. Deliberately crude: the mechanical checks must not call a model (PLAN 10.1), and a
# cue that over-fires costs a revision, while a model call on every turn costs the project's compute.
ADVICE_CUES = (
    "you should", "you could try", "have you tried", "why don't you", "what you need to do",
    "i'd suggest", "i would suggest", "my advice", "try to", "you need to",
)
ASSERTION_CUES = ("you feel", "you're feeling", "you are feeling", "you're angry", "you resent",
                  "you're afraid", "what you really want", "deep down you")
HEDGES = ("maybe", "perhaps", "it sounds like", "i might be wrong", "i wonder if", "could it be",
          "i could be off", "correct me")
# A hedge alone is not depth - "it sounds like a heavy week" is still a reflection. Depth appears when the
# hedge introduces an inference about wanting, needing or causation.
INFERENCE_MARKERS = ("want", "need", "because", "underneath", "really about", "afraid", "resent",
                     "unseen", "matter", "avoid", "protect")


@dataclass
class Violation:
    category: str
    detail: str
    weight: float = 0.0
    text: str = ""

    def __post_init__(self) -> None:
        if not self.weight:
            self.weight = CATEGORY_WEIGHTS.get(self.category, 0.3)

    def to_dict(self) -> dict:
        return {"category": self.category, "detail": self.detail, "weight": self.weight, "text": self.text}


@dataclass
class GroundingReport:
    violations: list[Violation] = field(default_factory=list)
    valid_spans: int = 0
    invalid_spans: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def mass(self) -> float:
        """Total violation weight, capped at 1.0, used as part of the conformal nonconformity score."""
        return min(1.0, sum(v.weight for v in self.violations))

    def has(self, category: str) -> bool:
        return any(v.category == category for v in self.violations)

    def to_dict(self) -> dict:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "valid_spans": self.valid_spans,
            "invalid_spans": self.invalid_spans,
            "mass": self.mass(),
        }


# --------------------------------------------------------------------------- span validation


def validate_span(span: Span | dict, turns: Sequence[dict]) -> tuple[bool, str]:
    """A span is valid only if its quote appears verbatim in the user turn it cites.

    Offsets are repaired when absent (-1) and rejected when present but wrong; a fabricated or
    paraphrased quote fails. This is the single gate every belief write passes through.
    """
    sp = span if isinstance(span, Span) else Span.from_dict(span)
    if not sp.quote or not sp.quote.strip():
        return False, "empty quote"
    turn = next((t for t in turns if t.get("turn_index") == sp.turn_index), None)
    if turn is None:
        return False, f"no turn with index {sp.turn_index}"
    if turn.get("role") != "user":
        return False, f"turn {sp.turn_index} is a {turn.get('role')} turn, not a user turn"
    found = locate_span(turn.get("text", ""), sp.quote)
    if found is None:
        return False, "quote not found verbatim in the cited turn"
    if sp.char_start >= 0 or sp.char_end >= 0:
        if (sp.char_start, sp.char_end) != found:
            return False, f"offsets {sp.char_start}:{sp.char_end} do not match {found[0]}:{found[1]}"
    return True, "ok"


def validated_spans(spans: Iterable[Span | dict], turns: Sequence[dict]) -> tuple[list[Span], list[str]]:
    good: list[Span] = []
    errors: list[str] = []
    for raw in spans or []:
        try:
            sp = raw if isinstance(raw, Span) else Span.from_dict(raw)
        except (AttributeError, TypeError, ValueError):
            errors.append(f"malformed span {str(raw)[:60]!r}")
            continue
        ok, why = validate_span(sp, turns)
        if not ok and sp.quote and sp.quote.strip():
            # Missing or wrong turn number (v1: 465 quotes cited a supporter turn). Grounding means the
            # words are the SEEKER's, verbatim: cite the latest seeker turn that contains them, if any.
            idx = next((t["turn_index"] for t in reversed(turns) if t.get("role") == "user"
                        and locate_span(t.get("text", ""), sp.quote)), None)
            if idx is not None:
                sp.turn_index, sp.char_start, sp.char_end = idx, -1, -1
                ok, why = validate_span(sp, turns)
        if ok:
            found = locate_span(next(t["text"] for t in turns if t["turn_index"] == sp.turn_index), sp.quote)
            sp.char_start, sp.char_end = found  # type: ignore[misc]
            good.append(sp)
        else:
            errors.append(f"turn {sp.turn_index}: {why}")
    return good, errors


def zero_unsupported_claims(analyzer_out: dict, turns: Sequence[dict], session_id: str = "") -> dict:
    """Every Analyzer claim keeps only validated spans; a claim left with none drops to confidence 0.

    Returns the analyzer dict with a `grounding_report` attached. Claims at confidence 0 are still visible
    to the Strategist (so it can see what was thrown out) but memory.py will refuse to write them.
    """
    report = GroundingReport()

    def fix(claim: dict) -> dict:
        raw = [{"quote": sp} if isinstance(sp, str) else sp for sp in claim.get("evidence_spans", []) or []]
        for sp in raw:
            if isinstance(sp, dict):
                sp.setdefault("session_id", session_id)
        good, errors = validated_spans(raw, turns)
        report.valid_spans += len(good)
        report.invalid_spans += len(errors)
        claim["evidence_spans"] = [s.to_dict() for s in good]
        if not good:
            claim["confidence"] = 0.0
            claim["grounding_error"] = errors or ["no evidence span supplied"]
        return claim

    if isinstance(analyzer_out.get("emotional_state"), dict):
        fix(analyzer_out["emotional_state"])
    if isinstance(analyzer_out.get("resistance_estimate"), dict):
        fix(analyzer_out["resistance_estimate"])
    analyzer_out["implicit_needs"] = [fix(n) for n in analyzer_out.get("implicit_needs", [])
                                      if isinstance(n, dict)]
    analyzer_out["grounding_report"] = report.to_dict()
    return analyzer_out


# --------------------------------------------------------------------------- draft checks


def _lower(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def classify_rung(draft: str, terminal_need_terms: Sequence[str] = ()) -> str:
    """Crude rung classifier for the mechanical depth check.

    The Critic also reports a rung from the model side; disagreement between the two is resolved toward
    the deeper rung, because over-estimating depth is the safe error here.
    """
    low = _lower(draft)
    names_need = any(term and term in low for term in (t.lower() for t in terminal_need_terms))
    hedged = any(h in low for h in HEDGES)
    asserts_state = any(c in low for c in ASSERTION_CUES)
    asks = "?" in draft
    if names_need and not hedged:
        return "L3"
    if names_need or (asserts_state and hedged):
        return "L2"
    if asserts_state:
        return "L3"
    if hedged and any(m in low for m in INFERENCE_MARKERS):
        return "L2"
    if asks:
        return "L1"
    return "L0"


def check_draft(
    draft: str,
    turns: Sequence[dict],
    permitted_rung: str = "L3",
    forbidden_inferences: Sequence[str] = (),
    phase: str = "listening",
    stale_references: Sequence[str] = (),
    terminal_need_terms: Sequence[str] = (),
    critic_rung: str | None = None,
) -> GroundingReport:
    """Mechanical checks on a drafted supporter turn. No model calls."""
    report = GroundingReport()
    low = _lower(draft)

    for denied in forbidden_inferences:
        if denied and _overlaps(low, denied):
            report.violations.append(Violation("denied_inference",
                                               f"re-proposes an inference the seeker denied: '{denied}'",
                                               text=denied))

    if any(cue in low for cue in ASSERTION_CUES) and not any(h in low for h in HEDGES):
        report.violations.append(Violation("unsupported_state_attribution",
                                           "asserts a state about the seeker without hedging or evidence"))

    drafted_rung = classify_rung(draft, terminal_need_terms)
    if critic_rung in RUNG_ORDER:
        drafted_rung = max(drafted_rung, critic_rung, key=lambda r: RUNG_ORDER[r])
    if RUNG_ORDER.get(drafted_rung, 0) > RUNG_ORDER.get(permitted_rung, 3):
        report.violations.append(Violation(
            "overreaching_depth", f"draft reads as {drafted_rung} but only {permitted_rung} is permitted"))

    if phase == "listening" and any(cue in low for cue in ADVICE_CUES):
        report.violations.append(Violation("unsolicited_advice",
                                           "advice given during the listening phase"))

    for stale in stale_references:
        if stale and _overlaps(low, stale):
            report.violations.append(Violation("stale_reference",
                                               f"refers to a decayed or resolved memory: '{stale}'"))
    return report


def check_fabricated_facts(draft: str, turns: Sequence[dict], candidate_facts: Sequence[str]) -> list[Violation]:
    """A claimed life fact must appear verbatim in some user turn."""
    said = " ".join(t.get("text", "") for t in turns if t.get("role") == "user").lower()
    out = []
    for fact in candidate_facts:
        if fact and fact.lower() not in said:
            out.append(Violation("fabricated_fact", f"'{fact}' does not appear in any seeker turn",
                                 text=fact))
    return out


def _overlaps(haystack_low: str, phrase: str, min_ratio: float = 0.6) -> bool:
    """True when most content words of `phrase` appear in the text. Substring match is too brittle."""
    words = [w for w in re.findall(r"\b\w{4,}\b", phrase.lower())]
    if not words:
        return phrase.lower() in haystack_low
    hits = sum(1 for w in words if w in haystack_low)
    return hits / len(words) >= min_ratio
