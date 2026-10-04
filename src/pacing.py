"""The disclosure ladder (PLAN Sec. 13.1), kept because Enhancement 2 is built on it.

The modified SOP drops Enhancement 5 (adaptive disclosure pacing): there is no readiness value, no
privileged teacher and no distillation here any more. What remains is the graded ladder itself, which E2
needs in two places:

  - the Strategist's allowed rungs, restricted to exploratory moves when Analyzer confidence is low;
  - the Critic's `overreaching_depth` grounding check on the drafted turn.

Enforcing depth in the prompt alone does not work - models drift upward the moment they think they have
figured the person out - so the cap is computed here and checked again in grounding.check_draft.
"""
from __future__ import annotations

from common import cfg

LADDER = ("L0", "L1", "L2", "L3")
RUNG_INDEX = {r: i for i, r in enumerate(LADDER)}

LADDER_DESCRIPTION = {
    "L0": "reflective acknowledgement - restate and validate, no inference, no question beyond an invitation",
    "L1": "open exploration - open, non-leading questions about what they already raised",
    "L2": "tentative inference - a hedged, checkable guess about an intermediate need",
    "L3": "naming the terminal need explicitly and working with it",
}


def at_most(rung: str) -> list[str]:
    return list(LADDER[: RUNG_INDEX.get(rung, 0) + 1])


def cap(a: str, b: str) -> str:
    """The stricter of two rungs."""
    return a if RUNG_INDEX.get(a, 0) <= RUNG_INDEX.get(b, 0) else b


def permitted_rung(analyzer_confidence: float = 1.0, memory_brief: dict | None = None,
                   confidence_floor: float | None = None, ceiling: str = "L3") -> tuple[str, str]:
    """Highest rung the supporter may use this turn, and why it was restricted.

    Two caps, both from Enhancement 2 and 4 rather than from a pacing policy:
      1. low Analyzer confidence restricts to exploratory moves - uncertainty must produce gentleness;
      2. L3 additionally requires a CONFIRMED terminal need in memory, never a mere hypothesis.
    """
    floor = float(cfg("agents.analyzer_confidence_floor", default=0.45)
                  if confidence_floor is None else confidence_floor)
    rung = ceiling if ceiling in RUNG_INDEX else "L3"
    reason = ""
    if float(analyzer_confidence) < floor:
        if RUNG_INDEX[rung] > RUNG_INDEX["L1"]:
            reason = (f"analyzer confidence {analyzer_confidence:.2f} below floor {floor:.2f}: "
                      f"restricted to exploratory moves")
        rung = cap(rung, "L1")
    if rung == "L3":
        best = (memory_brief or {}).get("best_terminal") or {}
        if best.get("status") != "confirmed":
            reason = reason or "terminal need is still a hypothesis: L3 withheld"
            rung = "L2"
    return rung, reason
