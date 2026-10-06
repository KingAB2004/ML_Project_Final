"""The monolithic listener: one prompt, one forward pass, Analysis + Strategy + response together.

This is cell A/B of the 2x2. It must use the SAME adapter and the SAME target format as the fine-tuned
supporter, or the ablation confounds decomposition with training. The gate can still be applied on top,
which is what separates cell A from cell B.
"""
from __future__ import annotations

from typing import Sequence

from agents import Critic, TurnResult, reflective_fallback
from build_sft import SUPPORTER_SYSTEM, gap_statement, parse_target, render_input
from common import cfg, fresh_turn
from conformal import Calibration
from pacing import permitted_rung


class MonolithicListener:
    """Same step() contract as AgentPipeline, so evaluate.py does not branch on architecture."""

    def __init__(self, llm, memory=None, calibration: Calibration | None = None, gate: bool = False,
                 memory_writer=None):
        self.llm = llm
        self.memory = memory
        # An AgentPipeline over the same memory whose observe() writes it (arm switch memory_writer). Without one
        # this arm only reads the memory, and a need-state brief never gets past "first session".
        self.memory_writer = memory_writer
        self.gate = gate
        # The critic role's own sampling (low temperature), as in the decomposed pipeline - not the supporter's.
        critic_llm = llm.view("critic") if hasattr(llm, "view") else llm
        self.critic = Critic(critic_llm, calibration=calibration, gate=gate) if gate else None
        self.max_revisions = int(cfg("agents.max_revisions", default=2))

    def _reply(self, raw: str, prompt: str) -> dict:
        """Parsed reply; one fresh attempt when the generation held only analysis/strategy and no reply."""
        parts = parse_target(raw)
        if not parts["response"]:
            parts = parse_target(self.llm.chat(
                f"{prompt}\n\nYour previous answer had no reply for the person. Put 1 to 3 sentences "
                f"inside <response></response>.", system=SUPPORTER_SYSTEM, use_cache=False))
        return parts

    def step(self, turns: Sequence[dict], session_id: str = "", phase: str = "listening",
             session_index: int = 1, gap_days: float = 0.0) -> TurnResult:
        brief = self.memory.brief() if self.memory else {}
        memory_block = (self.memory.render_brief(brief.get("open_questions", []))
                        if self.memory else "[memory] none.")
        # No analyzer state exists in this arm, so only the memory cap applies.
        rung, restricted = permitted_rung(analyzer_confidence=1.0, memory_brief=brief)

        prompt = render_input(list(turns), upto_index=max(t["turn_index"] for t in turns) + 1,
                              memory_block=memory_block,
                              gap_statement=gap_statement(gap_days, session_index) if session_index > 1 else "")
        parts = self._reply(fresh_turn(self.llm.chat, prompt, turns, "supporter", system=SUPPORTER_SYSTEM),
                            prompt)
        draft = parts["response"]

        analyzer_view = {"emotional_state": {"label": parts["analysis"], "confidence": None,
                                            "evidence_spans": []},
                         "implicit_needs": [], "note": "monolithic arm: no inspectable analyzer state"}
        critic_out: dict = {}
        revisions = 0
        path = "released"
        if self.critic is not None and draft:
            verdict = self.critic.review(draft, turns, analyzer_view, parts["strategy"], rung,
                                         memory=self.memory, phase=phase, revision_index=0)
            while verdict.decision == "revise" and revisions < self.max_revisions:
                revisions += 1
                revise_prompt = (f"{prompt}\n\nYour previous reply was rejected as too intrusive "
                                 f"({verdict.feedback}). Rewrite it shallower, at or below {rung}.")
                parts = self._reply(self.llm.chat(revise_prompt, system=SUPPORTER_SYSTEM, use_cache=False),
                                    revise_prompt)
                draft = parts["response"]
                if not draft:
                    break
                verdict = self.critic.review(draft, turns, analyzer_view, parts["strategy"], rung,
                                             memory=self.memory, phase=phase, revision_index=revisions)
            critic_out = verdict.to_dict()
            if verdict.decision == "fallback" or not draft:
                draft, path = reflective_fallback(turns), "fallback"
            elif revisions:
                path = "revised"

        if not draft:
            # Only thoughts came back, twice: a templated reflection, never the raw generation.
            draft, path = reflective_fallback(turns), "fallback"
        return TurnResult(text=draft, permitted_rung=rung, analyzer=analyzer_view,
                          strategist={"plan": parts["strategy"], "requested_rung": rung},
                          critic=critic_out, path=path, revisions=revisions,
                          memory_writes=self.memory_writer.observe(turns, session_id) if self.memory_writer else [],
                          restricted_reason=restricted)
