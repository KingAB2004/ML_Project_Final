Time has passed between two support sessions with the same person. Advance their state.

Elapsed: {gap_days} days
Profile at the end of the previous session:
{profile_json}
What happened in the previous session (transcript):
{transcript}
Transition to apply: {transition}
  - resolved   : the need the last session worked on is largely met; it recedes
  - intensified: the need was not addressed and now presses harder
  - displaced  : a new concern takes over as the focus; the old one lingers quietly

Return JSON only:
{{
  "emotion": "...",
  "feeling": "...",
  "need_chain": [{{"depth": 0, "text": "..."}}, {{"depth": 1, "text": "..."}}, {{"depth": 2, "text": "..."}}],
  "memory": [{{"text": "...", "when_relative": "...", "salience": 0.0}}],
  "resistance_level": "low|medium|high",
  "applied": "{transition}",
  "notes": "<one sentence on what changed>"
}}

Rules: keep chain node texts stable when the need itself did not change - only a displaced concern
introduces genuinely new chain text. Decay the salience of older events with the elapsed time. Add one or
two new events consistent with the gap.
