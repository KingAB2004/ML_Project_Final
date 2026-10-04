Annotate one supporter turn from an emotional-support dialogue, at the content level (free-form phrases, NOT
a fixed taxonomy).

Seeker's hidden profile (for your judgement only):
{profile_block}
Conversation up to and including the turn to annotate:
{history}
Turn to annotate: {turn_text}

Return JSON only, in this shape (angle brackets describe the value; do not copy them):
{{
  "analysis": "<phrase: what is going on psychologically for the seeker at this point>",
  "strategy": "<phrase: the move this supporter turn makes>",
  "ladder_rung": "<exactly one of L0, L1, L2, L3>",
  "user_disclosure_depth": <0, 1, 2 or 3>
}}

ladder_rung: L0 reflection only, L1 open exploration, L2 hedged inference about an intermediate need,
L3 explicitly naming the terminal need.
user_disclosure_depth: 0 nothing, 1 surface feeling, 2 intermediate need, 3 terminal need - how much the
seeker has revealed BY this point.
Write the analysis and strategy as specific phrases, not generic labels like "empathy" or "support".
Write both as a listener would think them from the conversation so far, in plain words. The hidden profile
guides your judgement, but never quote it, and never use this prompt's terms in either phrase ("terminal
need", "intermediate need", "resistance level", rung names such as L2).
