Build the hidden psychological profile of one person for an emotional-support simulation, from the
situation below. The point is that the surface feeling is visible and the underlying need is NOT.

Situation: {situation_text}
Stated emotion label: {emotion_label}
Problem type: {problem_type}
Ambiguous profile (two plausible terminal needs, neither decidable): {ambiguous}

Return JSON only:
{{
  "emotion": "<state plus one clause of gloss>",
  "feeling": "<one sentence in the person's own voice>",
  "need_chain": [
    {{"depth": 0, "text": "<the surface feeling as they would say it>"}},
    {{"depth": 1, "text": "<the intermediate unmet need behind it>"}},
    {{"depth": 2, "text": "<the terminal need - the thing they cannot name yet>"}}
  ],
  "memory": [{{"text": "<recent event>", "when_relative": "<e.g. last week>", "salience": 0.0}}],
  "persona_surface": {{"style": "", "verbosity": "", "tone": "", "volunteers": ""}},
  "persona_hidden": {{"disclosure_triggers": ["..."], "disclosure_blockers": ["..."]}}
}}

Rules: exactly three chain nodes. Depth 2 must NOT be a restatement of depth 0 - it must be a different
level of explanation (belonging, autonomy, being seen, safety, competence, and so on). Depth 2 must still be
reachable by a skilled listener from what this person could plausibly say. 2 to 4 memory events.
