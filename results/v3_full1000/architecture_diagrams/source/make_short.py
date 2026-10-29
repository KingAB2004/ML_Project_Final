s = open("architecture.py").read()
short = {
 '["situations from EmpatheticDialogues, stratified by ESConv problem type"]': '["human-written situations, stratified topics"]',
 '["feeling, past events, persona, resistance, and a hidden need chain used as ground truth"]': '["persona with hidden need chain"]',
 '["supporter and simulated person talk turn by turn: listening, then suggestion"]': '["supporter and seeker role-play"]',
 '["several sessions per person across real time gaps; the person resolves, intensifies or shifts"]': '["multi-session, real gaps, person changes"]',
 '["structure, need leaks, safety, active listening"]': '["drop leaky, unsafe, weak dialogues"]',
 '["each supporter turn: analysis, strategy, ladder rung"]': '["analysis + strategy per turn"]',
 '["input: time gap, memory line, conversation so far; target: thoughts + reply, or reply only"]': '["context in, thoughts + reply out"]',
 '["dialogues and replies stay unchanged"]': '["dialogues stay unchanged"]',
 '''["each person's sessions in order; the Analyzer writes the need-state graph after every supporter turn; "
                                         "beliefs fade over each real gap"]''': '["Analyzer builds memory per turn"]',
 '["the memory brief the supporter would have seen at every turn"]': '["brief saved at every turn"]',
 '["analysis and strategy rewritten with the brief in view, so the thoughts use what was remembered"]': '["thoughts rewritten using the brief"]',
 '''["the brief replaces 'memory: none' in every input"]''': '["brief added to every input"]',
 '''["quantized base model with a small trainable adapter", "",
          "best checkpoint chosen by validation loss"]''': '["quantized base, small trainable adapter"]',
 '''["learns to write its own analysis and strategy, then the reply",
           "", "with / without thoughts"]''': '["writes thoughts, then the reply"]',
 '["retrained on the memory-aware data, so it learns to use the brief"]': '["learns to use the brief"]',
 '''["reads the person's state and the needs behind it, each tied to their own words; "
           "writes these grounded claims to memory"]''': '["reads needs, quotes evidence"]',
 '''["how deep this turn may go: reflect, explore, tentative guess, or name the need. "
           "Low Analyzer confidence keeps it shallow"]''': '["limits how deep to probe"]',
 '''["picks the move for this turn within the cap; acts as a per-turn instruction "
           "for the model"]''': '["plans the move for this turn"]',
 '''[
           "receives the Analyzer's reading, the depth limit and the Strategist's plan as guidance, writes its own analysis + strategy, "
           "then the reply. The other agents stay the base model: the fine-tuned model brings the trained supporting style, "
           "the agents bring checked evidence, memory and pacing."]''': '["guided by agents, writes the reply"]',
 '''["checks the draft against the conversation and memory: quotes must be real, no denied need raised again, "
           "no stale reference, not deeper than the cap; estimates intrusiveness"]''': '["checks grounding and intrusiveness"]',
 '''["releases the draft only if its risk score is under a threshold calibrated "
           "on held-out conversations; otherwise revise with the Critic's note, then a safe reflective fallback"]''': '["release, revise, or safe fallback"]',
 '''["the fine-tuned model alone writes thoughts + reply from the "
      "conversation and brief; the Analyzer runs only to keep the memory"]''': '["model alone; Analyzer keeps memory"]',
 '["only the reply text is shown; analysis and strategy stay internal"]': '["only the reply is shown"]',
 '''["the person's own quotes as evidence, a status, a confidence, "
      "which sessions it came up in, and a logged history of every change"]''': '["quotes, status, confidence, history"]',
 '''["a new need, backed by the person's own words"]''': '["new need with a quote"]',
 '["a guess the supporter may test"]': '["an untested guess"]',
 '["the person clearly agreed; fades slowly; rival guesses weaken"]': '["person agreed; rivals weaken"]',
 '["faded; left out of the brief, can come back"]': '["faded; can come back"]',
 '["the person denied it; blocked, never re-proposed"]': '["denied; never proposed again"]',
 '["faded further; removed from the graph, kept in the log"]': '["removed, kept in log"]',
 '["acted on and reported helpful"]': '["need worked through"]',
 '["deliberately hard: needs fresh evidence"]': '["needs fresh evidence"]',
 '["the Analyzer writes after every supporter turn: propose, confirm, deny, resolve"]': '["Analyzer updates beliefs each turn"]',
 '["confidence fades; confirmed and often-repeated needs fade more slowly; weak nodes go dormant, then are pruned"]': '["confidence fades; weak nodes pruned"]',
 '["best guess, current chain, rivals, untested needs, denied needs, resolved needs; goes into the prompt"]': '["short brief goes into prompt"]',
}
for a, b in short.items():
    assert a in s, a[:60]
    s = s.replace(a, b)
s = s.replace('"Every step sees the conversation so far (verbatim) and the memory brief"', '"Every step sees conversation + memory brief"')
s = s.replace('d.save(OUT / "COCCON_architecture")', 'd.save(OUT / "COCCON_architecture_short")')
open("architecture_short.py", "w").write(s)
