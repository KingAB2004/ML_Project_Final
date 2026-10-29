# Memory metrics without a model - memory_1000_v2

## 1. Success Rate by time since the previous session

Judge Success per session (0-1), grouped by the gap before it. 95 % CI: bootstrap over profiles.

| arm | memory | session 1 | <= 7 d | 7-21 d | > 21 d |
|---|---|---|---|---|---|
| mem_needstate | needstate | 0.367 [0.352, 0.385] (n=100) | 0.363 [0.344, 0.381] (n=114) | 0.351 [0.319, 0.378] (n=47) | 0.367 [0.348, 0.391] (n=49) |

## 2. Re-proposal of denied inferences

Dialogue level: a supporter inference (>= L2) answered by an explicit seeker denial, and later supporter turns of the same profile containing 60 % of the denied sentence's content words. Lexical heuristic: a lower bound, for comparing arms. Recorded: nodes the need-state memory marked disconfirmed.

| arm | memory | sessions | denials detected | denials re-proposed | re-proposing turns (cross-session) | memory nodes | disconfirmed nodes | re-proposed after disconfirmation |
|---|---|---|---|---|---|---|---|---|
| mem_needstate | needstate | 310 | 0 | 0 | 0 (0) | 749 | 5 | 0 |

Every detected denial, with its claim and any re-proposals, is in `denials.jsonl`.

## 3. Transition recall and abstention (judge)

Transition recall: share of follow-up sessions whose post-session memory reflects what changed for the seeker (judge >= 5/7). Confident naming: share of profiles whose last session states one need as THE explanation (judge >= 5/7); on ambiguous profiles lower is better.

| arm | memory | transition recall | recall: resolved | recall: intensified | recall: displaced | confident naming, ambiguous profiles | confident naming, clear profiles |
|---|---|---|---|---|---|---|---|
| mem_needstate | needstate | 0.933 [0.897, 0.964] (n=210) | 0.883 [0.808, 0.949] (n=77) | 0.968 [0.929, 1.000] (n=95) | 0.947 [0.861, 1.000] (n=38) | 0.333 [0.111, 0.667] (n=9) | 0.363 [0.264, 0.462] (n=91) |
