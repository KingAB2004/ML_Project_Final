from draw import Diagram, MUTED
from pathlib import Path
import sys

OUT = Path(sys.argv[1]).resolve()
d = Diagram(1800, 1300, "COCCON: overall architecture",
            "Proactive emotional support across sessions: the dataset, fine-tuning, the agent pipeline around the fine-tuned "
            "model, and the need-state memory that links sessions. Solid = built, dashed = planned.")
d.legend(1420, 22, [("data", "data"), ("model", "fine-tuned model")])
d.legend(1580, 22, [("agent", "agents"), ("memory", "memory")])
d.legend(1700, 22, [("gate", "gate"), ("planned", "planned")])

TOP, H = 92, 700
# ================================================= 1. dataset creation
d.panel(28, TOP, 540, H, "1  DATASET CREATION")
x, w = 44, 244
main = [("Seeds", ["human-written situations, stratified topics"], 86),
        ("Profiles  (Mistral-Nemo)", ["persona with hidden need chain"], 86),
        ("Dialogues  (Qwen2.5-7B)", ["supporter and seeker role-play"], 86),
        ("Linked sessions", ["multi-session, real gaps, person changes"], 86),
        ("Quality filter", ["drop leaky, unsafe, weak dialogues"], 70),
        ("Annotate  (Mistral-Nemo)", ["analysis + strategy per turn"], 70),
        ("SFT data", ["context in, thoughts + reply out"], 86)]
y, boxes = 134, []
for t, ls, h in main:
    boxes.append(d.box(x, y, w, h, t, ls, "data", size=11.5))
    y += h + 12
for a, b in zip(boxes, boxes[1:]):
    d.arrow([(x + w / 2, a[1] + a[3]), (x + w / 2, b[1])])

bx = 308
d.text(bx, 128, "Memory-aware data (Option B)", 12.5, "#55555d", bold=True)
branch = [("Start from the annotated corpus", ["dialogues stay unchanged"], 88),
          ("Replay through the memory", ["Analyzer builds memory per turn"], 120),
          ("Record the brief", ["brief saved at every turn"], 80),
          ("Re-annotate later sessions", ["thoughts rewritten using the brief"], 100),
          ("Memory-aware SFT data", ["brief added to every input"], 80)]
y, bboxes = 144, []
for t, ls, h in branch:
    bboxes.append(d.box(bx, y, w, h, t, ls, "planned", size=11.5))
    y += h + 24
for a, b in zip(bboxes, bboxes[1:]):
    d.arrow([(bx + w / 2, a[1] + a[3]), (bx + w / 2, b[1])], dashed=True)
ann = boxes[5]
d.arrow([(x + w, ann[1] + ann[3] / 2), (298, ann[1] + ann[3] / 2), (298, 188), (bx, 188)], dashed=True)

# ================================================= 2. fine-tuning
d.panel(592, TOP, 240, H, "2  FINE-TUNING")
fx, fw = 606, 212
q = d.box(fx, 134, fw, 150, "QLoRA", ["quantized base, small trainable adapter"], "model", size=11.5)
ad = d.box(fx, 330, fw, 150, "Fine-tuned supporter", ["writes thoughts, then the reply"], "model", size=11.5)
ma = d.box(fx, 600, fw, 150, "Memory-aware supporter", ["learns to use the brief"], "planned", size=11.5)
d.arrow([(fx + fw / 2, 284), (fx + fw / 2, 330)])
sft = boxes[6]
d.arrow([(x + w, sft[1] + 43), (298, sft[1] + 43), (298, 780), (584, 780), (584, 209), (fx, 209)])
mem_sft = bboxes[4]
d.arrow([(bx + w, mem_sft[1] + mem_sft[3] / 2), (fx, mem_sft[1] + mem_sft[3] / 2)], dashed=True)

# ================================================= 3. supporter at inference
PX = 856
d.panel(PX, TOP, 916, H, "3  SUPPORTER AT INFERENCE: the agents around the fine-tuned model")
gx, gw = 884, 872
inp = d.box(gx, 132, gw, 58, "Every step sees conversation + memory brief", [], "plain", align="center")
r2, h2 = 222, 132
an = d.box(gx, r2, 272, h2, "Analyzer", ["reads needs, quotes evidence"], "agent", size=11.5)
pc = d.box(gx + 300, r2, 272, h2, "Pacing cap", ["limits how deep to probe"], "plain", size=11.5)
st = d.box(gx + 600, r2, 272, h2, "Strategist", ["plans the move for this turn"], "agent", size=11.5)
for xx in (gx + 136, gx + 436, gx + 736):
    d.arrow([(xx, 190), (xx, r2)])
d.arrow([(gx + 272, r2 + 66), (gx + 300, r2 + 66)])
d.arrow([(gx + 572, r2 + 66), (gx + 600, r2 + 66)])

r3, h3 = 404, 120
rs = d.box(gx, r3, gw, h3, "Fine-tuned model as the Responder", ["guided by agents, writes the reply"], "model", size=11.5, tag="planned run")
for xx, lab in ((gx + 136, "reading"), (gx + 436, "depth limit"), (gx + 736, "plan")):
    d.arrow([(xx, r2 + h2), (xx, r3)], lab, lpos=(xx, r2 + h2 + 30))
d.arrow([(ad[0] + fw, 405), (PX - 2, 405), (PX - 2, r3 + 60), (gx, r3 + 60)])
d.arrow([(PX - 2, r3 + 60), (PX - 2, 738), (gx, 738)])

r4, h4 = 566, 104
cr = d.box(gx, r4, 420, h4, "Critic", ["checks grounding and intrusiveness"], "agent", size=11.5)
gt = d.box(gx + 452, r4, 420, h4, "Calibrated gate", ["release, revise, or safe fallback"], "gate", size=11.5)
d.arrow([(gx + 210, r3 + h3), (gx + 210, r4)], "draft", lpos=(gx + 210, r3 + h3 + 22))
d.arrow([(gx + 420, r4 + 48), (gx + 452, r4 + 48)])
d.arrow([(gx + 560, r4), (gx + 560, r3 + h3)], "revise with note", lpos=(gx + 560, r3 + h3 + 22))

r5, h5 = 694, 88
d.box(gx, r5, 420, h5, "One-call mode  (fine-tuned runs so far)", ["model alone; Analyzer keeps memory"], "model", size=11.5)
d.box(gx + 452, r5, 420, h5, "Reply to the person", ["only the reply is shown"], "result", size=11.5)
d.arrow([(gx + 662, r4 + h4), (gx + 662, r5)], "release", lpos=(gx + 662, r4 + h4 + 14))
d.arrow([(gx + 420, r5 + 44), (gx + 452, r5 + 44)])
# Analyzer writes memory: down the left margin of the panel
GREEN = "#1baf7a"
d.arrow([(gx, r2 + 100), (870, r2 + 100), (870, 816)], "Analyzer writes", lpos=(930, 806), color=GREEN)
d.arrow([(1764, 816), (1764, 161), (gx + gw, 161)], "brief", lpos=(1730, 806), color=GREEN)

# ================================================= 4. need-state memory
MY = 816
d.panel(28, MY, 1744, 468, "4  NEED-STATE MEMORY: a belief about the person, carried across sessions")
# structure
d.text(44, MY + 56, "Structure", 13, "#0d6b49", bold=True)
sw = 160
s_root = d.box(44, MY + 70, sw, 64, "Surface feeling", ["root of a chain"], "memory", size=11)
s_root2 = d.box(44 + sw + 24, MY + 70, sw, 64, "Another feeling", ["second root"], "memory", size=11)
s_mid = d.box(44, MY + 176, sw, 64, "Intermediate need", ["hangs under it"], "memory", size=11)
s_riv = d.box(44 + sw + 24, MY + 176, sw, 64, "Rival need", ["sibling explanation"], "memory", size=11)
s_deep = d.box(44, MY + 282, sw, 64, "Deeper need", ["closer to the core"], "memory", size=11)
d.arrow([(124, MY + 134), (124, MY + 176)])
d.arrow([(124, MY + 155), (308, MY + 155), (308, MY + 176)])
d.arrow([(124, MY + 240), (124, MY + 282)])
d.box(44, MY + 366, 344, 88, "Every node keeps", ["quotes, status, confidence, history"], "plain", size=11)

# lifecycle
LX = 420
d.text(LX, MY + 56, "Lifecycle of a node", 13, "#0d6b49", bold=True)
c1, c2, c3, bw, bh = LX, LX + 300, LX + 600, 230, 88
q1, q2, q3 = MY + 70, MY + 220, MY + 366
d.box(c1, q1, bw, bh, "Analyzer claim", ["new need with a quote"], "agent", size=11)
d.box(c2, q1, bw, bh, "HYPOTHESIS", ["an untested guess"], "memory", size=11)
d.box(c3, q1, bw, bh, "CONFIRMED", ["person agreed; rivals weaken"], "memory", size=11)
d.box(c1, q2, bw, bh, "DORMANT", ["faded; can come back"], "plain", size=11)
d.box(c3, q2, bw, bh, "DISCONFIRMED", ["denied; never proposed again"], "gate", size=11)
d.box(c1, q3, bw, bh, "PRUNED", ["removed, kept in log"], "result", size=11)
d.box(c2, q3, bw, bh, "RESOLVED", ["need worked through"], "result", size=11)
d.box(c3, q3, bw, bh, "Reopen", ["needs fresh evidence"], "plain", size=11)
d.arrow([(c1 + bw, q1 + 44), (c2, q1 + 44)], "propose", lpos=(c1 + bw + 35, q1 + 36))
d.arrow([(c2 + bw, q1 + 30), (c3, q1 + 30)], "confirm", lpos=(c2 + bw + 35, q1 + 22))
d.arrow([(c2 + bw - 14, q1 + bh), (c2 + bw - 14, q2 + 44), (c3, q2 + 44)], "deny", lpos=(c2 + bw + 35, q2 + 36))
d.arrow([(c2 + 30, q1 + bh), (c2 + 30, q1 + bh + 26), (c1 + 115, q1 + bh + 26), (c1 + 115, q2)], "fades", lpos=(c1 + 200, q1 + bh + 20))
d.arrow([(c1 + bw, q2 + 60), (c2 + 110, q2 + 60), (c2 + 110, q1 + bh)], "reinstate, lower confidence", lpos=(c2 + 40, q2 + 54))
d.arrow([(c2 + 175, q1 + bh), (c2 + 175, q3)], "resolve", lpos=(c2 + 175, q3 - 24))
d.arrow([(c1 + 115, q2 + bh), (c1 + 115, q3)], "fades further", lpos=(c1 + 115, q3 - 24))
d.arrow([(c3 + bw / 2, q2 + bh), (c3 + bw / 2, q3)])

# across sessions
AX, aw = 1290, 466
d.text(AX, MY + 56, "Across sessions", 13, "#0d6b49", bold=True)
steps = [("During a session", ["Analyzer updates beliefs each turn"], "agent"),
         ("Over the real gap", ["confidence fades; weak nodes pruned"], "memory"),
         ("Brief for the next session", ["short brief goes into prompt"], "memory")]
y = MY + 70
for i, (t, ls, k) in enumerate(steps):
    d.box(AX, y, 380, 100, t, ls, k, size=11.5)
    if i < 2:
        d.arrow([(AX + 200, y + 100), (AX + 200, y + 128)])
    y += 128
d.arrow([(AX + 380, MY + 70 + 256 + 50), (AX + 422, MY + 70 + 256 + 50), (AX + 422, MY + 120), (AX + 380, MY + 120)],
        "next session", lpos=(AX + 422, MY + 250))
d.save(OUT / "COCCON_architecture_short")
