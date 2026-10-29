"""Talk to the supporter in the terminal, through the same pipeline classes evaluate.py uses.

    python scripts/chat.py                    # fine-tuned 7B, one call, no memory      (sft_with_thoughts)
    python scripts/chat.py --memory           # + need-state memory, base Analyzer writes it (sft7b_multi_mem)
    python scripts/chat.py --agents           # Analyzer -> Strategist -> fine-tuned Generator -> Critic + gate
                                              #   + need-state memory                     (step 12, not yet run)
    python scripts/chat.py --no-adapter ...   # same, on the untuned base
    python scripts/chat.py --adapter runs/full_1000/adapter_wo_thoughts

One 4-bit 7B is resident; the adapter is on for the supporter and off for analyzer/strategist/critic.
You play the seeker. Commands: /mem (print the memory brief), /session N (next session, N days later),
/reset, /quit.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import AgentPipeline
from baselines.listener_mono import MonolithicListener
from common import load_config
from conformal import Calibration
from llm import LLM
from memory import NeedStateMemory

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default=str(ROOT / "runs/full_1000/adapter_with_thoughts"))
ap.add_argument("--no-adapter", action="store_true")
ap.add_argument("--memory", action="store_true")
ap.add_argument("--agents", action="store_true")
ap.add_argument("--calibration", default=str(ROOT / "runs/baselines_1000/conformal/calibration.json"))
ap.add_argument("--backend", default="transformers")
args = ap.parse_args()

load_config("models")["roles"]["supporter"]["adapter"] = None if args.no_adapter else args.adapter
llm = LLM("supporter", backend=args.backend)
views = {r: llm.view("supporter" if r == "generator" else r) for r in ("analyzer", "strategist", "critic", "generator")}


def new_memory():
    return NeedStateMemory("chat") if (args.memory or args.agents) else None


def new_pipeline(memory):
    if args.agents:
        cal = Calibration.load(args.calibration) if Path(args.calibration).exists() else None
        return AgentPipeline(views["generator"], views=views, memory=memory, calibration=cal, gate=True)
    writer = AgentPipeline(views["generator"], views=views, memory=memory, gate=False) if memory else None
    return MonolithicListener(llm, memory=memory, memory_writer=writer)


memory = new_memory()
pipeline = new_pipeline(memory)
turns, session, gap = [], 1, 0.0
print("You are the seeker. /mem /session N /reset /quit\n")
while True:
    try:
        text = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text:
        continue
    if text == "/quit":
        break
    if text == "/reset":
        memory = new_memory()
        pipeline, turns, session, gap = new_pipeline(memory), [], 1, 0.0
        print("-- new person --\n")
        continue
    if text == "/mem":
        print(memory.render_brief() if memory else "[memory] off (use --memory or --agents)", "\n")
        continue
    if text.startswith("/session"):
        gap = float(text.split()[1]) if len(text.split()) > 1 else 7.0
        if memory:
            memory.decay(gap, session_id=f"chat-s{session + 1}")
        turns, session = [], session + 1
        print(f"-- session {session}, {gap:.0f} days later --\n")
        continue

    turns.append({"turn_index": len(turns), "role": "user", "text": text})
    r = pipeline.step(turns, session_id=f"chat-s{session}", session_index=session, gap_days=gap)
    if r.analyzer.get("emotional_state"):
        print(f"  [analysis] {r.analyzer['emotional_state'].get('label')}")
    print(f"  [strategy] {r.strategist.get('plan')}")
    c = r.critic or {}
    gate = (f", draft rung {c.get('draft_rung')}, score {c.get('nonconformity', 0):.2f}, "
            f"revisions {r.revisions}, violations {[v['category'] for v in c.get('grounding_violations', [])]}"
            if c else "")
    print(f"  [gate]     ceiling {r.permitted_rung}{gate}, path {r.path}")
    if r.memory_writes:
        print(f"  [memory]   {len(r.memory_writes)} write(s)")
    print(f"bot> {r.text}\n")
    turns.append({"turn_index": len(turns), "role": "supporter", "text": r.text})
