"""Demo server: the replay page plus a live chat through the real pipeline, standard library only.

    python demo/server.py                                     # replay only, any machine, http://localhost:8765
    python demo/server.py --live --backend ollama             # + live chat: four agents, conformal gate, memory
    python demo/server.py --live --backend transformers \\
        --adapter runs/full_1000/adapter_with_thoughts        # live with the fine-tuned Generator (step 12)

Live mode keeps one model resident (the project rule); the Analyzer, Strategist, Generator, Critic and the
simulated seeker are views of it, exactly as in evaluate.py. Requests are served one at a time. No judge runs
live (the judge must not share the GPU with the generator), so live turns show the Critic's predicted IP; the
judge's scores are on the replayed test conversations.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

SKELETON = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>:root{padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}'
            'body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}'
            '[hidden]{display:none!important}</style></head><body>{page}</body></html>')


class Live:
    """One person talking to the pipeline: memory, current session's turns, and the session counter."""

    def __init__(self, args):
        from agents import AgentPipeline
        from common import cfg, load_config, read_jsonl
        from conformal import Calibration
        from llm import LLM

        self.AgentPipeline, self.cfg = AgentPipeline, cfg
        if args.adapter:
            load_config("models")["roles"]["supporter"]["adapter"] = args.adapter
        role = "supporter" if args.adapter else "generator"
        self.llm = LLM(role, backend=args.backend)
        self.views = {r: self.llm.view(r if r != "generator" else "supporter")
                      for r in ("analyzer", "strategist", "critic", "generator")}
        self.sim = self.llm.view("simulator")
        cal = Path(args.calibration)
        if not cal.is_absolute() and not cal.exists():
            cal = ROOT / cal                     # a repo-relative path given from another directory
        cal = cal.resolve()
        self.calibration = Calibration.load(cal) if cal.exists() else None
        self.gate = not args.no_gate
        self.profiles = read_jsonl(ROOT / "data/profiles/profiles_test.jsonl")
        self.info = {"backend": args.backend, "adapter": args.adapter, "gate": self.gate,
                     "calibration": (str(cal.relative_to(ROOT)) if cal.is_relative_to(ROOT) else str(cal))
                                    if self.calibration else None,
                     "lambda": getattr(self.calibration, "lambda_hat", None),
                     "profiles": len(self.profiles)}
        self.lock = threading.Lock()
        self.reset({})

    def reset(self, body: dict) -> dict:
        from memory import NeedStateMemory
        self.memory = NeedStateMemory("live")
        self.pipeline = self.AgentPipeline(self.views["generator"], views=self.views, memory=self.memory,
                                           calibration=self.calibration, gate=self.gate)
        pid = body.get("profile")
        self.profile = next((p for p in self.profiles if p["profile_id"] == pid), None) if pid else None
        self.session, self.gap = 1, 0.0
        openers = self.cfg("corpus.openers", default=["Hey - how are things today?"])
        return self._open(random.choice(openers))

    def _open(self, opener: str) -> dict:
        from build_data import graph_state
        self.turns = [{"turn_index": 0, "role": "supporter", "text": opener, "phase": "listening",
                       "ladder_rung": "L0"}]
        return {"session": self.session, "gap": self.gap, "graph": graph_state(self.memory, self.table()),
                "nodes": self.table(), "turns": [{"i": 0, "role": "supporter", "text": opener, "rung": "L0",
                                                   "source": "opener_pool"}],
                "profile": self.profile_view()}

    def table(self) -> dict:
        return {n.node_id: [n.text, n.depth] for n in self.memory.nodes.values()}

    def profile_view(self) -> dict | None:
        p = self.profile
        if not p:
            return None
        return {"id": p["profile_id"], "emotion": p.get("emotion"), "feeling": p.get("feeling"),
                "terminal_need": p.get("terminal_need"), "problem": p.get("problem_type"),
                "resistance": p.get("resistance_level"),
                "chain": [c.get("text") for c in p.get("need_chain") or []]}

    def new_session(self, body: dict) -> dict:
        from build_data import new_transitions
        from sessions import gap_phrase
        self.gap = float(body.get("gap") or 7)
        self.session += 1
        mark = len(self.memory.transitions)
        self.memory.decay(self.gap, session_id=f"live-s{self.session}")
        out = self._open(f"Hey - it's been a little while. How have things been {gap_phrase(self.gap)}?")
        out["events"] = new_transitions(self.memory, mark)
        return out

    def seeker_turn(self) -> str:
        from common import fresh_turn
        from dialogue import seeker_prompt, simulator_system
        if not self.profile:
            raise ValueError("pick a test person first: the simulated seeker plays one")
        context = "" if self.session == 1 else f"This is a follow-up conversation, about {self.gap:.0f} days later."
        system = simulator_system(self.profile, context)
        return fresh_turn(self.sim.chat, seeker_prompt(self.turns), self.turns, "seeker", system=system)

    def turn(self, body: dict) -> dict:
        from build_data import compact_step, graph_state, new_transitions
        text = (body.get("text") or "").strip() or self.seeker_turn()
        sid = f"live-s{self.session}"
        self.turns.append({"turn_index": len(self.turns), "role": "user", "text": text, "phase": "listening"})
        mark = len(self.memory.transitions)
        r = self.pipeline.step(self.turns, session_id=sid, session_index=self.session, gap_days=self.gap)
        self.turns.append({"turn_index": len(self.turns), "role": "supporter", "text": r.text,
                           "phase": "listening", "ladder_rung": r.permitted_rung})
        step = compact_step({**r.to_dict(), "turn_index": len(self.turns) - 1})
        return {"seeker": {"i": len(self.turns) - 2, "role": "seeker", "text": text},
                "supporter": {"i": len(self.turns) - 1, "role": "supporter", "text": r.text,
                              "rung": r.permitted_rung, "step": step, "graph": graph_state(self.memory, None),
                              "events": new_transitions(self.memory, mark), "brief": self.memory.render_brief()},
                "nodes": self.table()}


class Handler(SimpleHTTPRequestHandler):
    live: Live | None = None
    live_error: str | None = None

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)

    def send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = SKELETON.replace("{page}", (STATIC / "app.html").read_text())
            body = page.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            live = self.live
            self.send_json({"live": live is not None, "error": self.live_error,
                            **(live.info if live else {}),
                            "people": [{"id": p["profile_id"], "problem": p.get("problem_type"),
                                        "resistance": p.get("resistance_level")} for p in (live.profiles if live else [])]})
        else:
            super().do_GET()

    def do_POST(self):
        routes = {"/api/live/reset": "reset", "/api/live/turn": "turn", "/api/live/session": "new_session"}
        if self.path not in routes:
            return self.send_json({"error": "unknown route"}, HTTPStatus.NOT_FOUND)
        if self.live is None:
            return self.send_json({"error": self.live_error or "live mode is off: start the server with --live"},
                                  HTTPStatus.SERVICE_UNAVAILABLE)
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            with self.live.lock:
                self.send_json(getattr(self.live, routes[self.path])(body))
        except ValueError as e:
            self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except Exception as e:                  # the page shows it; the server keeps running
            self.send_json({"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--live", action="store_true", help="load the model and enable the live chat")
    ap.add_argument("--backend", default="ollama", help="ollama | transformers | vllm | echo")
    ap.add_argument("--adapter", default=None, help="fine-tuned Generator adapter (transformers backend)")
    ap.add_argument("--calibration", default=str(ROOT / "results/v3_full1000/baselines_1000/conformal/calibration.json"),
                    help="gate calibration (use sft_agents/conformal/calibration.json with --adapter)")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    if not (STATIC / "data/replay.json").exists():
        sys.exit("demo/static/data/replay.json is missing: run python demo/build_data.py first")
    if args.live:
        try:
            Handler.live = Live(args)
            print(f"live mode: {Handler.live.info}")
        except Exception as e:
            Handler.live_error = f"live mode failed to start: {type(e).__name__}: {e}"
            print(Handler.live_error)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"demo on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
