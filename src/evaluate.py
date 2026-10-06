"""Phase 7 - the arm runner (PLAN Sec. 14.4, 15).

One arm is a set of component switches in configs/arms.yaml. This module resolves those switches into
objects, rolls out dialogues, and writes them to runs/<run_id>/dialogues/<arm>.jsonl. It does NOT score
anything: scoring is a separate pass with the judge resident alone (PLAN R2), run by metrics.py and
counterfactual.py over the files this stage produced.

  python src/evaluate.py --arm cellD_dec_gated --limit 10 --backend echo
  python src/evaluate.py --list
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agents import AgentPipeline
from baselines.listener_mono import MonolithicListener
from baselines.memory_dense import DenseMemory
from baselines.memory_event import EventMemory
from baselines.memory_none import NoMemory
from baselines.memory_summary import SummaryMemory
from common import (DATA, append_jsonl, cfg, existing_ids, fresh_turn, load_config, new_run_dir, read_jsonl,
                    render_transcript, rng_for, write_json)
from conformal import Calibration
from dialogue import seeker_prompt, simulator_system
from llm import LLM, pmap
from memory import NeedStateMemory
from sessions import advance_profile, gap_phrase, pick_transition, sample_gap_days, sample_session_count


def resolve_arm(name: str) -> dict:
    arms = load_config("arms")
    if name not in arms["arms"]:
        raise KeyError(f"unknown arm '{name}'. Known: {', '.join(sorted(arms['arms']))}")
    spec = dict(arms.get("defaults", {}))
    spec.update(arms["arms"][name])
    spec["arm"] = name
    spec["eval_set_spec"] = arms["eval_sets"].get(spec.get("eval_set", "own_heldout"), {})
    spec["adapter"] = (arms.get("adapters") or {}).get(spec.get("supporter"))
    return spec


def make_memory(kind: str, profile_id: str, llm) -> object:
    if kind == "needstate":
        return NeedStateMemory(profile_id)
    if kind == "summary":
        return SummaryMemory(profile_id, llm=llm)
    if kind == "dense":
        return DenseMemory(profile_id)
    if kind == "event":
        return EventMemory(profile_id, llm=llm)
    return NoMemory(profile_id)


def load_profiles(spec: dict, limit: int | None) -> list[dict]:
    path = Path(spec["eval_set_spec"].get("path", DATA / "profiles" / "profiles_test.jsonl"))
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"no profiles at {path} - build the corpus first, or point arms.yaml elsewhere")
    return rows[: limit or int(cfg("eval.n_dialogues_per_arm", default=150))]


def run_session(pipeline, sim_view, profile: dict, session_id: str, session_index: int, gap_days: float,
                n_supporter_turns: int, session_context: str = "", reactive: bool = False,
                corrective: bool = False) -> dict:
    """One dialogue. The supporter side is the arm under test; the seeker side is the simulator the arm
    names (proactive, or the help-seeking reactive one for reactive_baseline)."""
    turns: list[dict] = []
    artifacts: list[dict] = []
    openers = cfg("corpus.openers", default=["Hey - how are things today?"])
    rng = rng_for(session_id, "eval")
    opener = (f"Hey - it's been a little while. How have things been {gap_phrase(gap_days)}?"
              if session_index > 1 else openers[rng.randrange(len(openers))])
    turns.append({"turn_index": 0, "role": "supporter", "text": opener, "phase": "listening",
                  "ladder_rung": "L0", "meta": {"source": "opener_pool"}})
    sim_system = simulator_system(profile, session_context, reactive=reactive, corrective=corrective)
    stage_boundary = max(2, int(n_supporter_turns * 0.6))

    for i in range(n_supporter_turns):
        reply = fresh_turn(sim_view.chat, seeker_prompt(turns), turns, "seeker", system=sim_system)
        turns.append({"turn_index": len(turns), "role": "user", "text": reply,
                      "phase": "listening" if i < stage_boundary else "suggestion",
                      "disclosure_depth": None, "meta": {}})
        if i == n_supporter_turns - 1:
            break
        phase = "listening" if i + 1 < stage_boundary else "suggestion"
        result = pipeline.step(turns, session_id=session_id, phase=phase, session_index=session_index,
                               gap_days=gap_days)
        turns.append({"turn_index": len(turns), "role": "supporter", "text": result.text, "phase": phase,
                      "ladder_rung": result.permitted_rung,
                      "analysis": (result.analyzer.get("emotional_state", {}) or {}).get("label"),
                      "strategy": result.strategist.get("plan"),
                      "meta": {"path": result.path, "revisions": result.revisions,
                               "restricted_reason": result.restricted_reason}})
        artifacts.append({"turn_index": turns[-1]["turn_index"], **result.to_dict()})

    return {
        "session_id": session_id, "profile_id": profile["profile_id"], "session_index": session_index,
        "prev_session_id": None if session_index == 1 else f"{profile['profile_id']}-s{session_index - 1}",
        "gap_days_from_prev": None if session_index == 1 else gap_days,
        "profile_snapshot": profile, "stage_boundary_turn": stage_boundary, "turns": turns,
        "agent_artifacts": artifacts,
        "qc": {}, "annotations_complete": False,
    }


def run_arm(arm_name: str, limit: int | None = None, backend: str | None = None,
            run_dir: Path | None = None, calibration_path: str | None = None,
            adapter: str | None = None, supporter_base: str | None = None) -> dict:
    spec = resolve_arm(arm_name)
    if adapter:
        # an adapter trained in this run, so a pipeline run needs no hand edit of configs/arms.yaml
        spec["adapter"] = adapter
    profiles = load_profiles(spec, limit)
    run_dir = run_dir or new_run_dir(arm_name)
    out_path = run_dir / "dialogues" / f"{arm_name}.jsonl"
    done = existing_ids(out_path, "session_id")
    # resuming an advancing arm: later sessions continue from the profile the last written one used
    done_sessions = ({r["session_id"]: r for r in read_jsonl(out_path)} if spec.get("advance_profile") else {})

    calibration = None
    if spec.get("gate") == "conformal":
        path = Path(calibration_path) if calibration_path else run_dir / "conformal" / "calibration.json"
        if path.exists():
            calibration = Calibration.load(path)
            if spec.get("gate_alpha") and abs(calibration.alpha - float(spec["gate_alpha"])) > 1e-9:
                print(f"[warn] arm asks for alpha={spec['gate_alpha']} but the calibration was fitted at "
                      f"alpha={calibration.alpha}; re-run calibrate.py for this budget")
        else:
            print(f"[warn] no calibration at {path}: the gate falls back to a fixed 0.5 threshold, which is "
                  f"NOT the conformal claim. Run calibrate.py before reporting this arm.")

    role = "supporter" if spec["supporter"] != "base" else "generator"
    if spec.get("adapter"):
        # The arm names its own adapter, so a training-corpus comparison needs no config edits.
        load_config("models")["roles"][role]["adapter"] = spec["adapter"]
        if supporter_base:
            # the adapter's own base when it is not the resident model (scaling study, llm.TransformersBackend)
            load_config("models")["roles"][role]["adapter_base"] = supporter_base
    elif spec["supporter"] != "base":
        print(f"[warn] arm '{arm_name}' asks for supporter '{spec['supporter']}' but no adapter is "
              f"mapped in configs/arms.yaml: it will run on the UN-TUNED base model")
    llm = LLM(role, backend=backend, call_log=run_dir / "calls.jsonl")
    written = 0
    try:
        sim_view = llm.view("simulator")
        agent_views = {role: llm.view(role if role != "generator" else "supporter")
                       for role in ("analyzer", "strategist", "critic", "generator")}

        def roll_out(profile: dict) -> list[dict]:
            new: list[dict] = []
            pid = profile["profile_id"]
            memory = make_memory(spec.get("memory", "needstate"), pid, llm.view("generator"))
            pipeline_cls = (MonolithicListener if spec.get("architecture") == "monolithic"
                            else AgentPipeline)
            kwargs = dict(memory=memory, calibration=calibration,
                          gate=(spec.get("gate") == "conformal"))
            if pipeline_cls is AgentPipeline:
                pipeline = AgentPipeline(agent_views["generator"], views=agent_views, **kwargs)
            else:
                # memory_writer: the base Analyzer keeps the need-state memory for the fine-tuned reply (Option C)
                writer = (AgentPipeline(agent_views["generator"], views=agent_views, memory=memory, gate=False)
                          if spec.get("memory_writer") and isinstance(memory, NeedStateMemory) else None)
                pipeline = MonolithicListener(llm, memory_writer=writer, **kwargs)
            multi = spec.get("memory") != "none" or spec.get("multi_session")
            n_sessions = sample_session_count(pid) if multi else 1
            lo, hi = cfg("corpus.turns_per_session", default=[8, 12])
            current, prev = profile, None
            for index in range(1, n_sessions + 1):
                sid = f"{pid}-{arm_name}-s{index}"
                gap = 0.0 if index == 1 else sample_gap_days(sid)
                if index > 1:
                    if hasattr(memory, "decay"):
                        memory.decay(gap, session_id=sid)
                    if hasattr(memory, "advance_clock"):
                        memory.advance_clock(gap)
                if sid in done:
                    prev = done_sessions.get(sid, prev)
                    current = (prev or {}).get("profile_snapshot", current)
                    continue
                n_turns = rng_for(sid, "turns").randint(int(lo), int(hi))
                context = ("" if index == 1 else
                           f"This is a follow-up conversation, about {gap:.0f} days later.")
                if index > 1 and spec.get("advance_profile") and prev is not None:
                    # Move the seeker on as the corpus does (sessions.py): resolved / intensified / displaced.
                    # The applied transition is the ground truth memory_eval scores the memory against.
                    current = advance_profile(sim_view, current, prev, gap, pick_transition(pid, index))
                    context += (f" What has changed: "
                                f"{current['advancement'].get('notes') or 'some things shifted'}")
                session = run_session(pipeline, sim_view, current, sid, index, gap, n_turns, context,
                                      reactive=spec.get("simulator") == "reactive",
                                      corrective=spec.get("simulator") == "corrective")
                prev = session
                session["arm"] = arm_name
                session["arm_spec"] = spec
                session["in_distribution"] = bool(spec["eval_set_spec"].get("in_distribution", False))
                new.append(session)
                if hasattr(memory, "ingest_session"):
                    memory.ingest_session(session)
                if hasattr(memory, "update_from_session"):
                    memory.update_from_session(session)
                if isinstance(memory, NeedStateMemory):
                    memory.save(run_dir / "memory" / f"{pid}_{arm_name}.jsonl")
                # What every kind of memory would hand the next session: the same view for all memory arms, so
                # extra/memory_eval.py can score transition recall across them.
                session["memory_brief_after"] = memory.render_brief()
            return new

        for new in pmap(roll_out, profiles, llm):
            for session in new:
                append_jsonl(out_path, session)
                written += 1
    finally:
        llm.release()

    summary = {"arm": arm_name, "supporter": spec.get("supporter"), "adapter": spec.get("adapter"),
               "supporter_base": supporter_base,
               "sessions_written": written, "profiles": len(profiles),
               "dialogues_path": str(out_path), "gate": spec.get("gate"),
               "calibrated": calibration is not None, "memory": spec.get("memory"),
               "architecture": spec.get("architecture"),
               "in_distribution": bool(spec["eval_set_spec"].get("in_distribution", False))}
    write_json(run_dir / f"arm_summary_{arm_name}.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Roll out dialogues for one or more arms.")
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--all", action="store_true", help="every arm in configs/arms.yaml")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--adapter", default=None, help="supporter adapter dir; overrides configs/arms.yaml")
    ap.add_argument("--supporter-base", default=None,
                    help="base model of --adapter when it is not the resident 7B (scaling study); the 7B "
                         "keeps playing the seeker")
    args = ap.parse_args()

    if args.list:
        for name in sorted(load_config("arms")["arms"]):
            print(name)
        return
    names = list(load_config("arms")["arms"]) if args.all else (args.arm or [])
    if not names:
        raise SystemExit("pass --arm NAME (repeatable), --all, or --list")
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir("eval")
    for name in names:
        s = run_arm(name, args.limit, args.backend, run_dir, args.calibration, args.adapter,
                    args.supporter_base)
        print(f"{name}: {s['sessions_written']} sessions -> {s['dialogues_path']}")
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
