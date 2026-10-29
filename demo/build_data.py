"""Build the demo's replay data from recorded test runs (CPU only, no model is loaded).

The need-state graph is not copied from a summary: every supporter turn's stored Analyzer output is fed back
through the real `NeedStateMemory` (same `update_memory` code the pipeline runs), with the same decay between
sessions, so the graph the page shows at turn t is the graph the system held at turn t. The final replayed
graph is checked against the memory file the run saved.

    python demo/build_data.py                    # -> demo/static/data/replay.json
    python demo/build_data.py --people 40        # people per system (default 20)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import AgentPipeline  # noqa: E402
from memory import NeedStateMemory  # noqa: E402

R = ROOT / "results" / "v3_full1000"
S = R / "summaries"
OUT = Path(__file__).resolve().parent / "static" / "data" / "replay.json"

# The systems the page can replay: four agents with need-state memory, multi-session, on the 100 test people.
SYSTEMS = {
    "step12": {
        "label": "Full system",
        "detail": "Four agents, fine-tuned Qwen2.5-7B Generator, conformal gate, need-state memory",
        "dialogues": R / "sft_agents/dialogues/sft_dec_gated_mem.jsonl",
        "memory": R / "sft_agents/memory",
        "scores": R / "sft_agents/scores/sft_dec_gated_mem",
        "calibration": R / "sft_agents/conformal/calibration.json",
    },
    "needstate": {
        "label": "Untuned agents",
        "detail": "Four agents, untuned Qwen2.5-7B, no gate, need-state memory, seeker changes between sessions",
        "dialogues": R / "memory_v2/dialogues/mem_needstate.jsonl",
        "memory": R / "memory_v2/memory",
        "scores": R / "memory_v2/scores/mem_needstate",
        "calibration": None,
    },
}

# Arm-level table: (key, label, group, scores dir). Every number on the results tab is computed from these files.
ARMS = [
    ("base_instruct", "Base model, proactive seeker", "Baselines", "baselines_1000/scores/base_instruct"),
    ("reactive_baseline", "Base model, reactive seeker", "Baselines", "baselines_1000/scores/reactive_baseline"),
    ("cellA", "A: one call", "2 x 2 architecture", "baselines_1000/scores/cellA_mono_ungated"),
    ("cellB", "B: one call + gate", "2 x 2 architecture", "baselines_1000/scores/cellB_mono_gated"),
    ("cellC", "C: four agents", "2 x 2 architecture", "baselines_1000/scores/cellC_dec_ungated"),
    ("cellD", "D: four agents + gate", "2 x 2 architecture", "baselines_1000/scores/cellD_dec_gated"),
    ("step12", "Step 12: D + fine-tuned Generator + memory", "Full system", "sft_agents/scores/sft_dec_gated_mem"),
    ("sft_with", "Fine-tuned 7B, with thoughts", "Fine-tuning", "test_eval/qwen2.5_7B/scores/sft_with_thoughts"),
    ("sft_wo", "Fine-tuned 7B, without thoughts", "Fine-tuning", "test_eval/qwen2.5_7B/scores/sft_wo_thoughts"),
    ("sft_mem", "Fine-tuned 7B + need-state memory", "Fine-tuning", "sft7b_multi/scores/sft7b_multi_mem"),
    ("sft_nomem", "Fine-tuned 7B, no memory", "Fine-tuning", "sft7b_multi/scores/sft7b_multi_nomem"),
    ("mem_none", "No memory", "Memory", "memory_v2/scores/mem_none"),
    ("mem_summary", "Flat summary", "Memory", "memory_v2/scores/mem_summary"),
    ("mem_dense", "Dense retrieval", "Memory", "memory_v2/scores/mem_dense"),
    ("mem_needstate", "Need-state graph", "Memory", "memory_v2/scores/mem_needstate"),
    ("steer_ctrl", "Steering control", "Alternatives", "steering/scores/steer_mono_L12_c0"),
    ("steer", "Activation steering", "Alternatives", "steering/scores/steer_mono_L12_c0.2"),
    ("dpo_ctrl", "SFT control", "Alternatives", "preference/scores/pref_mono_sft"),
    ("dpo", "DPO", "Alternatives", "preference/scores/pref_mono_dpo"),
    ("extes_ours", "ExTES-trained, our people", "Corpus comparison", "corpus_extes/scores/corpus_extes"),
    ("extes_extes", "ExTES-trained, ExTES people", "Corpus comparison", "corpus_extes/scores/corpus_extes_extes_profiles"),
    ("ours_extes", "Our-corpus model, ExTES people", "Corpus comparison", "corpus_extes/scores/corpus_ours_extes_profiles"),
]

COST_PER_TURN = {"cellA": 1.00, "cellB": 3.33, "cellC": 4.00, "cellD": 5.36, "step12": 4.32,
                 "steer_ctrl": 1.00, "steer": 1.00, "dpo_ctrl": 1.00, "dpo": 1.00}   # S/24_cost_per_turn


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()] if path.exists() else []


def r3(x):
    return None if x is None else round(float(x), 3)


# -- statistics ---------------------------------------------------------------
def bootstrap_mean(by_person: dict[str, list[float]], reps: int = 1000, seed: int = 7) -> dict | None:
    people = [p for p, v in by_person.items() if v]
    if not people:
        return None
    flat = [x for p in people for x in by_person[p]]
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        vals = [x for p in (rng.choice(people) for _ in people) for x in by_person[p]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return {"mean": r3(sum(flat) / len(flat)), "lo": r3(means[int(0.025 * reps)]),
            "hi": r3(means[int(0.975 * reps) - 1]), "n": len(flat), "people": len(people)}


def person_of(target: str) -> str:
    return target.split("-")[0]


def scale_by_person(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(path):
        if not row.get("parse_failed") and row.get("normalized") is not None:
            out[person_of(row["target"])].append(float(row["normalized"]))
    return out


def arm_table() -> list[dict]:
    rows = []
    for key, label, group, rel in ARMS:
        d = R / rel
        if not d.exists():
            continue
        row = {"key": key, "label": label, "group": group, "source": f"results/v3_full1000/{rel}"}
        for scale in ("success", "ip", "pri", "aels"):
            row[scale] = bootstrap_mean(scale_by_person(d / f"{scale}.jsonl"))
        cf = d / "pri_summary.json"
        if cf.exists():
            s = json.loads(cf.read_text())
            row["pri_cf"] = {"mean": r3(s["pri"]), "lo": r3(s["ci_low"]), "hi": r3(s["ci_high"])}
        row["sessions"] = len(read_jsonl(d / "success.jsonl"))
        row["calls"] = COST_PER_TURN.get(key)
        rows.append(row)
    return rows


def contrasts() -> list[dict]:
    path = S / "23_multiple_comparisons/multiple_comparisons.txt"
    out = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 8 or parts[0] == "SOP part":
            continue
        m = re.match(r"([+-]?[\d.]+) \[([+-]?[\d.]+), ([+-]?[\d.]+)\]", parts[4])
        if not m:
            continue
        out.append({"part": parts[0], "contrast": parts[1], "scale": parts[2], "paired": parts[3] == "yes",
                    "diff": float(m[1]), "lo": float(m[2]), "hi": float(m[3]), "p": float(parts[5]),
                    "holm": float(parts[6]), "survives": parts[7] == "yes"})
    return out


# -- replay -------------------------------------------------------------------
# A snapshot row is [id, parent, status, confidence, dormant, evidence, against, sessions seen]; a node's text
# and depth never change, so they live once per person in `nodes` (page: NODE_FIELDS in app.js).
def node_view(n) -> list:
    return [n.node_id, n.parent_id, n.status, r3(n.confidence), int(n.dormant), len(n.supporting_spans),
            len(n.contradicting_spans), n.sessions_seen]


def graph_state(mem: NeedStateMemory, table: dict | None = None) -> dict:
    if table is not None:
        for n in mem.nodes.values():
            table[n.node_id] = [n.text, n.depth]
    best = mem.best_terminal()
    return {"nodes": [node_view(n) for n in mem.nodes.values()],
            "best": best.node_id if best else None,
            "chain": [n.node_id for n in mem.chain_path(best.node_id)] if best else []}


def new_transitions(mem: NeedStateMemory, start: int) -> list[dict]:
    return [{"node": t["node_id"], "from": t["from"], "to": t["to"], "reason": t["reason"],
             "quote": (t.get("span") or {}).get("quote")} for t in mem.transitions[start:]]


def compact_step(a: dict) -> dict:
    an, st, cr = a.get("analyzer") or {}, a.get("strategist") or {}, a.get("critic") or {}
    emo = an.get("emotional_state") if isinstance(an.get("emotional_state"), dict) else {}
    res = an.get("resistance_estimate") if isinstance(an.get("resistance_estimate"), dict) else {}
    g = an.get("grounding_report") or {}
    needs = [{"text": n.get("text"), "depth": n.get("depth"), "conf": r3(n.get("confidence") or 0),
              "hint": n.get("status_hint"),
              "quote": next((s.get("quote") for s in n.get("evidence_spans") or [] if isinstance(s, dict)), None)}
             for n in an.get("implicit_needs") or [] if isinstance(n, dict)]
    return {
        "analyzer": {"emotion": emo.get("label"), "gloss": emo.get("gloss"), "conf": r3(emo.get("confidence")),
                     "quote": next((s.get("quote") for s in emo.get("evidence_spans") or [] if isinstance(s, dict)), None),
                     "needs": needs, "resistance": res.get("level"), "resistance_conf": r3(res.get("confidence")),
                     "questions": (an.get("open_questions") or [])[:3],
                     "grounding": {"valid": g.get("valid_spans", 0), "invalid": g.get("invalid_spans", 0),
                                   "violations": len(g.get("violations") or [])}},
        "strategist": {"plan": st.get("plan"), "why": (st.get("justification_chain") or [])[:4],
                       "target": st.get("target_node_id"), "requested": st.get("requested_rung"),
                       "allowed": st.get("allowed_rungs"), "restricted": st.get("restricted_reason")},
        "critic": ({"ip_pred": r3(cr.get("ip_pred")), "score": r3(cr.get("nonconformity")),
                    "decision": cr.get("decision"), "items": cr.get("item_scores"),
                    "violations": [v.get("category") for v in cr.get("grounding_violations") or []],
                    "feedback": cr.get("feedback") or "", "rung": cr.get("draft_rung")} if cr else None),
        "path": a.get("path"), "revisions": a.get("revisions", 0), "permitted": a.get("permitted_rung"),
        "writes": [{"claim": w.get("claim"), "action": w.get("action"), "node": w.get("node_id")}
                   for w in a.get("memory_writes") or []],
    }


def replay_person(pid: str, sessions: list[dict], scores: dict) -> tuple[dict, NeedStateMemory]:
    mem, table = NeedStateMemory(pid), {}
    writer = AgentPipeline.__new__(AgentPipeline)        # only update_memory is used: it touches no model
    writer.memory = mem
    first = sessions[0]["profile_snapshot"]
    out_sessions = []
    for s in sorted(sessions, key=lambda x: x["session_index"]):
        sid, gap = s["session_id"], float(s.get("gap_days_from_prev") or 0)
        mark = len(mem.transitions)
        if s["session_index"] > 1:
            mem.decay(gap, session_id=sid)
        start = {"graph": graph_state(mem, table), "events": new_transitions(mem, mark)}
        arts = {a["turn_index"]: a for a in s.get("agent_artifacts") or []}
        turns = []
        for t in s["turns"]:
            row = {"i": t["turn_index"], "role": "seeker" if t["role"] == "user" else "supporter", "text": t["text"],
                   "phase": t.get("phase")}
            if t["role"] == "user":
                pri = scores["pri"].get(f"{sid}#{t['turn_index']}")
                if pri is not None:
                    row["pri"] = pri
            else:
                row["rung"] = t.get("ladder_rung")
                ip = scores["ip"].get(f"{sid}#{t['turn_index']}")
                if ip is not None:
                    row["ip"] = ip
                a = arts.get(t["turn_index"])
                if a:
                    mark = len(mem.transitions)
                    history = [x for x in s["turns"] if x["turn_index"] < t["turn_index"]]
                    writer.update_memory(a.get("analyzer") or {}, sid, history)
                    row["step"] = compact_step(a)
                    row["graph"] = graph_state(mem, table)
                    row["events"] = new_transitions(mem, mark)
                    row["brief"] = mem.render_brief()
                elif t.get("meta", {}).get("source"):
                    row["source"] = t["meta"]["source"]
            turns.append(row)
        snap = s["profile_snapshot"]
        out_sessions.append({
            "index": s["session_index"], "id": sid, "gap": gap, "start": start, "turns": turns,
            "success": scores["success"].get(sid), "boundary": s.get("stage_boundary_turn"),
            "change": (snap.get("advancement") or {}).get("notes") or (snap.get("advancement") or {}).get("kind"),
            "terminal_need": snap.get("terminal_need"),
        })
    profile = {
        "emotion": first.get("emotion"), "feeling": first.get("feeling"), "terminal_need": first.get("terminal_need"),
        "chain": [c.get("text") for c in first.get("need_chain") or []], "problem": first.get("problem_type"),
        "resistance": first.get("resistance_level"), "persona": first.get("persona_surface"),
        "memories": [m.get("text") for m in first.get("memory") or []][:4],
    }
    return {"id": pid, "profile": profile, "nodes": table, "sessions": out_sessions}, mem


def score_index(d: Path) -> dict:
    idx = {"ip": {}, "pri": {}, "success": {}}
    for scale in idx:
        for row in read_jsonl(d / f"{scale}.jsonl"):
            if row.get("normalized") is not None and not row.get("parse_failed"):
                idx[scale][row["target"]] = r3(row["normalized"])
    return idx


def check_against_saved(mem: NeedStateMemory, saved: Path) -> tuple[int, int]:
    """(matching nodes, saved nodes): text, status and parent must agree."""
    rows = read_jsonl(saved)
    ours = {n.node_id: n for n in mem.nodes.values()}
    ok = sum(1 for r in rows if (n := ours.get(r["node_id"])) and n.text == r["text"]
             and n.status == r["status"] and n.parent_id == r["parent_id"])
    return ok, len(rows)


def interest(person: dict) -> float:
    """Pick people whose graphs do something: confirmations, denials, revisions, more sessions."""
    ev = [e for s in person["sessions"] for t in s["turns"] for e in t.get("events", [])]
    tos = [e["to"] for e in ev]
    paths = [t["step"]["path"] for s in person["sessions"] for t in s["turns"] if "step" in t]
    return (3 * tos.count("confirmed") + 5 * tos.count("disconfirmed") + 4 * tos.count("resolved")
            + 2 * sum(p != "released" for p in paths) + len(person["sessions"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=20, help="people per system")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    systems, people = {}, []
    for key, spec in SYSTEMS.items():
        by_pid = defaultdict(list)
        for s in read_jsonl(spec["dialogues"]):
            by_pid[s["profile_id"]].append(s)
        scores = score_index(spec["scores"])
        replayed, match, total = [], 0, 0
        for pid, sessions in sorted(by_pid.items()):
            person, mem = replay_person(pid, sessions, scores)
            ok, n = check_against_saved(mem, spec["memory"] / f"{pid}_{sessions[0]['arm']}.jsonl")
            match, total = match + ok, total + n
            person["system"] = key
            replayed.append(person)
        replayed.sort(key=interest, reverse=True)
        keep = replayed[: args.people]
        if "p000394" not in {p["id"] for p in keep}:          # the worked example in S/09 is always included
            keep += [p for p in replayed if p["id"] == "p000394"]
        people += keep
        cal = json.loads(spec["calibration"].read_text()) if spec["calibration"] else None
        systems[key] = {"label": spec["label"], "detail": spec["detail"], "people": len(by_pid),
                        "replay_match": {"nodes": match, "saved": total},
                        "calibration": ({"alpha": cal["alpha"], "tau": cal["tau"], "delta": cal["delta"],
                                         "lambda": cal["lambda_hat"], "n": cal["n_calibration"],
                                         "curve": [{"lambda": r["lambda"], "release": r3(r["release_rate"]),
                                                    "risk": r3(r.get("risk")), "ucb": r3(r.get("risk_ucb"))}
                                                   for r in cal.get("risk_curve", [])]} if cal else None)}
        print(f"{key}: {len(by_pid)} people replayed, kept {len(keep)}; "
              f"final graph matches the saved memory on {match}/{total} nodes")

    recovery = json.loads((S / "14_gate_recovery_vs_cost/recovery_vs_cost.json").read_text())
    data = {
        "built_from": "results/v3_full1000 (judge Mistral-Nemo-12B; 100 held-out test people)",
        "systems": systems,
        "people": people,
        "arms": arm_table(),
        "contrasts": contrasts(),
        "recovery": {k: [{"lambda": r["lambda"], "cost": r3(r["cost"]), "recovery": r3(r["recovery"])} for r in v]
                     for k, v in recovery.items()},
        "corpus": json.loads((S / "01_dataset_quality__ours_vs_ExTES_vs_ESConv/dataset_stats.txt").read_text()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(people)} people)")


if __name__ == "__main__":
    main()
