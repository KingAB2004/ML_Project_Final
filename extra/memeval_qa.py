"""ES-MemEval: can each memory answer questions about a user's past sessions? (GPU via Ollama; lab server)

ES-MemEval (Chen et al., WWW 2026; the EvoEmo set) is a memory benchmark, not a set of seekers to simulate:
18 users, 401 dated past support sessions, and 1,427 questions with gold answers in five capabilities -
information extraction, user modelling, temporal reasoning, conflict detection and abstention. The SOP (III-D,
Expected Outcome 4) commits to testing the need-state memory on it against the flat-summary, dense-retrieval
and event-level memories, i.e. on data this project did not generate.

Per user and memory type:
  1. ingest   every past session in time order, with the real gaps between their dates (decay / clock). The
              need-state memory is written the way it is written live: the Analyzer reads each seeker turn and
              agents.AgentPipeline.update_memory applies only grounded claims. Summary and event memories use
              their own LLM updates; dense retrieval embeds the seeker turns; none keeps nothing.
  2. answer   each question from the memory's brief only (same token budget for every memory; dense and event
              retrieve with the question as the query), by the generator, which may say it does not know.
  3. grade    the judge compares the answer with the gold answer (>= 5/7 = correct). One model at a time:
              the generator is released before the judge loads.
Accuracy per memory type and capability, 95 % CI by bootstrap over users.

  python extra/memeval_qa.py --backend ollama                       # all users, all five memories
  python extra/memeval_qa.py --memories none needstate --users 2    # smaller
Writes runs/memeval/extra/memeval_qa/: answers.jsonl, summary.json, report.md, accuracy.png.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import date

from _shared import ROOT, cluster_bootstrap, fill, fmt, markdown_table, out_dir, pyplot, rate_items, \
    read_extra_prompt, read_jsonl, run_path, write_json, write_jsonl
from agents import AgentPipeline
from common import cfg
from evaluate import make_memory
from judge import assert_judge_separate
from llm import LLM, pmap

MEMORIES = ("none", "summary", "dense", "event", "needstate")
CAPABILITIES = ("information extraction", "user modeling", "temporal reasoning", "conflict detection", "abstention")
RAW = ROOT / "data" / "raw" / "es_memeval.json"


def to_sessions(user: dict) -> list[dict]:
    """ES-MemEval dialog_history -> this project's session records, oldest first, with real gaps."""
    hist = sorted(user["dialog_history"], key=lambda s: s.get("timestamp", ""))
    out, prev = [], None
    for k, s in enumerate(hist, start=1):
        day = date.fromisoformat(s["timestamp"][:10]) if s.get("timestamp") else None
        gap = (day - prev).days if day and prev else None
        prev = day or prev
        out.append({"session_id": f"{user['id']}-{s['id']}", "profile_id": f"memeval{user['id']}",
                    "session_index": k, "gap_days_from_prev": gap, "timestamp": s.get("timestamp"),
                    "turns": [{"turn_index": i, "role": "user" if t.get("role") == "seeker" else "supporter",
                               "text": t.get("content", "")} for i, t in enumerate(s.get("dialogue", []))]})
    return out


def questions(user: dict) -> list[dict]:
    return [{"user": user["id"], "group": g.get("id"), "idx": q.get("idx"), "capability": q["capability"],
             "question": q["question"], "answer": q["answer"]}
            for g in user["questions"] for q in g["questions"]]


def build_memory(kind: str, llm, user: dict, pipeline_views: dict):
    mem = make_memory(kind, f"memeval{user['id']}", llm.view("generator"))
    pipe = AgentPipeline(pipeline_views["generator"], views=pipeline_views, memory=mem, calibration=None,
                         gate=False) if kind == "needstate" else None
    for s in to_sessions(user):
        if s["gap_days_from_prev"]:
            if hasattr(mem, "decay"):
                mem.decay(float(s["gap_days_from_prev"]), session_id=s["session_id"])
            if hasattr(mem, "advance_clock"):
                mem.advance_clock(float(s["gap_days_from_prev"]))
        if pipe is not None:
            for i, t in enumerate(s["turns"]):
                if t["role"] == "user":                       # the Analyzer reads every seeker turn
                    brief = mem.brief()
                    analyzer = pipe.analyze(s["turns"][: i + 1], s["session_id"],
                                            mem.render_brief(brief.get("open_questions", [])))
                    pipe.update_memory(analyzer, s["session_id"], s["turns"][: i + 1])
        if hasattr(mem, "ingest_session"):
            mem.ingest_session(s)
        if hasattr(mem, "update_from_session"):
            mem.update_from_session(s)
    return mem


def brief_for(mem, question: str, budget: int) -> str:
    try:
        return mem.render_brief(token_budget=budget, query=question)     # dense, event: retrieve by question
    except TypeError:
        return mem.render_brief(token_budget=budget)


def main() -> None:
    ap = argparse.ArgumentParser(description="ES-MemEval question answering per memory type.")
    ap.add_argument("--memories", nargs="*", default=list(MEMORIES))
    ap.add_argument("--users", type=int, default=None, help="first N users (default all 18)")
    ap.add_argument("--budget", type=int, default=None, help="memory brief tokens (default agents config)")
    ap.add_argument("--run", default="runs/memeval")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    users = json.loads(RAW.read_text(encoding="utf-8"))[: args.users]
    budget = int(args.budget or cfg("agents.memory_brief_token_budget", default=320))
    run_dir = run_path(args.run)
    dest = out_dir(run_dir, "memeval_qa")
    answer_tpl, grade_tpl = read_extra_prompt("memeval_answer.md"), read_extra_prompt("memeval_grade.md")

    rows = []
    llm = LLM("generator", backend=args.backend, call_log=run_dir / "calls.jsonl")
    try:
        views = {r: llm.view(r if r != "generator" else "supporter")
                 for r in ("analyzer", "strategist", "critic", "generator")}
        for kind in args.memories:
            for user in users:
                mem = build_memory(kind, llm, user, views)
                qs = questions(user)
                name = (user.get("basic_info") or {}).get("name", "this person")

                def answer(q, mem=mem, name=name):
                    prompt = fill(answer_tpl, name=name, memory=brief_for(mem, q["question"], budget),
                                  question=q["question"])
                    return {**q, "memory": kind, "response": llm.chat(prompt, temperature=0.0)}

                rows += list(pmap(answer, qs, llm))
                print(f"{kind}: user {user['id']} answered {len(qs)}", flush=True)
    finally:
        llm.release()

    assert_judge_separate("judge")
    judge = LLM("judge", backend=args.backend, call_log=run_dir / "calls.jsonl")
    try:
        def grade(r):
            items = rate_items(judge, fill(grade_tpl, question=r["question"], gold=r["answer"],
                                           response=r["response"]), 1)
            return {**r, "rating": items[1] if items else None,
                    "correct": None if items is None else int(items[1] >= 5)}
        rows = list(pmap(grade, rows, judge))
    finally:
        judge.release()
    write_jsonl(dest / "answers.jsonl", rows)

    def acc(rs: list[dict], tag: str) -> dict | None:
        rs = [r for r in rs if r["correct"] is not None]
        if not rs:
            return None
        by: dict[str, list[int]] = {}
        for r in rs:
            by.setdefault(r["user"], []).append(r["correct"])
        return {"n": len(rs), "accuracy": statistics.fmean(r["correct"] for r in rs),
                "ci": cluster_bootstrap(list(by.values()), lambda g: statistics.fmean(v for vs in g for v in vs),
                                        args.reps, tag)}

    summary = {kind: {"all": acc([r for r in rows if r["memory"] == kind], f"mq:{kind}"),
                      **{c: acc([r for r in rows if r["memory"] == kind and r["capability"] == c], f"mq:{kind}:{c}")
                         for c in CAPABILITIES}}
               for kind in args.memories}
    write_json(dest / "summary.json", {"users": len(users), "budget_tokens": budget, "memories": summary})
    cell = lambda d: f"{fmt(d['accuracy'])} [{fmt(d['ci'][0])}, {fmt(d['ci'][1])}]" if d else "-"
    table = [{"memory": k, "all": cell(v["all"]), **{c: cell(v[c]) for c in CAPABILITIES}} for k, v in summary.items()]
    lines = ["# ES-MemEval question answering per memory type", "",
             f"{len(users)} users, {sum(len(questions(u)) for u in users)} questions per memory, memory brief "
             f"budget {budget} tokens. Correct = judge >= 5/7 against the gold answer. 95 % CI: bootstrap over "
             f"users.", "", markdown_table(table, list(table[0])), ""]
    plt = pyplot()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        w = 0.8 / max(1, len(summary))
        for k, (kind, v) in enumerate(summary.items()):
            ax.bar([i + k * w for i in range(len(CAPABILITIES))], [(v[c] or {}).get("accuracy", 0) for c in CAPABILITIES],
                   width=w, label=kind)
        ax.set_xticks([i + 0.4 - w / 2 for i in range(len(CAPABILITIES))], CAPABILITIES, fontsize=7)
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "accuracy.png", dpi=150)
        plt.close(fig)
        lines.append("![accuracy.png](accuracy.png)")
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
