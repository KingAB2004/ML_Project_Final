"""Choosing the next move by expected information gain minus reactance (needs need_sets.py first, for beta).

At a decision point the system holds a belief p over K candidate needs (need_sets: p = softmax(beta r)).
Each candidate move a - one per disclosure rung L0..L3, written by the pipeline's Generator, plus the move
the arm actually made - is scored by

    EIG(a)    = H(p) - E_{reply ~ sim(a)} [ H(p | prefix, a, reply) ]     (bits; expected entropy reduction)
    PRI(a)    = E_{reply ~ sim(a)} [ judge reactance of the reply ]       (the PRI instrument, metric_pri.md)
    a*(lam)   = argmax_a  EIG(a) - lam * PRI(a)

lam is the Lagrange multiplier of the constrained problem  max E[EIG]  s.t.  E[PRI] <= b.  The dual
g(lam) = mean_d max_a [EIG - lam (PRI - b)] is convex and piecewise linear in lam; its minimiser lam* gives
the policy that spends at most the reactance budget b, and g(lam*) upper-bounds the best mean EIG any
policy (even a randomised one) can reach within b. The default budget b is the reactance of the arm's own
moves, so the comparison is: more information at no extra reactance?

Also reported (never used to choose): the oracle gain E[log p'(true) - log p(true)], i.e. whether the
information gained was about the RIGHT need. Expectations are Monte Carlo over S simulated replies, from
a clean prefix each time (counterfactual.replay_user_reply). The re-scored belief is the model's judgement,
not an exact Bayesian posterior, so EIG here is an estimate.

Two passes, one model resident at a time: generator/simulator/analyzer views of the base model, then the
judge.

  python extra/eig_probe.py --run runs/v3_50 --profiles data/profiles/profiles.jsonl --arm cellC_dec_ungated
"""
from __future__ import annotations

import argparse
import math
import statistics

import numpy as np

from _shared import RUNG_ORDER, fmt, markdown_table, out_dir, pyplot, read_extra_prompt, read_jsonl, run_path, \
    write_json, write_jsonl
from common import fill, fresh_turn, read_json, read_prompt, render_transcript
from counterfactual import replay_user_reply
from judge import Judge
from llm import LLM, pmap
from metrics import sample_turn_indices
from need_sets import candidate_list, score_prefix, softmax

RUNG_PLANS = {
    "L0": "Reflect back what they just said, in your own words, adding nothing new.",
    "L1": "Invite them to say more about what they mentioned, with one open question.",
    "L2": "Offer a gentle, hedged guess about what might be underneath, and leave them room to correct it.",
    "L3": "Name directly the deeper need you think lies underneath what they are saying.",
}


def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def belief(llm, template: str, prefix: list[dict], cands: list[str], beta: float) -> np.ndarray | None:
    r = score_prefix(llm, template, prefix, cands)
    return None if r is None else softmax(np.asarray(r, dtype=float), beta)


def generate_pass(gen, sim, ana, sessions: list[dict], pool: list[dict], k: int, beta: float,
                  n_replies: int) -> list[dict]:
    gen_tpl = read_prompt("agent_generator.md")
    need_tpl = read_extra_prompt("need_candidates.md")

    def one(session: dict) -> dict | None:
        idx = next(iter(sample_turn_indices(session, ["middle"])), None)
        if idx is None:
            return None
        turns = session["turns"]
        prefix = [t for t in turns if t["turn_index"] < idx]
        cands, truth = candidate_list(session, pool, k)
        p0 = belief(ana, need_tpl, prefix, cands, beta)
        if p0 is None:
            return None
        moves = {"original": next(t["text"] for t in turns if t["turn_index"] == idx)}
        for rung, plan in RUNG_PLANS.items():
            prompt = fill(gen_tpl, plan=plan, permitted_rung=rung, memory_block="[memory] not used here.",
                          history=render_transcript(prefix, numbered=True), revision_note="")
            moves[rung] = fresh_turn(gen.chat, prompt, prefix, "supporter")
        out = {"session_id": session["session_id"], "turn_index": idx, "candidates": cands, "truth": truth,
               "p0": p0.tolist(), "moves": {}}
        profile = session.get("profile_snapshot", {})
        for name, text in moves.items():
            replies = []
            for s in range(n_replies):
                reply = replay_user_reply(sim, profile, prefix, text, f"{session['session_id']}#{idx}:{name}:{s}")
                after = prefix + [{"turn_index": idx, "role": "supporter", "text": text},
                                  {"turn_index": idx + 1, "role": "user", "text": reply}]
                p1 = belief(ana, need_tpl, after, cands, beta)
                replies.append({"reply": reply, "p1": None if p1 is None else p1.tolist()})
            out["moves"][name] = {"text": text, "replies": replies}
        return out

    return [d for d in pmap(one, sessions, gen) if d is not None]


def judge_pass(judge: Judge, decisions: list[dict], sessions: dict[str, dict]) -> None:
    jobs = [(d, name, r) for d in decisions for name, m in d["moves"].items() for r in m["replies"]]

    def one(job) -> None:
        d, name, r = job
        session = sessions[d["session_id"]]
        prefix = [t for t in session["turns"] if t["turn_index"] < d["turn_index"]]
        res = judge.score("pri", f"{d['session_id']}#{d['turn_index']}:{name}",
                          render_transcript(prefix + [{"role": "supporter", "turn_index": d["turn_index"],
                                                       "text": d["moves"][name]["text"]}]),
                          session.get("profile_snapshot", {}), target_text=r["reply"])
        r["reactance"] = None if res.parse_failed else res.normalized

    list(pmap(one, jobs, judge.llm))


def move_stats(d: dict) -> dict[str, dict]:
    p0 = np.asarray(d["p0"])
    h0 = entropy_bits(p0)
    out = {}
    for name, m in d["moves"].items():
        ok = [r for r in m["replies"] if r["p1"] is not None and r.get("reactance") is not None]
        if not ok:
            continue
        out[name] = {
            "eig": h0 - statistics.fmean(entropy_bits(np.asarray(r["p1"])) for r in ok),
            "pri": statistics.fmean(r["reactance"] for r in ok),
            "oracle_gain": statistics.fmean(math.log(max(r["p1"][d["truth"]], 1e-12)) -
                                            math.log(max(p0[d["truth"]], 1e-12)) for r in ok),
        }
    return out


def choose(stats: list[dict[str, dict]], lam: float, names=None) -> list[str]:
    names = names or RUNG_ORDER
    return [max((n for n in names if n in s), key=lambda n: s[n]["eig"] - lam * s[n]["pri"]) for s in stats]


def policy_value(stats: list[dict[str, dict]], picks: list[str]) -> dict:
    return {k: statistics.fmean(s[p][k] for s, p in zip(stats, picks)) for k in ("eig", "pri", "oracle_gain")}


def dual_lambda(stats: list[dict[str, dict]], budget: float, grid: np.ndarray) -> tuple[float, float]:
    """argmin_lam g(lam), g(lam) = mean_d max_a [EIG - lam (PRI - b)] over the candidate rungs."""
    def g(lam):
        return statistics.fmean(max(s[n]["eig"] - lam * (s[n]["pri"] - budget) for n in RUNG_ORDER if n in s)
                                for s in stats)
    vals = [g(lam) for lam in grid]
    i = int(np.argmin(vals))
    return float(grid[i]), float(vals[i])


def main() -> None:
    ap = argparse.ArgumentParser(description="Information gain vs reactance for the next supporter move.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--profiles", default="data/profiles/profiles.jsonl")
    ap.add_argument("--arm", default="cellC_dec_ungated")
    ap.add_argument("--limit", type=int, default=40, help="decision points (one per session)")
    ap.add_argument("--replies", type=int, default=3, help="simulated replies per move")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=float, default=None, help="reactance budget; default the arm's own")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    run_dir = run_path(args.run)
    summary_path = run_dir / "extra" / "need_sets" / "summary.json"
    beta = 1.0
    if summary_path.exists():
        betas = [r["beta"] for r in read_json(summary_path)["positions"].values() if "beta" in r]
        beta = statistics.median(betas) if betas else 1.0
    else:
        print("[warn] no need_sets summary: beta = 1.0 (run extra/need_sets.py first for a fitted beta)")
    pool = [{"profile_id": p["profile_id"], "text": (p.get("terminal_need") or "").strip(),
             "problem_type": p.get("problem_type")} for p in read_jsonl(run_path(args.profiles))]
    sessions = read_jsonl(run_dir / "dialogues" / f"{args.arm}.jsonl")[: args.limit]

    gen = LLM("generator", backend=args.backend, call_log=run_dir / "calls.jsonl")
    try:
        decisions = generate_pass(gen, gen.view("simulator"), gen.view("analyzer"), sessions, pool, args.k,
                                  beta, args.replies)
    finally:
        gen.release()
    judge = Judge(backend=args.backend, call_log=run_dir / "calls.jsonl")
    try:
        judge_pass(judge, decisions, {s["session_id"]: s for s in sessions})
    finally:
        judge.llm.release()

    dest = out_dir(run_dir, "eig_probe")
    write_jsonl(dest / "decisions.jsonl", decisions)
    stats = [s for s in (move_stats(d) for d in decisions) if all(n in s for n in (*RUNG_ORDER, "original"))]
    if not stats:
        raise SystemExit("no decision point with every move scored")
    budget = args.budget if args.budget is not None else statistics.fmean(s["original"]["pri"] for s in stats)
    grid = np.linspace(0, 50, 501)
    lam_star, g_star = dual_lambda(stats, budget, grid)
    policies = {f"always {r}": policy_value(stats, [r] * len(stats)) for r in RUNG_ORDER}
    policies["original (arm's own move)"] = policy_value(stats, ["original"] * len(stats))
    policies["max EIG (lam = 0)"] = policy_value(stats, choose(stats, 0.0))
    policies[f"EIG - lam* PRI (lam* = {lam_star:.2f})"] = policy_value(stats, choose(stats, lam_star))
    frontier = [{"lambda": float(l), **policy_value(stats, choose(stats, l))} for l in grid[::10]]
    picks = choose(stats, lam_star)
    write_json(dest / "summary.json", {"beta": beta, "budget": budget, "lambda_star": lam_star,
                                       "dual_value": g_star, "n_decisions": len(stats), "policies": policies,
                                       "frontier": frontier,
                                       "rung_mix_at_lambda_star": {r: picks.count(r) for r in RUNG_ORDER}})
    lines = [f"# Information gain vs reactance - {run_dir.name}, arm `{args.arm}`", "",
             f"{len(stats)} decision points, {args.replies} simulated replies per move, beta = {beta:.2f}. "
             f"Reactance budget b = {budget:.3f} (the arm's own moves). Dual: lam* = {lam_star:.2f}, "
             f"g(lam*) = {g_star:.3f} bits, an upper bound on mean EIG of any policy within the budget.", "",
             markdown_table([{"policy": k, "EIG (bits)": fmt(v["eig"]), "reactance": fmt(v["pri"]),
                              "oracle gain (nats)": fmt(v["oracle_gain"])} for k, v in policies.items()],
                            ["policy", "EIG (bits)", "reactance", "oracle gain (nats)"]), "",
             f"Rung mix chosen at lam*: {', '.join(f'{r} {picks.count(r)}' for r in RUNG_ORDER)}.", "",
             "Read: does the lam* policy gain more information than the arm's own move at no more reactance, "
             "and is that information about the true need (oracle gain > 0)?"]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt = pyplot()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([f["pri"] for f in frontier], [f["eig"] for f in frontier], marker=".", label="EIG - lam PRI")
        for name, v in policies.items():
            if name.startswith(("always", "original")):
                ax.scatter(v["pri"], v["eig"], s=25)
                ax.annotate(name.replace("always ", ""), (v["pri"], v["eig"]), fontsize=7)
        ax.set_xlabel("mean reactance (PRI)")
        ax.set_ylabel("mean expected information gain (bits)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "frontier.png", dpi=150)
        plt.close(fig)
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
