"""Conformal prediction sets for the hidden need, with temperature scaling.

The system today makes one guess at the terminal need and Success Rate scores it yes/no. Here the Analyzer
model (the pipeline's own, conversation-only view) rates K candidate needs from the dialogue so far, and
split conformal prediction turns the ratings into a SET with a finite-sample guarantee:

    P(true need in C(x)) >= 1 - alpha        (exchangeable calibration and test sessions)

Candidates  the session's true terminal need + K-1 distractors: other profiles' terminal needs, same problem
            type first (hard negatives), minus near-paraphrases of the truth (memory.similarity >= the
            memory module's own equivalence threshold). Order shuffled per session (position bias).
            This is the multiple-choice setting of conformal prediction for LLMs (Kumar et al., 2023).
Ratings     1-7 per candidate, judge-style 'number: score' (need_candidates.md), parsed by judge.parse_scores.
Calibration p_k = softmax(beta * r_k). beta is fitted by maximum likelihood on the calibration sessions
            (temperature scaling, Guo et al. 2017); the log-likelihood is concave in beta, so Newton is exact.
Sets        LAC (Sadinle et al. 2019): s = 1 - p_true, C = {k : 1 - p_k <= q_hat}
            APS (Romano et al. 2020): s = randomized cumulative mass down to the true label
            q_hat = the ceil((n+1)(1-alpha))-th smallest calibration score.
Positions   prefixes cut after 25 / 50 / 75 / 100 % of the seeker turns, each calibrated on its own: sets
            should shrink as the seeker says more.

Calibration sessions come from the calibration profiles (default arm calib_dec_ungated); test sessions from
test profiles in an arm with the same architecture and simulator (default cellC_dec_ungated), so they are
exchangeable as the guarantee requires.

  python extra/need_sets.py --run runs/v3_50 --profiles data/profiles/profiles.jsonl
"""
from __future__ import annotations

import argparse
import math
import statistics

import numpy as np

from _shared import fill, fmt, markdown_table, out_dir, pyplot, rate_items, read_extra_prompt, read_jsonl, \
    rng_for, run_path, write_json, write_jsonl
from common import cfg, render_transcript, user_turns
from llm import LLM, pmap
from memory import similarity

POSITIONS = (0.25, 0.5, 0.75, 1.0)
ALPHAS = (0.1, 0.2)


# --------------------------------------------------------------------------- items


def candidate_list(session: dict, pool: list[dict], k: int) -> tuple[list[str], int]:
    """The true need plus k-1 distractors, shuffled per session. Returns (candidates, index of truth)."""
    profile = session.get("profile_snapshot", {})
    truth = (profile.get("terminal_need") or "").strip()
    ptype = profile.get("problem_type")
    pid = session.get("profile_id")
    overlap = float(cfg("memory.equivalence_overlap_min", default=0.6))
    rng = rng_for(session["session_id"], "need_sets")
    others = [p for p in pool if p["profile_id"] != pid and p["text"]
              and similarity(p["text"], truth) < overlap]
    same = [p for p in others if p.get("problem_type") == ptype]
    rest = [p for p in others if p.get("problem_type") != ptype]
    rng.shuffle(same)
    rng.shuffle(rest)
    chosen: list[str] = []
    for p in same + rest:
        if len(chosen) == k - 1:
            break
        if all(similarity(p["text"], c) < overlap for c in chosen):
            chosen.append(p["text"])
    cands = [truth] + chosen
    rng.shuffle(cands)
    return cands, cands.index(truth)


def prefix_at(session: dict, frac: float) -> list[dict]:
    """Turns up to and including the seeker turn at this fraction of the session's seeker turns."""
    turns = session.get("turns", [])
    users = user_turns(turns)
    if not users:
        return []
    cut = users[max(0, math.ceil(frac * len(users)) - 1)]["turn_index"]
    return [t for t in turns if t["turn_index"] <= cut]


def score_prefix(llm, template: str, prefix: list[dict], candidates: list[str]) -> list[int] | None:
    lines = "\n".join(f"{i}. {' '.join(c.split())}" for i, c in enumerate(candidates, start=1))
    items = rate_items(llm, fill(template, candidates=lines, diag=render_transcript(prefix)), len(candidates))
    return [items[i] for i in range(1, len(candidates) + 1)] if items else None


def build_and_score(llm, sessions: list[dict], pool: list[dict], k: int, split: str) -> list[dict]:
    template = read_extra_prompt("need_candidates.md")

    def one(session: dict) -> list[dict]:
        cands, truth = candidate_list(session, pool, k)
        rows = []
        for pos in POSITIONS:
            ratings = score_prefix(llm, template, prefix_at(session, pos), cands)
            rows.append({"session_id": session["session_id"], "profile_id": session.get("profile_id"),
                         "split": split, "position": pos, "candidates": cands, "truth": truth,
                         "ratings": ratings, "parse_failed": ratings is None})
        return rows

    return [r for part in pmap(one, sessions, llm) for r in part]


# --------------------------------------------------------------------------- calibration and sets


def softmax(r: np.ndarray, beta: float) -> np.ndarray:
    z = beta * (r - r.max())
    e = np.exp(z)
    return e / e.sum()


def fit_temperature(ratings: list[list[int]], truths: list[int], iters: int = 50) -> float:
    """MLE of beta in p = softmax(beta r). d/dbeta log p_y = r_y - E_p[r]; d2 = -Var_p[r] < 0: concave."""
    beta = 1.0
    R = [np.asarray(r, dtype=float) for r in ratings]
    for _ in range(iters):
        g = h = 0.0
        for r, y in zip(R, truths):
            p = softmax(r, beta)
            mean = p @ r
            g += r[y] - mean
            h -= p @ (r - mean) ** 2
        if h > -1e-12:
            break
        step = g / h
        beta = min(50.0, max(1e-3, beta - step))
        if abs(step) < 1e-8:
            break
    return beta


def conformal_quantile(scores: list[float], alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else sorted(scores)[k - 1]


def aps_score(p: np.ndarray, y: int, u: float) -> float:
    """Randomized APS: mass of labels ranked above y, plus u times y's own mass (exact coverage)."""
    order = np.argsort(-p, kind="stable")
    rank = int(np.where(order == y)[0][0])
    return float(p[order[:rank]].sum() + u * p[y])


def aps_set(p: np.ndarray, q: float, u: float) -> list[int]:
    order = np.argsort(-p, kind="stable")
    out, mass = [], 0.0
    for idx in order:
        if mass + u * p[idx] <= q:
            out.append(int(idx))
        mass += p[idx]
    return out


def ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    total, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            total += len(idx) / n * abs(statistics.fmean(conf[i] for i in idx) -
                                        statistics.fmean(correct[i] for i in idx))
    return total


def evaluate_position(cal: list[dict], test: list[dict]) -> dict:
    cal = [r for r in cal if not r["parse_failed"]]
    test = [r for r in test if not r["parse_failed"]]
    if not cal or not test:
        return {"n_cal": len(cal), "n_test": len(test)}
    beta = fit_temperature([r["ratings"] for r in cal], [r["truth"] for r in cal])
    P_cal = [softmax(np.asarray(r["ratings"], float), beta) for r in cal]
    P_test = [softmax(np.asarray(r["ratings"], float), beta) for r in test]
    u_cal = [rng_for(r["session_id"] + str(r["position"]), "aps").random() for r in cal]
    u_test = [rng_for(r["session_id"] + str(r["position"]), "aps").random() for r in test]
    top1 = [int(np.argmax(p)) == r["truth"] for p, r in zip(P_test, test)]
    out = {"n_cal": len(cal), "n_test": len(test), "beta": beta, "k": len(test[0]["candidates"]),
           "top1_accuracy": statistics.fmean(top1),
           "ece_raw": ece([float(softmax(np.asarray(r["ratings"], float), 1.0).max()) for r in test], top1),
           "ece_scaled": ece([float(p.max()) for p in P_test], top1), "sets": {}}
    for alpha in ALPHAS:
        q_lac = conformal_quantile([1 - p[r["truth"]] for p, r in zip(P_cal, cal)], alpha)
        q_aps = conformal_quantile([aps_score(p, r["truth"], u) for p, r, u in zip(P_cal, cal, u_cal)], alpha)
        for name, sets in (
            ("LAC", [[k for k in range(len(p)) if 1 - p[k] <= q_lac] for p in P_test]),
            ("APS", [aps_set(p, q_aps, u) for p, u in zip(P_test, u_test)]),
        ):
            cover = [r["truth"] in s for s, r in zip(sets, test)]
            out["sets"][f"{name}@{alpha}"] = {
                "coverage": statistics.fmean(cover), "mean_size": statistics.fmean(len(s) for s in sets),
                "singleton_rate": statistics.fmean(len(s) == 1 for s in sets),
                "empty_rate": statistics.fmean(len(s) == 0 for s in sets),
                "guarantee": f">= {1 - alpha:.2f} (and <= {1 - alpha + 1 / (len(cal) + 1):.3f} with continuous scores)"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Conformal prediction sets for the terminal need.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--profiles", default="data/profiles/profiles.jsonl", help="distractor pool")
    ap.add_argument("--calib-arm", default="calib_dec_ungated")
    ap.add_argument("--test-arm", default="cellC_dec_ungated")
    ap.add_argument("--k", type=int, default=8, help="candidates per session")
    ap.add_argument("--limit", type=int, default=None, help="sessions per split")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    run_dir = run_path(args.run)
    pool = [{"profile_id": p["profile_id"], "text": (p.get("terminal_need") or "").strip(),
             "problem_type": p.get("problem_type")} for p in read_jsonl(run_path(args.profiles))]
    splits = {}
    for split, arm in (("calibration", args.calib_arm), ("test", args.test_arm)):
        sessions = read_jsonl(run_dir / "dialogues" / f"{arm}.jsonl")
        splits[split] = sessions[: args.limit] if args.limit else sessions

    dest = out_dir(run_dir, "need_sets")
    llm = LLM("analyzer", backend=args.backend, call_log=run_dir / "calls.jsonl")
    try:
        rows = [r for split, sessions in splits.items()
                for r in build_and_score(llm, sessions, pool, args.k, split)]
    finally:
        llm.release()
    write_jsonl(dest / "scored.jsonl", rows)

    results = {}
    for pos in POSITIONS:
        cal = [r for r in rows if r["split"] == "calibration" and r["position"] == pos]
        test = [r for r in rows if r["split"] == "test" and r["position"] == pos]
        results[str(pos)] = evaluate_position(cal, test)
    write_json(dest / "summary.json", {"calib_arm": args.calib_arm, "test_arm": args.test_arm, "k": args.k,
                                       "positions": results,
                                       "parse_failures": sum(r["parse_failed"] for r in rows)})

    table = []
    for pos, r in results.items():
        if "beta" not in r:
            continue
        row = {"position": f"{float(pos):.0%}", "n cal/test": f"{r['n_cal']}/{r['n_test']}",
               "top-1": fmt(r["top1_accuracy"]), "beta": fmt(r["beta"], 2),
               "ECE raw/scaled": f"{fmt(r['ece_raw'])}/{fmt(r['ece_scaled'])}"}
        for key, s in r["sets"].items():
            row[f"{key} cover/size"] = f"{fmt(s['coverage'])}/{fmt(s['mean_size'], 2)}"
        table.append(row)
    lines = [f"# Conformal need sets - {run_dir.name}", "",
             f"{args.k} candidate needs per session (truth + hard negatives). Calibrated on `{args.calib_arm}`, "
             f"tested on `{args.test_arm}`. Coverage should sit at or just above 1 - alpha; smaller sets are "
             "better. Chance top-1 accuracy is " + fmt(1 / args.k) + ".", "",
             markdown_table(table, list(table[0])) if table else "(no parsed items)", "",
             "The guarantee is marginal and holds only for test sessions exchangeable with the calibration "
             "sessions (same generator, simulator and profile distribution)."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt = pyplot()
    if plt is not None and table:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
        xs = [float(p) for p, r in results.items() if "beta" in r]
        for key in next(r for r in results.values() if "sets" in r)["sets"]:
            a1.plot(xs, [results[str(x)]["sets"][key]["coverage"] for x in xs], marker="o", label=key)
            a2.plot(xs, [results[str(x)]["sets"][key]["mean_size"] for x in xs], marker="o", label=key)
        a1.set_xlabel("share of seeker turns seen")
        a1.set_ylabel("coverage")
        a2.set_xlabel("share of seeker turns seen")
        a2.set_ylabel("mean set size")
        a2.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "coverage_size.png", dpi=150)
        plt.close(fig)
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
