"""Enhancement 3 - counterfactual attribution for PRI (PLAN Sec. 11.3).

Each supporter turn is treated as an intervention:
  factual  : keep the turn, re-generate the seeker's reply N times
  control  : replace it with a NEUTRAL REFLECTIVE control turn, re-generate N times with the SAME seeds
  PRI      : mean(r_factual) - mean(r_control), with a paired bootstrap CI over the seed-matched pairs

Positive PRI means the actual move provoked resistance beyond what a neutral reflection would have. The
simulator is reset to the prefix for every rollout: a replay never continues a contaminated conversation.

  python src/counterfactual.py --dialogues runs/<id>/dialogues/arm.jsonl --backend echo
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Sequence

from common import (cfg, fill, read_prompt, read_jsonl, render_transcript, rng_for, word_count,
                    write_json, write_jsonl)
from dialogue import simulator_system
from judge import Judge
from llm import LLM, pmap


def make_control_turn(llm: LLM, turns: Sequence[dict], supporter_index: int) -> str:
    """Length-matched, inference-free reflection, so the contrast is not confounded by verbosity."""
    prior_user = [t for t in turns if t["role"] == "user" and t["turn_index"] < supporter_index]
    original = next((t for t in turns if t["turn_index"] == supporter_index), {"text": ""})
    last_user = prior_user[-1]["text"] if prior_user else ""
    prompt = fill(read_prompt("control_reflective.md"), user_turn=last_user,
                  target_words=str(max(8, word_count(original.get("text", "")))))
    return llm.chat(prompt, temperature=0.2).strip()


def replay_user_reply(sim: LLM, profile: dict, prefix: Sequence[dict], supporter_text: str,
                      seed_tag: str) -> str:
    """One rollout of the seeker's next reply, from a clean prefix plus the (factual or control) turn."""
    turns = list(prefix) + [{"turn_index": 10_000, "role": "supporter", "text": supporter_text}]
    system = simulator_system(profile)
    prompt = (f"{render_transcript(turns)}\n\nWrite only your next reply as the person described in your "
              f"instructions. [rollout {seed_tag}]")
    return sim.chat(prompt, system=system, use_cache=True,
                    temperature=float(sim.spec.get("temperature", 0.85))).strip()


def paired_bootstrap(factual: Sequence[float], control: Sequence[float], resamples: int | None = None,
                     seed_tag: str = "pri") -> dict:
    """CI over seed-matched pairs. Pairing is the whole point: the same seed, two interventions."""
    n = min(len(factual), len(control))
    if n == 0:
        return {"pri": None, "ci_low": None, "ci_high": None, "n_pairs": 0}
    diffs = [float(factual[i]) - float(control[i]) for i in range(n)]
    point = statistics.fmean(diffs)
    reps = int(resamples or cfg("pri.bootstrap_resamples", default=10000))
    rng = rng_for(seed_tag, "bootstrap")
    means = []
    for _ in range(reps):
        means.append(statistics.fmean(diffs[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int(0.025 * reps)]
    hi = means[min(reps - 1, int(0.975 * reps))]
    return {"pri": point, "ci_low": lo, "ci_high": hi, "n_pairs": n}


def aggregate_pri(rows: Sequence[dict], resamples: int | None = None) -> dict:
    by_pair: dict[tuple, dict] = {}
    for r in rows:
        if r.get("parse_failed"):
            continue
        key = (r["session_id"], r["turn_index"], r["rollout_index"])
        by_pair.setdefault(key, {})[r["condition"]] = float(r["reactance_score"])
    factual = [v["factual"] for v in by_pair.values() if "factual" in v and "control" in v]
    control = [v["control"] for v in by_pair.values() if "factual" in v and "control" in v]
    out = paired_bootstrap(factual, control, resamples)
    out.update({
        "mean_factual": statistics.fmean(factual) if factual else None,
        "mean_control": statistics.fmean(control) if control else None,
        "n_turns_scored": len({(k[0], k[1]) for k in by_pair}),
        "judge_model": rows[0]["judge_model"] if rows else "",
    })
    return out


def run(dialogues_path: Path, out_dir: Path, backend: str | None = None,
        turns_per_dialogue: int | None = None) -> dict:
    from metrics import sample_turn_indices

    sessions = read_jsonl(dialogues_path)
    k = int(turns_per_dialogue or cfg("pri.turns_sampled_per_dialogue", default=4))
    sim = LLM("simulator", backend=backend)
    rows: list[dict] = []
    try:
        pending: list[tuple[dict, int]] = []
        for session in sessions:
            for idx in sample_turn_indices(session)[:k]:
                pending.append((session, idx))
        # The simulator is resident here; the judge cannot be (PLAN R2), so replies are generated first.
        n = int(cfg("pri.n_rollouts", default=5))

        def replay(item: tuple[dict, int]) -> list[dict]:
            session, idx = item
            out: list[dict] = []
            turns = session["turns"]
            prefix = [t for t in turns if t["turn_index"] < idx]
            factual_text = next((t["text"] for t in turns if t["turn_index"] == idx), "")
            control_text = make_control_turn(sim, turns, idx)
            for r in range(n):
                for condition, text in (("factual", factual_text), ("control", control_text)):
                    tag = f"{session['session_id']}#{idx}:{r}"
                    out.append({
                        "session_id": session["session_id"], "turn_index": idx, "condition": condition,
                        "rollout_index": r, "supporter_text": text,
                        "user_reply": replay_user_reply(sim, session.get("profile_snapshot", {}),
                                                        prefix, text, tag),
                        "prefix_len": len(prefix),
                    })
            return out

        replay_cache = [rec for part in pmap(replay, pending, sim) for rec in part]
    finally:
        sim.release()

    judge = Judge(backend=backend)
    by_id = {s["session_id"]: s for s in sessions}
    try:
        def judge_one(rec: dict) -> dict:
            session = by_id[rec["session_id"]]
            turns = session["turns"]
            prefix = [t for t in turns if t["turn_index"] < rec["turn_index"]]
            scored = judge.score(
                "pri", f"{rec['session_id']}#{rec['turn_index']}:{rec['rollout_index']}:{rec['condition']}",
                render_transcript(prefix + [{"role": "supporter", "text": rec["supporter_text"],
                                             "turn_index": rec["turn_index"]}]),
                session.get("profile_snapshot", {}), target_text=rec["user_reply"])
            rec.update({"reactance_score": scored.normalized, "parse_failed": scored.parse_failed,
                        "judge_model": scored.judge_model,
                        "context_redactions": scored.context_redactions})
            return rec

        rows.extend(pmap(judge_one, replay_cache, judge.llm))
    finally:
        judge.llm.release()

    write_jsonl(out_dir / "pri_rollouts.jsonl", rows)
    summary = aggregate_pri(rows)
    write_json(out_dir / "pri_summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Counterfactual PRI: factual vs neutral-control replay.")
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--turns", type=int, default=None, help="supporter turns scored per dialogue")
    args = ap.parse_args()
    dialogues = Path(args.dialogues)
    out_dir = Path(args.out) if args.out else dialogues.parent.parent / "scores" / dialogues.stem
    summary = run(dialogues, out_dir, args.backend, args.turns)
    print(f"PRI = {summary['pri']} (95% CI {summary['ci_low']} .. {summary['ci_high']}) over "
          f"{summary['n_pairs']} seed-matched pairs on {summary['n_turns_scored']} turns")


if __name__ == "__main__":
    main()
