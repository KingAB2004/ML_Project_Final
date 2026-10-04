"""Fit the conformal critic gate on a held-out split (PLAN Sec. 10.4).

Two passes, because the critic and the judge are different models and only one may be resident:
  pass 1 - critic scores every calibration turn -> nonconformity s
  pass 2 - judge scores the same turns on IP    -> violation labels (IP_norm > tau)
then the risk curve is built and lambda_hat selected as the largest lambda whose upper confidence bound on
risk stays within the budget alpha.

Calibration turns must come from held-out profiles of the SAME generator and profile distribution the gate
will run on; `--distribution-id` records which, and every arm that runs elsewhere reports an empirical
violation rate instead of a guarantee.

  python src/calibrate.py --dialogues runs/<id>/dialogues/cellC_dec_ungated.jsonl --alpha 0.1 --backend echo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agents import Critic
from common import cfg, read_jsonl, write_json
from conformal import calibrate, sweep_alphas
from judge import Judge
from llm import LLM, pmap
from metrics import sample_turn_indices


def critic_scores(dialogues: list[dict], backend: str | None, limit: int) -> list[dict]:
    """Pass 1: the cheap in-pipeline score, exactly as it is computed at decision time."""
    llm = LLM("critic", backend=backend)
    try:
        critic = Critic(llm, calibration=None, gate=False)
        pending: list[tuple[dict, dict]] = []
        for session in dialogues:
            for idx in sample_turn_indices(session):
                turn = next((t for t in session["turns"] if t["turn_index"] == idx), None)
                if turn is not None and len(pending) < limit:
                    pending.append((session, turn))

        def review(item: tuple[dict, dict]) -> dict:
            session, turn = item
            idx = turn["turn_index"]
            prefix = [t for t in session["turns"] if t["turn_index"] < idx]
            artifact = next((a for a in session.get("agent_artifacts", [])
                             if a.get("turn_index") == idx), {})
            verdict = critic.review(turn["text"], prefix, artifact.get("analyzer", {}),
                                    (artifact.get("strategist") or {}).get("plan", ""),
                                    turn.get("ladder_rung", "L3"), memory=None,
                                    phase=turn.get("phase", "listening"))
            return {"session_id": session["session_id"], "turn_index": idx,
                    "score": verdict.nonconformity, "ip_pred": verdict.ip_pred,
                    "violations": [v["category"] for v in verdict.grounding_violations]}

        rows = list(pmap(review, pending, llm))
    finally:
        llm.release()
    return rows


def judge_ip(dialogues: list[dict], rows: list[dict], backend: str | None) -> list[dict]:
    """Pass 2: the independent judge supplies the labels the gate is calibrated against."""
    by_session = {s["session_id"]: s for s in dialogues}
    judge = Judge(backend=backend)
    try:
        def label(row: dict) -> None:
            session = by_session[row["session_id"]]
            res = judge.score_turn("ip", session, row["turn_index"], session.get("profile_snapshot", {}))
            row["ip_norm"] = res.normalized
            row["ip_parse_failed"] = res.parse_failed
            row["judge_model"] = res.judge_model

        for _ in pmap(label, rows, judge.llm):
            pass
    finally:
        judge.llm.release()
    return [r for r in rows if not r.get("ip_parse_failed")]


def run(dialogues_path: Path, out_path: Path, alpha: float | None, backend: str | None,
        distribution_id: str, limit: int | None = None) -> dict:
    dialogues = read_jsonl(dialogues_path)
    if not dialogues:
        raise SystemExit(f"no dialogues at {dialogues_path}")
    n = int(limit or cfg("conformal.n_calibration_turns", default=300))
    rows = critic_scores(dialogues, backend, n)
    rows = judge_ip(dialogues, rows, backend)
    if not rows:
        raise SystemExit("every calibration turn failed to parse - fix the judge prompt first")
    cal = calibrate([r["score"] for r in rows], [r["ip_norm"] for r in rows], alpha=alpha,
                    distribution_id=distribution_id,
                    provenance={"dialogues": str(dialogues_path), "n_turns": len(rows),
                                "judge_model": rows[0].get("judge_model", "")})
    cal.save(out_path)
    write_json(out_path.with_name("calibration_rows.json"), rows)
    write_json(out_path.with_name("alpha_sweep.json"),
               sweep_alphas([r["score"] for r in rows], [r["ip_norm"] for r in rows]))
    return {"lambda_hat": cal.lambda_hat, "alpha": cal.alpha, "tau": cal.tau, "n": cal.n_calibration,
            "vacuous": cal.vacuous, "path": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the conformal critic gate.")
    ap.add_argument("--dialogues", required=True, help="ungated dialogues from held-out profiles")
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--distribution-id", default="own_generator_v1")
    args = ap.parse_args()
    dialogues = Path(args.dialogues)
    out = Path(args.out) if args.out else dialogues.parent.parent / "conformal" / "calibration.json"
    stats = run(dialogues, out, args.alpha, args.backend, args.distribution_id, args.limit)
    print(f"lambda_hat={stats['lambda_hat']:.3f} at alpha={stats['alpha']} tau={stats['tau']} "
          f"on n={stats['n']} turns"
          + ("  [VACUOUS: no threshold meets this budget - report it, do not raise alpha]"
             if stats["vacuous"] else ""))
    print(f"saved {stats['path']}")


if __name__ == "__main__":
    main()
