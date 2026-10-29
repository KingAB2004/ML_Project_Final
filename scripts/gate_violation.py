"""Realised gate violation rate, in and out of distribution (CPU only).

The conformal gate bounds R = share of turns that the gate releases AND the judge finds intrusive (IP_norm > tau)
by alpha, but only on turns exchangeable with the calibration split. Here R is measured on the judge-scored turns of
each gated arm: in distribution (cell D, step 12 on our test people) and out of distribution (cell D on ExTES
people), as the SOP asks: a measured quantity, not a guarantee. A turn counts as released when the gate let a draft
through (path released or revised); fallback turns are the template reflection and never count as violations.

    python scripts/gate_violation.py --out temp_results/17_gate_violation_rate
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = [("cell D, our test people (in distribution)", "runs/baselines_1000", "cellD_dec_gated",
         "runs/baselines_1000/conformal/calibration.json"),
        ("step 12, fine-tuned Generator, our test people (in distribution)", "runs/sft_agents", "sft_dec_gated_mem",
         "runs/sft_agents/conformal/calibration.json"),
        ("cell D, ExTES people (out of distribution)", "runs/baselines_1000", "cellD_extes",
         "runs/baselines_1000/conformal/calibration.json")]


def rate(rows: list[tuple[str, bool, bool]], reps: int = 2000) -> tuple[float, float, float, float]:
    """R (released and intrusive / all turns), its by-person bootstrap 95% CI, and intrusive share of released."""
    by = defaultdict(list)
    for person, released, bad in rows:
        by[person].append(released and bad)
    people, rng = list(by), random.Random(0)
    point = sum(v for l in by.values() for v in l) / len(rows)
    boots = []
    for _ in range(reps):
        draw = [v for p in (rng.choice(people) for _ in people) for v in by[p]]
        boots.append(sum(draw) / len(draw))
    boots.sort()
    rel = [bad for _, released, bad in rows if released]
    return point, boots[int(.025 * reps)], boots[int(.975 * reps)], (sum(rel) / len(rel) if rel else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/17_gate_violation_rate"))
    args = ap.parse_args()
    lines = [__doc__.split("\n\n")[0], "",
             "arm  scored turns  released share  R = released & IP>tau [95% CI]  alpha  intrusive share of released"]
    for label, run, arm, cal in ARMS:
        run, scores = ROOT / run, ROOT / run / "scores" / arm / "ip.jsonl"
        if not scores.exists():
            lines.append(f"{label}: no IP scores at {scores}, skipped")
            continue
        c = json.load(open(ROOT / cal))
        turns = {}
        for line in open(run / "dialogues" / f"{arm}.jsonl"):
            s = json.loads(line)
            for t in s["turns"]:
                turns[f"{s['session_id']}#{t['turn_index']}"] = (s["profile_id"], t.get("meta", {}).get("path"))
        rows = []
        for line in open(scores):
            x = json.loads(line)
            if x.get("normalized") is None or x["target"] not in turns:
                continue
            person, path = turns[x["target"]]
            rows.append((person, path in ("released", "revised"), x["normalized"] > c["tau"]))
        r, lo, hi, share = rate(rows)
        lines.append(f"{label}  {len(rows)}  {sum(rel for _, rel, _ in rows) / len(rows):.3f}  "
                     f"{r:.3f} [{lo:.3f}, {hi:.3f}]  {c['alpha']}  {share:.3f}   (tau {c['tau']}, "
                     f"lambda_hat {c['lambda_hat']})")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_violation_rate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
