"""Gate recovery vs cost (CPU only).

Offline curve, from the calibration rows (critic score s and judge IP per draft, ungated): for every threshold
lambda, cost = share of drafts the gate would stop (sent to revision), recovery = share of intrusive drafts
(judge IP > tau) it stops, risk = intrusive share among the drafts it releases. One curve for the untuned
Generator (runs/baselines_1000) and one for the fine-tuned Generator (step 12a, runs/sft_agents).

Realised point, from the 2 x 2 run: cost = revised + fallback share of turns in the gated cell, benefit = drop in
judge IP and PRI from the ungated to the gated cell (A -> B one call, C -> D four agents), with Success alongside.

    python scripts/recovery_cost.py --out temp_results/14_gate_recovery_vs_cost
"""
import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURVES = {"untuned Generator": ROOT / "runs/baselines_1000/conformal",
          "fine-tuned Generator (step 12a)": ROOT / "runs/sft_agents/conformal"}
CELLS = [("one call", "cellA_mono_ungated", "cellB_mono_gated"),
         ("four agents", "cellC_dec_ungated", "cellD_dec_gated")]


def curve(rows: list[dict], tau: float) -> list[dict]:
    rows = [r for r in rows if not r.get("ip_parse_failed")]
    bad = [r["ip_norm"] > tau for r in rows]
    out = []
    for i in range(41):
        lam = i / 40
        stop = [r["score"] > lam for r in rows]
        released = [b for b, s in zip(bad, stop) if not s]
        out.append({"lambda": lam, "cost": sum(stop) / len(rows),
                    "recovery": sum(s for b, s in zip(bad, stop) if b) / max(1, sum(bad)),
                    "risk": sum(released) / len(released) if released else 0.0})
    return out


def mean_by_person(run: Path, cell: str, scale: str) -> dict[str, list[float]]:
    sessions = {json.loads(l)["session_id"]: json.loads(l)["profile_id"] for l in open(run / "dialogues" / f"{cell}.jsonl")}
    by = defaultdict(list)
    for line in open(run / "scores" / cell / f"{scale}.jsonl"):
        x = json.loads(line)
        if x.get("normalized") is not None:
            by[sessions[x["target"].split("#")[0]]].append(x["normalized"])
    return by


def diff_ci(a: dict, b: dict, reps: int = 2000) -> tuple[float, float, float]:
    """b - a, bootstrap over people (unpaired cells: each resample draws people independently per cell)."""
    rng = random.Random(0)
    pa, pb = list(a), list(b)
    point = st.mean(v for l in b.values() for v in l) - st.mean(v for l in a.values() for v in l)
    ds = sorted(st.mean(v for p in (rng.choice(pb) for _ in pb) for v in b[p])
                - st.mean(v for p in (rng.choice(pa) for _ in pa) for v in a[p]) for _ in range(reps))
    return point, ds[int(.025 * reps)], ds[int(.975 * reps)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/14_gate_recovery_vs_cost"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [__doc__.split("\n\n")[0], ""]
    curves = {}
    for name, d in CURVES.items():
        if not (d / "calibration_rows.json").exists():
            lines.append(f"{name}: no calibration rows at {d}, skipped")
            continue
        cal = json.load(open(d / "calibration.json"))
        rows = json.load(open(d / "calibration_rows.json"))
        c = curves[name] = curve(rows, cal["tau"])
        at = min(c, key=lambda p: abs(p["lambda"] - cal["lambda_hat"]))
        lines += [f"== {name}: {len(rows)} calibration drafts, tau {cal['tau']}, alpha {cal['alpha']}, "
                  f"lambda_hat {cal['lambda_hat']}",
                  f"   at lambda_hat: cost {at['cost']:.3f} (drafts stopped), recovery {at['recovery']:.3f} "
                  f"(intrusive drafts stopped), intrusive share of released {at['risk']:.3f}, "
                  f"released-and-intrusive share of all drafts {at['risk'] * (1 - at['cost']):.3f} (the controlled risk)",
                  "   lambda  cost  recovery  risk"]
        lines += [f"   {p['lambda']:.3f}  {p['cost']:.3f}  {p['recovery']:.3f}  {p['risk']:.3f}" for p in c[::4]]
        lines.append("")
    run = ROOT / "runs/baselines_1000"
    lines.append("== realised in the 2 x 2 run (gate at lambda_hat 0.20, untuned 7B, 100 test people)")
    for label, ungated, gated in CELLS:
        paths = Counter(t.get("meta", {}).get("path") for l in open(run / "dialogues" / f"{gated}.jsonl")
                        for t in json.loads(l)["turns"]
                        if t["role"] == "supporter" and t.get("meta", {}).get("source") != "opener_pool")
        n = sum(paths.values())
        lines.append(f"   {label}: {ungated} -> {gated}; gated turns released {paths['released'] / n:.1%}, "
                     f"revised {paths['revised'] / n:.1%}, fallback {paths['fallback'] / n:.1%}")
        for scale in ("ip", "pri", "success"):
            d, lo, hi = diff_ci(mean_by_person(run, ungated, scale), mean_by_person(run, gated, scale))
            lines.append(f"      {scale:8s} gated - ungated {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (out / "recovery_vs_cost.txt").write_text("\n".join(lines) + "\n")
    json.dump(curves, open(out / "recovery_vs_cost.json", "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for name, c in curves.items():
            ax.plot([p["cost"] for p in c], [p["recovery"] for p in c], marker=".", label=name)
            cal = json.load(open(CURVES[name] / "calibration.json"))
            at = min(c, key=lambda p: abs(p["lambda"] - cal["lambda_hat"]))
            ax.scatter([at["cost"]], [at["recovery"]], s=60, zorder=3)
            ax.annotate(f"λ̂={cal['lambda_hat']}", (at["cost"], at["recovery"]), textcoords="offset points",
                        xytext=(6, -12), fontsize=8)
        ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1, label="random stopping")
        ax.set_xlabel("cost: share of drafts the gate stops")
        ax.set_ylabel("recovery: share of intrusive drafts stopped")
        ax.set_title("Conformal gate: recovery vs cost (calibration drafts)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "recovery_vs_cost.png", dpi=150)
    except ImportError:
        pass
    print("\n".join(lines))


if __name__ == "__main__":
    main()
