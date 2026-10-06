"""Plots of the untuned baselines and the 2 x 2 of one evaluation run (base_instruct, reactive_baseline, cellA-D).

Reads only what the pipeline already wrote under the run: dialogues/, scores/<arm>/, conformal/. No model calls.
Every interval is a 95 % percentile bootstrap over profiles (a profile's sessions and turns are not independent).

  scores_by_arm.png         mean judge score per arm and scale, with CI           (which arm is better, and is it noise?)
  score_distributions.png   per-dialogue Success (raw 1-7) and AELS               (ceiling / floor of the judge)
  ip_per_turn.png           per-turn intrusiveness, with the gate's tau            (how often a turn oversteps)
  pri_counterfactual.png    seeker reactance after the real turn vs a neutral one  (does the supporter provoke pushback?)
  gate_calibration.png      conformal risk curve and nonconformity scores          (why the gate does or does not fire)
  dialogue_diagnostics.png  reply length, leaked thoughts, copied seeker lines, gate paths (data-quality checks)

  python extra/baseline_plots.py --run results/v1/runs/week1_v1    -> <run>/extra/baseline_plots/
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from _shared import cluster_bootstrap, out_dir, pyplot, read_jsonl, run_path, write_json

ARMS = ("base_instruct", "reactive_baseline", "cellA_mono_ungated", "cellB_mono_gated", "cellC_dec_ungated",
        "cellD_dec_gated")
SCALES = ("success", "aels", "basic", "crs", "rac", "ip", "pri")
TITLES = {"success": "Success (need named) ↑", "aels": "AELS active listening ↑", "basic": "Basic qualities ↑",
          "crs": "CRS comforting ↑", "rac": "RAC competence ↑", "ip": "Intrusiveness IP ↓",
          "pri": "Reactance PRI (observed) ↓"}
LEAK = re.compile(r"\*{0,2}(analysis|strategy)\*{0,2}\s*:", re.I)


def profile_of(target: str) -> str:
    return target.split("-")[0]


def short(arm: str) -> str:
    return arm.replace("_mono_", " mono ").replace("_dec_", " dec ").replace("_baseline", "").replace("_instruct", "")


def load_scores(run: Path) -> dict[str, dict[str, list[dict]]]:
    """arm -> scale -> parsed rows"""
    out: dict = defaultdict(dict)
    for arm in ARMS:
        for scale in SCALES:
            f = run / "scores" / arm / f"{scale}.jsonl"
            if f.exists():
                rows = [r for r in read_jsonl(f) if not r.get("parse_failed")]
                if rows:
                    out[arm][scale] = rows
    return out


def mean_ci(rows: list[dict], tag: str) -> tuple[float, float, float]:
    by_profile: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_profile[profile_of(r["target"])].append(r["normalized"])
    clusters = list(by_profile.values())
    stat = lambda cs: statistics.fmean(v for c in cs for v in c)  # noqa: E731
    lo, hi = cluster_bootstrap(clusters, stat, 2000, tag)
    return stat(clusters), lo, hi


def dialogue_stats(run: Path) -> dict[str, dict]:
    stats = {}
    for arm in ARMS:
        f = run / "dialogues" / f"{arm}.jsonl"
        if not f.exists():
            continue
        words, leaks, copies, seeker_n, paths = [], 0, 0, 0, Counter()
        for s in read_jsonl(f):
            seen: set[str] = set()
            for t in s["turns"]:
                text = (t.get("text") or "").strip()
                if t["role"] == "supporter" and t.get("meta", {}).get("source") != "opener_pool":
                    words.append(len(text.split()))
                    leaks += bool(LEAK.search(text))
                    paths[t.get("meta", {}).get("path", "released")] += 1
                elif t["role"] == "user":
                    seeker_n += 1
                    copies += text.lower() in seen
                seen.add(text.lower())
        stats[arm] = {"sessions": sum(1 for _ in open(f)), "supporter_turns": len(words),
                      "words_median": statistics.median(words) if words else 0, "words": words,
                      "leak_rate": leaks / max(1, len(words)), "seeker_copy_rate": copies / max(1, seeker_n),
                      "paths": dict(paths)}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = run_path(args.run)
    dest = out_dir(run, "baseline_plots")
    plt = pyplot()
    if plt is None:
        raise SystemExit("matplotlib is needed for the plots")
    scores = load_scores(run)
    dstats = dialogue_stats(run)
    summary: dict = {"scores": {}, "dialogues": {a: {k: v for k, v in d.items() if k != "words"}
                                                 for a, d in dstats.items()}}
    scored = [a for a in ARMS if a in scores]
    colors = {a: f"C{i}" for i, a in enumerate(ARMS)}

    # 1. mean score per arm and scale
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, scale in zip(axes.flat, SCALES):
        arms = [a for a in scored if scale in scores[a]]
        for i, a in enumerate(arms):
            m, lo, hi = mean_ci(scores[a][scale], f"{a}:{scale}")
            summary["scores"].setdefault(a, {})[scale] = {"mean": m, "ci": [lo, hi], "n": len(scores[a][scale])}
            ax.bar(i, m, color=colors[a], alpha=0.8)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="k", capsize=4)
            ax.text(i, hi, f"{m:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(arms)), [short(a) for a in arms], fontsize=7)
        ax.set_title(TITLES[scale], fontsize=9)
        ax.set_ylim(0, 1.08 if scale not in ("ip", "pri") else None)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.5, "bars: mean normalised score (0-1)\nwhiskers: 95 % CI, bootstrap over profiles\n"
                       "IP / PRI: lower is better", fontsize=8, va="center")
    fig.suptitle(f"Judge scores per arm ({run.name})", fontsize=10)
    fig.tight_layout()
    fig.savefig(dest / "scores_by_arm.png", dpi=150)
    plt.close(fig)

    # 2. per-dialogue distributions: Success raw 1-7 and AELS mean
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.8 / max(1, len(scored))
    for i, a in enumerate(scored):
        if "success" in scores[a]:
            c = Counter(round(r["mean"]) for r in scores[a]["success"])
            n = sum(c.values())
            a1.bar([k + (i - len(scored) / 2 + 0.5) * width for k in range(1, 8)],
                   [c.get(k, 0) / n for k in range(1, 8)], width, color=colors[a], label=short(a))
        if "aels" in scores[a]:
            a2.hist([r["mean"] for r in scores[a]["aels"]], bins=[x / 4 for x in range(4, 29)], alpha=0.5,
                    color=colors[a], label=short(a), density=True)
    a1.set_xlabel("Success rating (1 = need missed, 7 = need named and confirmed)")
    a1.set_ylabel("share of dialogues")
    a1.legend(fontsize=7)
    a1.set_title("Success per dialogue", fontsize=9)
    a2.set_xlabel("AELS mean item rating (1-7)")
    a2.set_title("Active listening per dialogue: near the ceiling of 7", fontsize=9)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "score_distributions.png", dpi=150)
    plt.close(fig)

    # 3. per-turn IP with the gate's tau
    calib_path = run / "conformal" / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    tau = calib.get("tau", 0.5)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for a in scored:
        if "ip" in scores[a]:
            vals = [r["normalized"] for r in scores[a]["ip"]]
            ax.hist(vals, bins=[x / 20 for x in range(21)], alpha=0.5, color=colors[a], density=True,
                    label=f"{short(a)} ({100 * sum(v > tau for v in vals) / len(vals):.1f} % > tau)")
    ax.axvline(tau, color="k", ls="--", lw=1)
    ax.text(tau, ax.get_ylim()[1] * 0.9, f" tau = {tau}", fontsize=8)
    ax.set_xlabel("IP per supporter turn (0 = not intrusive, 1 = maximally intrusive)")
    ax.set_ylabel("density")
    ax.set_title("Intrusiveness per turn (judge)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "ip_per_turn.png", dpi=150)
    plt.close(fig)

    # 4. counterfactual PRI: factual vs control and the paired difference
    pri = {a: json.loads((run / "scores" / a / "pri_summary.json").read_text()) for a in ARMS
           if (run / "scores" / a / "pri_summary.json").exists()}
    if pri:
        summary["pri_counterfactual"] = pri
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
        for i, (a, p) in enumerate(pri.items()):
            a1.bar([i - 0.2, i + 0.2], [p["mean_factual"], p["mean_control"]], 0.4,
                   color=[colors[a], "lightgray"], edgecolor="k", lw=0.5)
            a2.errorbar(p["pri"], i, xerr=[[p["pri"] - p["ci_low"]], [p["ci_high"] - p["pri"]]], fmt="o",
                        color=colors[a], capsize=4)
        a1.set_xticks(range(len(pri)), [short(a) for a in pri], fontsize=8)
        a1.set_ylabel("mean seeker reactance (0-1)")
        a1.set_title("Reactance after the real turn (colour) vs a neutral reflection (grey)", fontsize=8)
        a2.axvline(0, color="k", lw=0.8)
        a2.set_yticks(range(len(pri)), [short(a) for a in pri], fontsize=8)
        a2.set_xlabel("PRI = real - neutral (95 % CI)")
        a2.set_title("Counterfactual PRI: 0 = no extra pushback", fontsize=8)
        fig.tight_layout()
        fig.savefig(dest / "pri_counterfactual.png", dpi=150)
        plt.close(fig)

    # 5. conformal gate: risk curve and score distribution
    rows_path = run / "conformal" / "calibration_rows.json"
    if calib and rows_path.exists():
        rows = json.loads(rows_path.read_text())
        curve = calib["risk_curve"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
        lam = [c["lambda"] for c in curve]
        a1.plot(lam, [c["release_rate"] for c in curve], label="release rate")
        a1.plot(lam, [c["risk"] for c in curve], label="violation rate among released")
        a1.plot(lam, [c["risk_ucb"] for c in curve], ls="--", label="upper confidence bound")
        a1.axhline(calib["alpha"], color="k", lw=0.8, ls=":")
        a1.axvline(calib["lambda_hat"], color="r", lw=1)
        a1.text(calib["lambda_hat"], 0.5, f"lambda_hat = {calib['lambda_hat']} ", color="r", ha="right",
                fontsize=8)
        a1.text(0, calib["alpha"], f" alpha = {calib['alpha']}", fontsize=8, va="bottom")
        a1.set_xlabel("threshold lambda (release a turn if its score <= lambda)")
        a1.set_title(f"Conformal risk control, n = {calib['n_calibration']} calibration turns", fontsize=9)
        a1.legend(fontsize=7)
        viol = [r["score"] for r in rows if r.get("ip_norm") is not None and r["ip_norm"] > tau]
        ok = [r["score"] for r in rows if r.get("ip_norm") is not None and r["ip_norm"] <= tau]
        a2.hist([ok, viol], bins=[x / 20 for x in range(21)], stacked=True, color=["C2", "C3"],
                label=[f"judge IP <= tau ({len(ok)})", f"judge IP > tau = violation ({len(viol)})"])
        a2.set_xlabel("gate nonconformity score (0.7 critic IP + 0.3 rule violations)")
        a2.set_ylabel("calibration turns")
        a2.set_title("What the gate sees, coloured by the judge's verdict", fontsize=9)
        a2.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "gate_calibration.png", dpi=150)
        plt.close(fig)

    # 6. dialogue diagnostics over every arm, scored or not
    if dstats:
        arms = list(dstats)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].boxplot([dstats[a]["words"] for a in arms], showfliers=False)
        axes[0].set_xticks(range(1, len(arms) + 1), [short(a) for a in arms], fontsize=7, rotation=20)
        axes[0].set_ylabel("words per supporter turn")
        axes[0].set_title("Reply length (a confound in the 2 x 2)", fontsize=9)
        x = range(len(arms))
        axes[1].bar([i - 0.2 for i in x], [100 * dstats[a]["leak_rate"] for a in arms], 0.4,
                    label="supporter turns showing an 'Analysis:' or 'Strategy:' label")
        axes[1].bar([i + 0.2 for i in x], [100 * dstats[a]["seeker_copy_rate"] for a in arms], 0.4,
                    label="seeker turns copying an earlier line")
        axes[1].set_xticks(list(x), [short(a) for a in arms], fontsize=7, rotation=20)
        axes[1].set_ylabel("% of turns")
        axes[1].set_title("Data-quality problems", fontsize=9)
        axes[1].set_ylim(0, 1.4 * max(100 * max(d["leak_rate"], d["seeker_copy_rate"]) for d in dstats.values()) + 1)
        axes[1].legend(fontsize=7, loc="upper right")
        bottom = [0.0] * len(arms)
        for path, color in (("released", "C2"), ("revised", "C1"), ("fallback", "C3")):
            share = [100 * dstats[a]["paths"].get(path, 0) / max(1, dstats[a]["supporter_turns"]) for a in arms]
            axes[2].bar(list(x), share, bottom=bottom, color=color, label=path)
            bottom = [b + s for b, s in zip(bottom, share)]
        axes[2].set_xticks(list(x), [short(a) for a in arms], fontsize=7, rotation=20)
        axes[2].set_ylabel("% of supporter turns")
        axes[2].set_title("What the gate did with each turn", fontsize=9)
        axes[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        fig.savefig(dest / "dialogue_diagnostics.png", dpi=150)
        plt.close(fig)

    write_json(dest / "summary.json", summary)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
