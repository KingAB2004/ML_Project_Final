"""Phase 7 - tables, CIs, guards (PLAN Sec. 14.5, 14.6).

Reads only what earlier stages wrote. Four guards are built in and fail the build rather than footnoting:
  G1 no table cell without n;
  G2 no cell whose judge model equals the supporter's or the simulator's;
  G3 no out-of-distribution cell described as a guarantee - it is a measured violation rate;
  G4 every IP/PRI headline carries the judge-human agreement figure (or says it is missing).

Statistics: paired bootstrap at the PROFILE level (turns inside a dialogue are not independent), Holm
correction inside a family, and main effects plus interaction for the 2x2.

  python src/report.py --run runs/<run_id>
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Sequence

from common import REPORTS, cfg, load_config, read_json, read_jsonl, rng_for


class ReportGuardError(RuntimeError):
    pass


# --------------------------------------------------------------------------- statistics


def paired_bootstrap_diff(a: dict[str, float], b: dict[str, float], resamples: int | None = None,
                          tag: str = "diff") -> dict:
    """Mean difference with a 95% CI over the profiles present in both arms."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n": 0, "diff": None, "ci_low": None, "ci_high": None}
    diffs = [float(a[k]) - float(b[k]) for k in keys]
    reps = int(resamples or cfg("eval.bootstrap_resamples", default=10000))
    rng = rng_for(tag, "report_bootstrap")
    means = sorted(statistics.fmean(diffs[rng.randrange(len(diffs))] for _ in diffs) for _ in range(reps))
    return {"n": len(keys), "diff": statistics.fmean(diffs),
            "ci_low": means[int(0.025 * reps)], "ci_high": means[min(reps - 1, int(0.975 * reps))]}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm correction inside one family of comparisons."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out, reject_all = {}, True
    for i, (name, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        reject_all = reject_all and p <= threshold
        out[name] = {"p": p, "threshold": threshold, "significant": reject_all}
    return out


def two_by_two(cells: dict[str, float]) -> dict:
    """Main effects and interaction for decomposition x gate. Read the answer off the table, don't argue it."""
    need = ("mono_ungated", "mono_gated", "dec_ungated", "dec_gated")
    if any(k not in cells for k in need):
        return {"error": f"need all four cells: {need}"}
    a, b, c, d = (cells[k] for k in need)
    return {
        "effect_decomposition": ((c + d) / 2) - ((a + b) / 2),
        "effect_gate": ((b + d) / 2) - ((a + c) / 2),
        "interaction": (d - c) - (b - a),
        "cells": dict(cells),
    }


# --------------------------------------------------------------------------- guards


def guard_sample_size(cell: dict, where: str) -> None:
    if not cell.get("n"):
        raise ReportGuardError(f"G1 {where}: a reported cell has no sample size")


def guard_judge_separation(judge_model: str, where: str) -> None:
    roles = load_config("models")["roles"]
    for role in ("supporter", "simulator", "generator"):
        if judge_model and roles.get(role, {}).get("model") == judge_model:
            raise ReportGuardError(f"G2 {where}: judge '{judge_model}' is also role '{role}'")


def guard_distribution_language(text: str, in_distribution: bool, where: str) -> None:
    if not in_distribution and "guarantee" in text.lower():
        raise ReportGuardError(f"G3 {where}: out-of-distribution results may not be called a guarantee")


def guard_agreement(metric: str, agreement: dict | None, where: str) -> str:
    if metric not in ("ip", "pri"):
        return ""
    if not agreement:
        return " (judge-human agreement: NOT YET MEASURED - run human_eval/)"
    return (f" (judge-human: kappa={agreement.get('weighted_kappa')}, "
            f"alpha={agreement.get('krippendorff_alpha')}, rho={agreement.get('spearman_rho')})")


# --------------------------------------------------------------------------- assembly


def collect_arm(run_dir: Path, arm: str) -> dict:
    """Per-arm scores, keyed by profile so comparisons can be paired."""
    scores_dir = run_dir / "scores" / arm
    if not scores_dir.exists():
        scores_dir = run_dir / "scores"
    out: dict = {"arm": arm, "metrics": {}}
    for scale in ("success", "aels", "crs", "rac", "basic", "ip", "pri"):
        rows = read_jsonl(scores_dir / f"{scale}.jsonl")
        if not rows:
            continue
        by_profile: dict[str, list[float]] = {}
        for r in rows:
            if r.get("parse_failed"):
                continue
            pid = str(r.get("target", "")).split("-")[0]
            by_profile.setdefault(pid, []).append(float(r.get("normalized", 0.0)))
        per_profile = {k: statistics.fmean(v) for k, v in by_profile.items()}
        vals = list(per_profile.values())
        out["metrics"][scale] = {
            "n": len(vals),
            "mean": statistics.fmean(vals) if vals else None,
            "per_profile": per_profile,
            "judge_model": rows[0].get("judge_model", ""),
            "redactions": rows[0].get("context_redactions", []),
        }
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import metrics as metrics_mod

    out["aggregates"] = {}
    for scale in ("basic", "crs", "rac"):
        rows = read_jsonl(scores_dir / f"{scale}.jsonl")
        if rows:
            out["aggregates"][scale] = metrics_mod.aggregate(rows, scale)

    summary_path = run_dir / f"arm_summary_{arm}.json"
    if summary_path.exists():
        out["summary"] = read_json(summary_path)
    pri_path = scores_dir / "pri_summary.json"
    if pri_path.exists():
        out["pri_counterfactual"] = read_json(pri_path)
    return out


def markdown_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def fmt(x: object, places: int = 3) -> str:
    return f"{x:.{places}f}" if isinstance(x, (int, float)) else "-"


def build(run_dir: Path, out_path: Path | None = None) -> str:
    arms = sorted(p.stem for p in (run_dir / "dialogues").glob("*.jsonl")) if \
        (run_dir / "dialogues").exists() else []
    agreement = None
    agr_path = run_dir / "scores" / "agreement.json"
    if agr_path.exists():
        agreement = read_json(agr_path)

    collected = {arm: collect_arm(run_dir, arm) for arm in arms}
    lines = [f"# Results - {run_dir.name}", "",
             "Standalone benchmarks: the baseline corpus and construction pipeline are unavailable and the "
             "original work is Chinese, so comparability rests on the instruments (item wording published in "
             "`prompts/`), not on shared code.", ""]

    rows = []
    for arm, data in collected.items():
        in_dist = bool((data.get("summary") or {}).get("in_distribution", False))
        row = {"arm": arm, "distribution": "in" if in_dist else "out"}
        for scale in ("success", "ip", "pri", "aels", "crs", "rac", "basic"):
            cell = data["metrics"].get(scale)
            if not cell:
                row[scale] = "-"
                continue
            guard_sample_size(cell, f"{arm}/{scale}")
            guard_judge_separation(cell.get("judge_model", ""), f"{arm}/{scale}")
            row[scale] = f"{fmt(cell['mean'])} (n={cell['n']})"
            if scale in ("ip", "pri"):
                row[scale] += guard_agreement(scale, agreement, f"{arm}/{scale}")
        rows.append(row)
    lines += ["## Per-arm summary", "",
              markdown_table(rows, ["arm", "distribution", "success", "ip", "pri", "aels", "crs", "rac",
                                    "basic"]), ""]

    corpus_rows = []
    for arm, data in collected.items():
        summary = data.get("summary") or {}
        if not arm.startswith("corpus_"):
            continue
        basic = data["metrics"].get("basic", {})
        crs = data["metrics"].get("crs", {})
        rac = data["metrics"].get("rac", {})
        success = data["metrics"].get("success", {})
        agg = data.get("aggregates", {})
        corpus_rows.append({
            "training dataset": summary.get("supporter", arm),
            "profiles": "ExTES" if "extes_profiles" in arm else "ours",
            "SR": fmt(success.get("mean")),
            "Basic Avg": fmt(agg.get("basic", {}).get("basic_avg_100"), 1),
            "Aff": fmt(agg.get("crs", {}).get("affective_improvement"), 2),
            "Neg": fmt(agg.get("crs", {}).get("negative_helper"), 2),
            "Sup": fmt(agg.get("rac", {}).get("supportiveness"), 2),
            "Man": fmt(agg.get("rac", {}).get("management"), 2),
            "n": min([m.get("n", 0) for m in (success, basic, crs, rac) if m] or [0]),
        })
    if corpus_rows:
        lines += ["## Training-corpus comparison", "",
                  "One base model, one adapter per training corpus, the same user profiles and the same "
                  "judge - the shape of the baseline paper's Tables 3 and 4. `Basic Avg` is the mean of the "
                  "six basic metrics rescaled to 0-100; `Aff`/`Neg` are the Comforting Responses Scale "
                  "dimensions and `Sup`/`Man` the RAC dimensions, on the 1-7 Likert scale. Lower `Neg` is "
                  "better; every other column is higher-is-better.", "",
                  markdown_table(corpus_rows, ["training dataset", "profiles", "SR", "Basic Avg", "Aff",
                                               "Neg", "Sup", "Man", "n"]),
                  "", "Corpora other than ours carry no Analysis/Strategy annotation, so their adapters are "
                  "trained response-only - the same condition as our `w/o thoughts` arm. The item groupings "
                  "behind Aff/Neg/Sup/Man are ours and are listed in `src/metrics.py`; the instruments "
                  "publish items, not groupings.", ""]

    cal_path = run_dir / "conformal" / "calibration.json"
    if cal_path.exists():
        cal = read_json(cal_path)
        claim = (f"alpha={cal['alpha']}, tau={cal['tau']}, lambda_hat={fmt(cal['lambda_hat'])}, "
                 f"n_calibration={cal['n_calibration']}, bound={cal['bound']}")
        lines += ["## Conformal critic gate", "", f"- {claim}",
                  f"- calibration distribution: `{cal['distribution_id']}`",
                  "- The bound holds only for turns exchangeable with this calibration split. Arms marked "
                  "`out` above report a MEASURED violation rate, not a guarantee.", ""]
        if cal.get("vacuous"):
            lines.append("- **No threshold met this budget.** Reported as a finding; alpha was not raised "
                         "to manufacture one.\n")
        sweep_path = cal_path.with_name("alpha_sweep.json")
        if sweep_path.exists():
            sweep = read_json(sweep_path)
            lines += ["### Risk-coverage frontier", "",
                      markdown_table([{k: fmt(v) if isinstance(v, float) else v for k, v in r.items()}
                                      for r in sweep],
                                     ["alpha", "lambda_hat", "release_rate", "risk", "risk_ucb",
                                      "vacuous"]), ""]

    cells = {}
    mapping = {"cellA_mono_ungated": "mono_ungated", "cellB_mono_gated": "mono_gated",
               "cellC_dec_ungated": "dec_ungated", "cellD_dec_gated": "dec_gated"}
    for arm, key in mapping.items():
        cell = collected.get(arm, {}).get("metrics", {}).get("success")
        if cell and cell.get("mean") is not None:
            cells[key] = cell["mean"]
    if len(cells) == 4:
        effects = two_by_two(cells)
        lines += ["## 2x2: decomposition x gate (Success Rate)", "",
                  f"- main effect of decomposition: {fmt(effects['effect_decomposition'])}",
                  f"- main effect of the gate: {fmt(effects['effect_gate'])}",
                  f"- interaction: {fmt(effects['interaction'])}",
                  "", "A gate-only effect is a legitimate result and is reported as such.", ""]

    pairs = [("sft_with_thoughts", "sft_wo_thoughts"), ("cellD_dec_gated", "cellA_mono_ungated"),
             ("mem_needstate", "mem_dense"), ("mem_needstate", "mem_event")]
    diff_rows = []
    for a, b in pairs:
        ma = collected.get(a, {}).get("metrics", {}).get("success", {})
        mb = collected.get(b, {}).get("metrics", {}).get("success", {})
        if ma.get("per_profile") and mb.get("per_profile"):
            d = paired_bootstrap_diff(ma["per_profile"], mb["per_profile"], tag=f"{a}v{b}")
            diff_rows.append({"comparison": f"{a} - {b}", "diff": fmt(d["diff"]),
                              "ci": f"[{fmt(d['ci_low'])}, {fmt(d['ci_high'])}]", "n": d["n"]})
    if diff_rows:
        lines += ["## Pre-registered paired comparisons (Success Rate)", "",
                  markdown_table(diff_rows, ["comparison", "diff", "ci", "n"]),
                  "", "Resampling unit is the profile. A CI containing zero is reported as no effect.", ""]

    lines += ["## Honesty notes", "",
              "- IP and PRI measure resistance realized by a SIMULATED user; transfer to human reactance is "
              "out of scope.",
              "- The four agent roles ran sequentially on one resident model; concurrency was not achieved "
              "on a 12 GB budget.",
              "- See `reports/substitutions.md` for every reduced-scale substitution.", ""]

    text = "\n".join(lines)
    for arm, data in collected.items():
        guard_distribution_language(text, bool((data.get("summary") or {}).get("in_distribution", True)),
                                    arm)
    out_path = out_path or REPORTS / "results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    ensure_substitutions_file()
    return str(out_path)


def ensure_substitutions_file() -> None:
    path = REPORTS / "substitutions.md"
    if path.exists():
        return
    path.write_text(
        "# Substitutions\n\nEvery deviation from the planned configuration, with its reason.\n\n"
        "| date | component | planned | actual | reason |\n|---|---|---|---|---|\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble results tables with the reporting guards on.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path = build(Path(args.run), Path(args.out) if args.out else None)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
