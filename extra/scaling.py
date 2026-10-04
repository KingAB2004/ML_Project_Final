"""Scaling study: does the supporter get better with size, and do the thoughts help small models more? (CPU)

Collects, for every size in _shared.MODEL_SIZES (0.5B, 3B, 7B, same QLoRA recipe, same 1000-profile SFT files):
  training     best validation loss and its epoch (train_summary_<arm>.json log_history, or the last
               checkpoint's trainer_state.json for runs trained before train.py saved it)
  test         Success Rate, IP, PRI (judge) and counterfactual PRI from scripts/scaling_eval.sh
               (runs/scaling_eval/qwen2.5_<size>/), every size against the same 7B seeker
  reply NLL    from extra/pvi.py, the same reply tokens under every model, per validation turn

Fits
  power law    L(N) = a N^-b  ->  log L = log a - b log N, least squares on the sizes present (N = non-embedding
               parameters). With three sizes the fit has one residual degree of freedom, so the CI on b
               (bootstrap over validation profiles, from the per-turn reply NLL) is the honest statement, not
               the point estimate. L(N) = a N^-b + c needs a fourth size to be identified and is not fitted.
  thoughts gain  D(N) = metric_with - metric_wo per size, paired over the same test profiles; interaction =
               slope of D against log10 N (negative for Success = the thoughts help small models more).
Validation losses of the two arms are NOT compared with each other: the with_thoughts target also contains the
thoughts. Arms are compared on reply NLL (same tokens) and on the test metrics.

  python extra/scaling.py
Writes runs/scaling/extra/scaling/: summary.json, report.md, scaling_loss.png, scaling_test.png, thoughts_gain.png.
"""
from __future__ import annotations

import argparse
import math
import statistics

import numpy as np

from _shared import (MODEL_SIZES, ROOT, SFT_ARMS, STUDY, adapter_dir, cluster_bootstrap, fmt, markdown_table,
                     out_dir, pyplot, read_jsonl, run_path, write_json)
from common import read_json
from report import collect_arm

EVAL_ROOT = "runs/scaling_eval"
TEST_METRICS = ("success", "ip", "pri")


def training_record(size: str, arm: str) -> dict | None:
    run = ROOT / MODEL_SIZES[size]["dir"]
    history = None
    for f in (run / f"train_summary_{arm}.json", run / "train_summary.json"):
        if f.exists():
            s = read_json(f)
            if s.get("arm") == arm and s.get("log_history"):
                history = s["log_history"]
                break
    if history is None:
        ckpts = sorted(adapter_dir(size, arm).glob("checkpoint-*/trainer_state.json"),
                       key=lambda p: int(p.parent.name.split("-")[1]))
        if not ckpts:
            return None
        history = read_json(ckpts[-1])["log_history"]
    evals = [(h["epoch"], h["eval_loss"]) for h in history if "eval_loss" in h]
    train = [(h["epoch"], h["loss"]) for h in history if "loss" in h]
    if not evals:
        return None
    best_epoch, best = min(evals, key=lambda e: e[1])
    return {"eval_loss_by_epoch": evals, "train_loss": train, "best_eval_loss": best, "best_epoch": best_epoch}


def test_record(size: str, arm: str) -> dict | None:
    run = ROOT / EVAL_ROOT / f"qwen2.5_{size}"
    name = f"sft_{arm}"
    if not (run / "scores" / name).exists():
        return None
    got = collect_arm(run, name)
    out = {m: {"mean": got["metrics"][m]["mean"], "per_profile": got["metrics"][m]["per_profile"]}
           for m in TEST_METRICS if m in got["metrics"]}
    if "pri_counterfactual" in got:
        out["pri_counterfactual"] = got["pri_counterfactual"]
    return out


def power_law(n: list[float], loss: list[float]) -> dict | None:
    if len(n) < 2:
        return None
    x, y = np.log(n), np.log(loss)
    slope, intercept = np.polyfit(x, y, 1)
    return {"a": float(math.exp(intercept)), "b": float(-slope),
            "residual_df": len(n) - 2}


def reply_nll_by_profile(size: str) -> dict[str, dict[str, tuple[float, int]]] | None:
    """{profile: {arm: (sum NLL, tokens)}} from pvi.py's per-turn file."""
    path = run_path(STUDY) / "extra" / "pvi" / f"per_example_{size}.jsonl"
    if not path.exists():
        return None
    out: dict[str, dict[str, list]] = {}
    for r in read_jsonl(path):
        if r["skipped"]:
            continue
        d = out.setdefault(r["profile_id"], {"with": [0.0, 0], "wo": [0.0, 0]})
        for arm in ("with", "wo"):
            d[arm][0] -= r[f"logp_{arm}"]
            d[arm][1] += r["n_tokens"]
    return out


def slope_ci(sizes: list[str], by_size: dict[str, dict], arm: str, reps: int) -> tuple | None:
    """Bootstrap CI on b from reply NLL, resampling validation profiles (the same draw for every size)."""
    common = sorted(set.intersection(*(set(by_size[s]) for s in sizes)))
    if len(sizes) < 2 or not common:
        return None
    n = [MODEL_SIZES[s]["params"] for s in sizes]

    def stat(groups):
        losses = []
        for s in sizes:
            tot = sum(by_size[s][p][arm][0] for p in groups)
            tok = sum(by_size[s][p][arm][1] for p in groups)
            losses.append(tot / tok)
        fit = power_law(n, losses)
        return fit["b"] if fit else None

    point = stat(common)
    return point, cluster_bootstrap(common, stat, reps, f"scaling:b:{arm}")


def gain_stats(sizes: list[str], tests: dict, metric: str, reps: int) -> dict | None:
    have = [s for s in sizes if all((tests.get((s, a)) or {}).get(metric) for a in SFT_ARMS)]
    if not have:
        return None
    per = {}
    for s in have:
        w = tests[(s, "with_thoughts")][metric]["per_profile"]
        o = tests[(s, "wo_thoughts")][metric]["per_profile"]
        per[s] = {p: w[p] - o[p] for p in set(w) & set(o)}
    out = {}
    for s in have:
        ds = list(per[s].values())
        out[s] = {"gain": statistics.fmean(ds), "ci": cluster_bootstrap(ds, statistics.fmean, reps, f"gain:{metric}:{s}"),
                  "n_profiles": len(ds)}
    if len(have) >= 2:
        common = sorted(set.intersection(*(set(per[s]) for s in have)))
        x = [math.log10(MODEL_SIZES[s]["params"]) for s in have]

        def slope(groups):
            y = [statistics.fmean(per[s][p] for p in groups) for s in have]
            return float(np.polyfit(x, y, 1)[0])

        if common:
            out["slope_per_decade"] = {"value": slope(common),
                                       "ci": cluster_bootstrap(common, slope, reps, f"gain-slope:{metric}")}
    return out


def plots(sizes, train, tests, fits, gains, dest) -> list[str]:
    plt = pyplot()
    if plt is None:
        return []
    made = []
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for arm, style in zip(SFT_ARMS, ("-o", "--s")):
        pts = [(MODEL_SIZES[s]["params"], train[(s, arm)]["best_eval_loss"]) for s in sizes if train.get((s, arm))]
        if pts:
            ax.plot(*zip(*pts), style, label=arm)
            fit = fits.get(arm)
            if fit:
                xs = np.geomspace(min(p[0] for p in pts), max(p[0] for p in pts), 50)
                ax.plot(xs, fit["a"] * xs ** -fit["b"], ":", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("best validation loss")
    ax.set_title("Validation loss vs size (arms differ in target)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "scaling_loss.png", dpi=150)
    plt.close(fig)
    made.append("scaling_loss.png")

    metrics = [m for m in TEST_METRICS if any((tests.get((s, a)) or {}).get(m) for s in sizes for a in SFT_ARMS)]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            for arm, style in zip(SFT_ARMS, ("-o", "--s")):
                pts = [(s, tests[(s, arm)][m]) for s in sizes if (tests.get((s, arm)) or {}).get(m)]
                if not pts:
                    continue
                x = [MODEL_SIZES[s]["params"] for s, _ in pts]
                y = [r["mean"] for _, r in pts]
                cis = [cluster_bootstrap(list(r["per_profile"].values()), statistics.fmean, 1000, f"plot:{m}:{s}:{arm}")
                       for s, r in pts]
                ax.errorbar(x, y, yerr=[[yi - c[0] for yi, c in zip(y, cis)], [c[1] - yi for yi, c in zip(y, cis)]],
                            fmt=style, capsize=3, label=arm)
            ax.set_xscale("log")
            ax.set_title({"success": "Success Rate", "ip": "Intrusiveness (IP)", "pri": "Reactance (PRI)"}[m], fontsize=9)
            ax.set_xlabel("N")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "scaling_test.png", dpi=150)
        plt.close(fig)
        made.append("scaling_test.png")

    if gains:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for m, g in gains.items():
            ss = [s for s in sizes if s in g]
            x = [MODEL_SIZES[s]["params"] for s in ss]
            y = [g[s]["gain"] for s in ss]
            ax.errorbar(x, y, yerr=[[yi - g[s]["ci"][0] for yi, s in zip(y, ss)], [g[s]["ci"][1] - yi for yi, s in zip(y, ss)]],
                        marker="o", capsize=3, label=m)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("with_thoughts - wo_thoughts")
        ax.set_title("What the thoughts buy, by size", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "thoughts_gain.png", dpi=150)
        plt.close(fig)
        made.append("thoughts_gain.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-size scaling summary.")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    sizes = list(MODEL_SIZES)
    train = {(s, a): training_record(s, a) for s in sizes for a in SFT_ARMS}
    tests = {(s, a): test_record(s, a) for s in sizes for a in SFT_ARMS}
    sizes = [s for s in sizes if any(train[(s, a)] or tests[(s, a)] for a in SFT_ARMS)]
    if not sizes:
        raise SystemExit("nothing trained or evaluated yet")

    fits = {}
    for arm in SFT_ARMS:
        have = [s for s in sizes if train[(s, arm)]]
        fits[arm] = power_law([MODEL_SIZES[s]["params"] for s in have], [train[(s, arm)]["best_eval_loss"] for s in have])
    nll = {s: reply_nll_by_profile(s) for s in sizes}
    nll_sizes = [s for s in sizes if nll[s]]
    reply_b = {arm: slope_ci(nll_sizes, nll, arm, args.reps) for arm in ("with", "wo")} if len(nll_sizes) >= 2 else {}
    gains = {m: g for m in TEST_METRICS if (g := gain_stats(sizes, tests, m, args.reps))}

    dest = out_dir(run_path(STUDY), "scaling")
    figs = plots(sizes, train, tests, fits, gains, dest)
    write_json(dest / "summary.json", {
        "sizes": {s: MODEL_SIZES[s] for s in sizes},
        "training": {f"{s}|{a}": train[(s, a)] for s in sizes for a in SFT_ARMS},
        "test": {f"{s}|{a}": ({m: {"mean": v["mean"], "n": len(v["per_profile"])} if isinstance(v, dict) and "per_profile" in v else v
                               for m, v in tests[(s, a)].items()} if tests[(s, a)] else None)
                 for s in sizes for a in SFT_ARMS},
        "power_law_val_loss": fits, "power_law_reply_nll": reply_b, "thoughts_gain": gains})

    rows = []
    for s in sizes:
        for a in SFT_ARMS:
            t, e = train[(s, a)], tests[(s, a)] or {}
            rows.append({"size": s, "arm": a, "N (non-emb)": f"{MODEL_SIZES[s]['params'] / 1e9:.2f}B",
                         "best val loss": fmt(t and t["best_eval_loss"], 4), "best epoch": fmt(t and t["best_epoch"], 0),
                         **{m: fmt(e.get(m, {}).get("mean")) for m in TEST_METRICS},
                         "PRI (counterfactual)": fmt(e.get("pri_counterfactual", {}).get("pri"))})
    lines = ["# Scaling study: Qwen2.5 0.5B / 3B / 7B, same QLoRA recipe, 1000-profile corpus", "",
             "Test metrics: every size plays against the same Qwen2.5-7B seeker (scripts/scaling_eval.sh), "
             "scored by the Mistral-Nemo judge. Validation losses of the two arms are not comparable to each "
             "other (the with_thoughts target also holds the thoughts).", "", markdown_table(rows, list(rows[0])), ""]
    for arm, f in fits.items():
        if f:
            lines.append(f"- {arm}: best val loss ~ {f['a']:.3g} N^-{f['b']:.3f} ({f['residual_df']} residual df)")
    for arm, r in reply_b.items():
        if r:
            lines.append(f"- reply NLL ({arm} thoughts): b = {fmt(r[0])}, 95 % CI [{fmt(r[1][0])}, {fmt(r[1][1])}] "
                         f"(bootstrap over validation profiles)")
    for m, g in gains.items():
        parts = [f"{s} {fmt(g[s]['gain'])} [{fmt(g[s]['ci'][0])}, {fmt(g[s]['ci'][1])}]" for s in sizes if s in g]
        lines.append(f"- thoughts gain on {m}: " + "; ".join(parts))
        if "slope_per_decade" in g:
            sl = g["slope_per_decade"]
            lines.append(f"  - per decade of N: {fmt(sl['value'])} [{fmt(sl['ci'][0])}, {fmt(sl['ci'][1])}]"
                         + ("  (CI excludes 0)" if sl["ci"][0] is not None and (sl["ci"][0] > 0 or sl["ci"][1] < 0) else ""))
    lines += ["", *[f"![{f}]({f})" for f in figs], "",
              "Every number is measured on simulated seekers and read by an LLM judge."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
