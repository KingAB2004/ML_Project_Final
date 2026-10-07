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

from _shared import (MODEL_SIZES, ROOT, SFT_ARMS, STUDY, adapter_dir, cluster_bootstrap, eval_tag, family, fmt,
                     markdown_table,
                     out_dir, pyplot, read_jsonl, run_path, write_json)
from common import read_json
from report import collect_arm

EVAL_ROOT = "runs/scaling_eval"
TEST_METRICS = ("success", "ip", "pri")
FAMILIES = ("qwen2.5", "qwen3")               # Qwen3: controls for the self-generated data, never on the fitted curve
FAMILY_STYLE = {"with_thoughts": "^", "wo_thoughts": "v"}


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
    run = ROOT / EVAL_ROOT / eval_tag(size)
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
    have = [s for s in have if family(s) == "qwen2.5"]      # the slope is along the Qwen2.5 curve only
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


def size_axis(plt, ax, sizes) -> None:
    """Log-N axis labelled with the model sizes instead of 4 x 10^8-style ticks."""
    ax.set_xscale("log")
    ax.set_xticks([MODEL_SIZES[s]["params"] for s in sizes], [s for s in sizes], fontsize=8)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("model size (log non-embedding parameters)", fontsize=8)


def plots(sizes, train, tests, fits, gains, dest) -> list[str]:
    plt = pyplot()
    if plt is None:
        return []
    made = []
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for (arm, style), fam in ((a, f) for a in zip(SFT_ARMS, ("-o", "--s")) for f in FAMILIES):
        pts = [(MODEL_SIZES[s]["params"], train[(s, arm)]["best_eval_loss"]) for s in sizes
               if train.get((s, arm)) and family(s) == fam]
        if pts:
            ax.plot(*zip(*pts), style if fam == "qwen2.5" else FAMILY_STYLE[arm], label=f"{arm} ({fam})")
            fit = fits.get(arm) if fam == "qwen2.5" else None
            if fit:
                xs = np.geomspace(min(p[0] for p in pts), max(p[0] for p in pts), 50)
                ax.plot(xs, fit["a"] * xs ** -fit["b"], ":", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([MODEL_SIZES[s]["params"] for s in sizes], [f"{MODEL_SIZES[s]['params'] / 1e9:.2f}B\n({s})" for s in sizes],
                  fontsize=7)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    for axis in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):   # 0.70, not 7 x 10^-1
        axis(plt.FormatStrFormatter("%.2f"))
    ax.set_xlabel("non-embedding parameters N")
    ax.set_ylabel("best validation loss")
    ax.set_title("Validation loss vs size (arms differ in target)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "scaling_loss.png", dpi=150)
    plt.close(fig)
    made.append("scaling_loss.png")

    runs = [(s, a) for s in sizes for a in SFT_ARMS if train.get((s, a))]
    if runs:
        fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.0), squeeze=False, sharey=True)
        for ax, (s, a) in zip(axes[0], runs):
            tr = train[(s, a)]["train_loss"]
            k = 10                                       # moving average over 10 logging steps (100 optimizer steps)
            sm = [sum(l for _, l in tr[i - k + 1:i + 1]) / k for i in range(k - 1, len(tr))]
            ax.plot([e for e, _ in tr[k - 1:]], sm, lw=1, label="train")
            ax.plot(*zip(*train[(s, a)]["eval_loss_by_epoch"]), "o-", label="validation")
            ax.set_title(f"{s} {a}", fontsize=8)
            ax.set_xlabel("epoch")
        axes[0][0].set_ylabel("loss")
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "training_curves.png", dpi=150)
        plt.close(fig)
        made.append("training_curves.png")
        # one figure per size too, both arms on the same axes (copied next to that size's training results)
        for s in sizes:
            arms = [a for a in SFT_ARMS if train.get((s, a))]
            if not arms:
                continue
            fig, ax = plt.subplots(figsize=(4.8, 3.4))
            for i, (a, color) in enumerate(zip(arms, ("C0", "C1"))):
                tr = train[(s, a)]["train_loss"]
                k = min(10, len(tr))
                sm = [sum(l for _, l in tr[i - k + 1:i + 1]) / k for i in range(k - 1, len(tr))]
                ax.plot([e for e, _ in tr[k - 1:]], sm, lw=1, color=color, alpha=0.6, label=f"{a} train")
                ax.plot(*zip(*train[(s, a)]["eval_loss_by_epoch"]), "o-", color=color, label=f"{a} validation")
                be, bl = train[(s, a)]["best_epoch"], train[(s, a)]["best_eval_loss"]
                ax.annotate(f"best {bl:.3f}", (be, bl), textcoords="offset points", xytext=(4, 6 if i else -10), fontsize=6,
                            color=color)
            ax.set_title(f"{MODEL_SIZES[s]['base'].split('/')[-1]} QLoRA: loss per epoch", fontsize=8)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(dest / f"training_curves_{s}.png", dpi=150)
            plt.close(fig)
            made.append(f"training_curves_{s}.png")

    metrics = [m for m in TEST_METRICS if any((tests.get((s, a)) or {}).get(m) for s in sizes for a in SFT_ARMS)]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.2), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            for (arm, style), fam in ((a, f) for a in zip(SFT_ARMS, ("-o", "--s")) for f in FAMILIES):
                pts = [(s, tests[(s, arm)][m]) for s in sizes if (tests.get((s, arm)) or {}).get(m) and family(s) == fam]
                if not pts:
                    continue
                style, label = (style, arm) if fam == "qwen2.5" else (FAMILY_STYLE[arm], f"{arm} ({fam})")
                x = [MODEL_SIZES[s]["params"] for s, _ in pts]
                y = [r["mean"] for _, r in pts]
                cis = [cluster_bootstrap(list(r["per_profile"].values()), statistics.fmean, 1000, f"plot:{m}:{s}:{arm}")
                       for s, r in pts]
                ax.errorbar(x, y, yerr=[[yi - c[0] for yi, c in zip(y, cis)], [c[1] - yi for yi, c in zip(y, cis)]],
                            fmt=style, capsize=3, label=label)
            size_axis(plt, ax, sizes)
            ax.set_title({"success": "Success Rate", "ip": "Intrusiveness (IP)", "pri": "Reactance (PRI)"}[m], fontsize=9)
        axes[0][0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dest / "scaling_test.png", dpi=150)
        plt.close(fig)
        made.append("scaling_test.png")

    if gains:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for (m, g), fam in ((x, f) for x in gains.items() for f in FAMILIES):
            ss = [s for s in sizes if s in g and family(s) == fam]
            if not ss:
                continue
            x = [MODEL_SIZES[s]["params"] for s in ss]
            y = [g[s]["gain"] for s in ss]
            ax.errorbar(x, y, yerr=[[yi - g[s]["ci"][0] for yi, s in zip(y, ss)], [g[s]["ci"][1] - yi for yi, s in zip(y, ss)]],
                        marker="o" if fam == "qwen2.5" else "^", ls="-" if fam == "qwen2.5" else "none", capsize=3,
                        label=m if fam == "qwen2.5" else f"{m} ({fam})")
        ax.axhline(0, color="k", lw=0.8)
        size_axis(plt, ax, sizes)
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
        have = [s for s in sizes if train[(s, arm)] and family(s) == "qwen2.5"]
        fits[arm] = power_law([MODEL_SIZES[s]["params"] for s in have], [train[(s, arm)]["best_eval_loss"] for s in have])
    nll = {s: reply_nll_by_profile(s) for s in sizes}
    nll_sizes = [s for s in sizes if nll[s] and family(s) == "qwen2.5"]
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
             "Qwen3 rows are controls for the self-generated data (Qwen2.5-7B wrote the corpus): same recipe, "
             "another family, so they are shown but never enter the fits or the per-decade slopes.", "",
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
