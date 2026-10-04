"""How much usable information do the Analysis/Strategy thoughts carry about the reply? (GPU, forward passes)

Pointwise V-information (Ethayarajh, Choi & Swayamdipta 2022), with the two adapters of one size as the two
predictive families:

  PVI_i = log2 p_with(reply_i | context_i, thoughts_i) - log2 p_wo(reply_i | context_i)

p_with is the with_thoughts adapter, teacher-forced through the annotated <analysis> and <strategy>; p_wo is
the wo_thoughts adapter, trained on the same turns without them (the "null input" model of the definition).
The reply is the identical "<response>...</response>" + EOS string in both, so the two log-probabilities are
of the same token sequence (examples where the tokenization does not line up are skipped and counted).
Mean PVI estimates the V-usable information I_V(thoughts -> reply | context) in bits. PVI < 0 marks a turn
whose annotation made the gold reply LESS predictable: a candidate bad label.

Scored on the validation split (never trained on). The thoughts are the annotator's, so this measures the
information in gold thoughts, not in thoughts the model writes for itself.

  python extra/pvi.py                       # every size whose two adapters exist
  python extra/pvi.py --sizes 0.5B --limit 200

Writes runs/scaling/extra/pvi/: per_example_<size>.jsonl (kept, so a rerun only adds new sizes),
summary.json, report.md, pvi_hist.png, pvi_by_size.png.
"""
from __future__ import annotations

import argparse
import math
import statistics

from _shared import (MODEL_SIZES, ROOT, STUDY, adapter_dir, cluster_bootstrap, fmt, markdown_table, out_dir,
                     pyplot, read_jsonl, run_path, sizes_with_adapters, write_json, write_jsonl)

LN2 = math.log(2)
TAG = "<response>"


def paired_rows(split: str) -> list[tuple[dict, dict]]:
    """(with_thoughts row, wo_thoughts row) for the same supporter turn, joined on session and turn."""
    wo = {(r["session_id"], r["turn_index"]): r for r in read_jsonl(ROOT / "data" / "sft" / f"wo_thoughts_{split}.jsonl")}
    out = []
    for r in read_jsonl(ROOT / "data" / "sft" / f"with_thoughts_{split}.jsonl"):
        w = wo.get((r["session_id"], r["turn_index"]))
        if w and TAG in r["target"] and w["target"].lstrip().startswith(TAG):
            out.append((r, w))
    return out


def score_size(size: str, pairs: list[tuple[dict, dict]]) -> list[dict]:
    import _lm

    base = MODEL_SIZES[size]["base"]
    logs: dict[str, list] = {}
    for arm in ("with_thoughts", "wo_thoughts"):
        model, tok = _lm.load(base, str(adapter_dir(size, arm)))
        vals = []
        for k, (w, n) in enumerate(pairs):
            row = w if arm == "with_thoughts" else n
            start = row["target"].index(TAG)
            vals.append(_lm.target_logprobs(model, tok, row, scored_from=start))
            if (k + 1) % 250 == 0:
                print(f"  {size} {arm}: {k + 1}/{len(pairs)}", flush=True)
        logs[arm] = vals
        _lm.release(model)
    rows = []
    for (w, _), a, b in zip(pairs, logs["with_thoughts"], logs["wo_thoughts"]):
        ok = a is not None and b is not None and len(a) == len(b)
        rows.append({"session_id": w["session_id"], "profile_id": w["profile_id"], "turn_index": w["turn_index"],
                     "ladder_rung": w.get("ladder_rung"), "skipped": not ok,
                     "thought_chars": w["target"].index(TAG),
                     **({"n_tokens": len(a), "logp_with": sum(a), "logp_wo": sum(b),
                         "pvi_bits": (sum(a) - sum(b)) / LN2} if ok else {})})
    return rows


def summarise(rows: list[dict], reps: int, tag: str) -> dict:
    ok = [r for r in rows if not r["skipped"]]
    if not ok:
        return {"n": 0, "skipped": len(rows)}
    pvi = [r["pvi_bits"] for r in ok]
    by_profile: dict[str, list[dict]] = {}
    for r in ok:
        by_profile.setdefault(r["profile_id"], []).append(r)
    clusters = list(by_profile.values())

    def mean_pvi(groups):
        xs = [r["pvi_bits"] for g in groups for r in g]
        return statistics.fmean(xs) if xs else None

    def nll(arm):
        def f(groups):
            tot = sum(-r[f"logp_{arm}"] for g in groups for r in g)
            n = sum(r["n_tokens"] for g in groups for r in g)
            return tot / n if n else None
        return f

    # does more thought text buy more information? Spearman would need ties handling; Pearson on ranks is it
    def ranks(xs):
        order = sorted(range(len(xs)), key=xs.__getitem__)
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    rho = statistics.correlation(ranks([r["thought_chars"] for r in ok]), ranks(pvi)) if len(ok) > 2 else None
    by_rung = {}
    for rung in sorted({r["ladder_rung"] for r in ok if r.get("ladder_rung")}):
        xs = [r["pvi_bits"] for r in ok if r.get("ladder_rung") == rung]
        by_rung[rung] = {"n": len(xs), "mean_pvi_bits": statistics.fmean(xs)}
    return {
        "n": len(ok), "skipped": len(rows) - len(ok), "profiles": len(clusters),
        "v_information": statistics.fmean(pvi), "v_information_ci": cluster_bootstrap(clusters, mean_pvi, reps, f"{tag}:v"),
        "pvi_bits_per_token": sum(pvi) / sum(r["n_tokens"] for r in ok),
        "median_pvi_bits": statistics.median(pvi), "share_negative": sum(x < 0 for x in pvi) / len(pvi),
        "reply_nll_with": nll("with")(clusters), "reply_nll_wo": nll("wo")(clusters),
        "reply_nll_with_ci": cluster_bootstrap(clusters, nll("with"), reps, f"{tag}:nw"),
        "reply_nll_wo_ci": cluster_bootstrap(clusters, nll("wo"), reps, f"{tag}:nn"),
        "spearman_thought_length_vs_pvi": rho, "by_ladder_rung": by_rung,
    }


def plots(results: dict, per_size: dict, dest) -> list[str]:
    plt = pyplot()
    if plt is None or not results:
        return []
    made = []
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for size, rows in per_size.items():
        xs = [r["pvi_bits"] for r in rows if not r["skipped"]]
        lo, hi = sorted(xs)[int(0.01 * len(xs))], sorted(xs)[int(0.99 * len(xs)) - 1]
        ax.hist([x for x in xs if lo <= x <= hi], bins=60, alpha=0.5, label=f"{size} (mean {statistics.fmean(xs):.2f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("PVI (bits): thoughts -> reply, per validation turn")
    ax.set_ylabel("turns")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(dest / "pvi_hist.png", dpi=150)
    plt.close(fig)
    made.append("pvi_hist.png")

    sizes = list(results)
    x = [MODEL_SIZES[s]["params"] for s in sizes]
    y = [results[s]["v_information"] for s in sizes]
    err = [[y[i] - results[s]["v_information_ci"][0] for i, s in enumerate(sizes)],
           [results[s]["v_information_ci"][1] - y[i] for i, s in enumerate(sizes)]]
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.errorbar(x, y, yerr=err, marker="o", capsize=4)
    ax.set_xscale("log")
    for xi, yi, s in zip(x, y, sizes):
        ax.annotate(s, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("non-embedding parameters")
    ax.set_ylabel("V-information (bits / turn)")
    ax.set_title("Usable information in the thoughts", fontsize=9)
    fig.tight_layout()
    fig.savefig(dest / "pvi_by_size.png", dpi=150)
    plt.close(fig)
    made.append("pvi_by_size.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="Pointwise V-information of the thoughts, per model size.")
    ap.add_argument("--sizes", nargs="*", default=None, help="default: every size with both adapters")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None, help="validation turns (default: all)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--rescore", action="store_true", help="recompute sizes already scored")
    args = ap.parse_args()

    dest = out_dir(run_path(STUDY), "pvi")
    pairs = paired_rows(args.split)[: args.limit] if args.limit else paired_rows(args.split)
    print(f"{len(pairs)} paired {args.split} turns")
    per_size = {}
    for size in sizes_with_adapters(args.sizes):
        path = dest / f"per_example_{size}.jsonl"
        if path.exists() and not args.rescore:
            per_size[size] = read_jsonl(path)
            continue
        rows = score_size(size, pairs)
        write_jsonl(path, rows)
        per_size[size] = rows
    for f in sorted(dest.glob("per_example_*.jsonl")):            # sizes scored on an earlier run
        per_size.setdefault(f.stem.removeprefix("per_example_"), read_jsonl(f))
    per_size = {s: per_size[s] for s in MODEL_SIZES if s in per_size}
    if not per_size:
        raise SystemExit("no size has both adapters yet")

    results = {s: summarise(rows, args.reps, f"pvi:{s}") for s, rows in per_size.items()}
    figs = plots(results, per_size, dest)
    write_json(dest / "summary.json", {"split": args.split, "sizes": results})
    ci = lambda r, k: f"{fmt(r[k])} [{fmt(r[k + '_ci'][0])}, {fmt(r[k + '_ci'][1])}]"
    table = [{"size": s, "turns": r["n"], "V-info (bits/turn)": ci(r, "v_information"),
              "bits/token": fmt(r["pvi_bits_per_token"]), "PVI < 0": f"{r['share_negative']:.1%}",
              "reply NLL with": ci(r, "reply_nll_with"), "reply NLL wo": ci(r, "reply_nll_wo"),
              "rho(len, PVI)": fmt(r["spearman_thought_length_vs_pvi"])} for s, r in results.items()]
    lines = [f"# Usable information in the thoughts (PVI) - {args.split} split", "",
             "PVI = log2 p_with(reply | context, thoughts) - log2 p_wo(reply | context), per supporter turn; its "
             "mean is the V-usable information the gold Analysis/Strategy carry about the reply. 95 % CIs: "
             "bootstrap over profiles. Reply NLL is nats per reply token.", "",
             markdown_table(table, list(table[0])), ""]
    for s, r in results.items():
        if r["by_ladder_rung"]:
            lines += [f"**{s} by ladder rung:** " + ", ".join(f"{k} {v['mean_pvi_bits']:.2f} bits (n={v['n']})"
                                                           for k, v in r["by_ladder_rung"].items()), ""]
    lines += [f"![{f}]({f})" for f in figs]
    lines += ["", "Turns with PVI < 0 are listed in per_example_<size>.jsonl (sort by pvi_bits): annotations "
              "that made the gold reply less predictable are the first ones to audit."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
