"""Are the need-state memory's beliefs calibrated? (one judge pass, then CPU)

Enhancement 4 keeps each inferred need as a belief with a status (hypothesis / confirmed / disconfirmed /
resolved) and a confidence. Nothing so far checks those numbers against the truth, which every profile
records: its need chain. Here every final memory node (runs/<id>/memory/<profile>_<arm>.jsonl) is matched
by the judge against that profile's true chain nodes (depth >= 1, all sessions), and the memory's confidence
is scored as a probability forecast of "this hypothesis is a real need".

Proper scoring  Brier score and log loss; Murphy's decomposition Brier = REL - RES + UNC (+ a within-bin
                term), i.e. miscalibration, minus discrimination, plus irreducible base-rate uncertainty.
Calibration     expected calibration error over 10 bins, and a reliability diagram.
Discrimination  AUROC (Mann-Whitney).
Status audit    P(real | confirmed) and, the costly error, P(real | disconfirmed): a true need the memory
                marked disconfirmed is blocked from re-proposal (memory.py), so this rate is the price of the
                disconfirmation mechanism the SOP argues for.
Recalibration   logistic regression (IRLS, small L2) on logit(confidence), evidence counts, depth and
                dormancy, cross-validated over profile-grouped folds: can the evidence the memory already
                stores give better probabilities than its hand-set confidence rules?

The judge never sees the system's confidence or status, only the two texts.

  python extra/memory_calibration.py --run runs/v3_50
"""
from __future__ import annotations

import argparse
import math
import statistics

import numpy as np

from _shared import fill, fmt, markdown_table, out_dir, pyplot, rate_items, read_extra_prompt, read_jsonl, \
    rng_for, run_path, write_json, write_jsonl
from judge import assert_judge_separate
from llm import LLM, pmap

THRESHOLD = 5
BINS = 10


def load_nodes(run_dir) -> list[dict]:
    """Final memory nodes joined with their profile's true chain (from the arm's dialogues)."""
    truth: dict[tuple[str, str], list[str]] = {}
    for f in sorted((run_dir / "dialogues").glob("*.jsonl")):
        for s in read_jsonl(f):
            chain = (s.get("profile_snapshot") or {}).get("need_chain") or []
            texts = truth.setdefault((s.get("profile_id"), f.stem), [])
            for node in chain:
                if isinstance(node, dict) and int(node.get("depth", 0)) >= 1 and node.get("text") not in texts:
                    texts.append(node["text"])
    nodes = []
    for f in sorted((run_dir / "memory").glob("*.jsonl")):
        pid, arm = f.stem.split("_", 1)
        ref = truth.get((pid, arm))
        if not ref:
            continue
        for n in read_jsonl(f):
            nodes.append({"profile_id": pid, "arm": arm, "node_id": n["node_id"], "text": n.get("text", ""),
                          "status": n.get("status"), "confidence": float(n.get("confidence") or 0.0),
                          "depth": int(n.get("depth") or 0), "dormant": bool(n.get("dormant")),
                          "n_support": len(n.get("supporting_spans") or []),
                          "n_contra": len(n.get("contradicting_spans") or []), "reference": ref})
    return nodes


def label_nodes(llm, nodes: list[dict]) -> None:
    template = read_extra_prompt("memory_match.md")

    def one(n: dict) -> None:
        prompt = fill(template, reference="\n".join(f"- {t}" for t in n["reference"]), hypothesis=n["text"])
        items = rate_items(llm, prompt, 1)
        n["match_rating"] = items[1] if items else None
        n["real"] = None if items is None else int(items[1] >= THRESHOLD)

    list(pmap(one, nodes, llm))


# --------------------------------------------------------------------------- scoring


def brier_decomposition(p: np.ndarray, y: np.ndarray, bins: int = BINS) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ybar = y.mean()
    rel = res = ece = 0.0
    curve = []
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        w = m.mean()
        pk, yk = p[m].mean(), y[m].mean()
        rel += w * (pk - yk) ** 2
        res += w * (yk - ybar) ** 2
        ece += w * abs(pk - yk)
        curve.append({"bin": b, "n": int(m.sum()), "mean_confidence": float(pk), "observed": float(yk)})
    brier = float(((p - y) ** 2).mean())
    return {"brier": brier, "reliability": rel, "resolution": res, "uncertainty": float(ybar * (1 - ybar)),
            "within_bin": brier - (rel - res + ybar * (1 - ybar)), "ece": ece, "curve": curve}


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def auroc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def features(nodes: list[dict]) -> np.ndarray:
    c = np.clip(np.array([n["confidence"] for n in nodes]), 0.01, 0.99)
    return np.column_stack([np.ones(len(nodes)), np.log(c / (1 - c)),
                            np.log1p([n["n_support"] for n in nodes]), np.log1p([n["n_contra"] for n in nodes]),
                            [n["depth"] for n in nodes], [float(n["dormant"]) for n in nodes]])


def logistic_irls(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    """Newton / IRLS for penalised logistic regression (intercept unpenalised)."""
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - mu) - pen * w
        H = (X.T * (mu * (1 - mu))) @ X + np.diag(pen)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def cross_validated_recalibration(nodes: list[dict], y: np.ndarray, folds: int = 5) -> dict:
    fold = np.array([rng_for(n["profile_id"], "memory_cv").randrange(folds) for n in nodes])
    X = features(nodes)
    pred = np.zeros(len(nodes))
    for k in range(folds):
        train, test = fold != k, fold == k
        if not test.any() or len(set(y[train].tolist())) < 2:
            pred[test] = y[train].mean() if train.any() else 0.5
            continue
        pred[test] = 1 / (1 + np.exp(-X[test] @ logistic_irls(X[train], y[train])))
    p_raw = np.array([n["confidence"] for n in nodes])
    return {"brier_raw": float(((p_raw - y) ** 2).mean()), "brier_recalibrated": float(((pred - y) ** 2).mean()),
            "logloss_raw": log_loss(p_raw, y), "logloss_recalibrated": log_loss(pred, y),
            "coefficients_full_data": dict(zip(["intercept", "logit_confidence", "log1p_support",
                                                "log1p_contra", "depth", "dormant"],
                                               logistic_irls(X, y).round(4).tolist()))}


def summarise(nodes: list[dict]) -> dict:
    p = np.array([n["confidence"] for n in nodes])
    y = np.array([n["real"] for n in nodes], dtype=float)
    out = {"n": len(nodes), "base_rate": float(y.mean()), "logloss": log_loss(p, y), "auroc": auroc(p, y),
           **brier_decomposition(p, y)}
    out["by_status"] = {s: {"n": sum(n["status"] == s for n in nodes),
                            "p_real": statistics.fmean(n["real"] for n in nodes if n["status"] == s)}
                        for s in sorted({n["status"] for n in nodes})}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration of the need-state memory's beliefs.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    run_dir = run_path(args.run)
    nodes = load_nodes(run_dir)
    if not nodes:
        raise SystemExit("no memory nodes with a matching dialogue file under this run")
    assert_judge_separate("judge")
    llm = LLM("judge", backend=args.backend, call_log=run_dir / "calls.jsonl")
    try:
        label_nodes(llm, nodes)
    finally:
        llm.release()
    dest = out_dir(run_dir, "memory_calibration")
    write_jsonl(dest / "labelled_nodes.jsonl", nodes)
    nodes = [n for n in nodes if n["real"] is not None]
    if len(nodes) < 10:
        raise SystemExit(f"only {len(nodes)} nodes labelled: too few to score")

    pooled = summarise(nodes)
    arms = {a: summarise([n for n in nodes if n["arm"] == a]) for a in sorted({n["arm"] for n in nodes})}
    y = np.array([n["real"] for n in nodes], dtype=float)
    recal = cross_validated_recalibration(nodes, y) if len(set(y.tolist())) == 2 else None
    write_json(dest / "summary.json", {"pooled": pooled, "arms": arms, "recalibration": recal})

    rows = [{"set": name, "nodes": r["n"], "real": fmt(r["base_rate"]), "Brier": fmt(r["brier"]),
             "REL": fmt(r["reliability"], 4), "RES": fmt(r["resolution"], 4), "ECE": fmt(r["ece"]),
             "AUROC": fmt(r["auroc"]),
             "P(real|confirmed)": fmt(r["by_status"].get("confirmed", {}).get("p_real")),
             "P(real|disconfirmed)": fmt(r["by_status"].get("disconfirmed", {}).get("p_real"))}
            for name, r in [("pooled", pooled), *arms.items()]]
    lines = [f"# Need-state memory calibration - {run_dir.name}", "",
             "Each final memory node's confidence is scored as a forecast that it is one of the profile's true "
             "chain needs (judge match >= 5/7). Brier = REL - RES + UNC: lower REL is better calibrated, higher "
             "RES better discrimination.", "", markdown_table(rows, list(rows[0])), ""]
    if recal:
        lines += [f"Profile-grouped 5-fold recalibration from the evidence the memory stores: Brier "
                  f"{fmt(recal['brier_raw'])} -> {fmt(recal['brier_recalibrated'])}, log loss "
                  f"{fmt(recal['logloss_raw'])} -> {fmt(recal['logloss_recalibrated'])}.", ""]
    lines += ["P(real | disconfirmed) is the share of TRUE needs the memory blocked from re-proposal."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt = pyplot()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        for name, r in [("pooled", pooled), *arms.items()]:
            ax.plot([c["mean_confidence"] for c in r["curve"]], [c["observed"] for c in r["curve"]],
                    marker="o", label=name)
        ax.set_xlabel("memory confidence")
        ax.set_ylabel("observed share of real needs")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(dest / "reliability.png", dpi=150)
        plt.close(fig)
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
