"""Do IP and PRI measure what they claim? Reliability, factor structure and item response theory on the
per-item judge ratings the pipeline already stored (CPU only, no model calls).

Data: runs/<id>/scores/<arm>/{pri,ip}.jsonl, one row per rated turn with all seven item scores (1-7).

Reliability   Cronbach alpha = k/(k-1) (1 - sum var_i / var_total), overall and per hypothesised subscale,
              plus corrected item-total correlations.
Dimensions    Horn's parallel analysis: keep factors whose correlation-matrix eigenvalue beats the 95th
              percentile of eigenvalues from random normal data of the same n x p.
Factor model  maximum-likelihood factor analysis Sigma = L L' + Psi fitted by EM (Rubin & Thayer 1982) for
              m = 1..3 factors: log-likelihood, BIC, Bartlett-corrected chi-square test against the saturated
              model, RMSEA; varimax-rotated loadings. PRI's theory (Dillard & Shen 2005, cited in the SOP) says
              reactance is anger AND negative cognition, so a 2-factor PRI should split items 1-2 from 3-5.
IRT           Samejima's graded response model, P(Y >= c | theta) = sigmoid(a (theta - b_c)), fitted by
              marginal maximum a posteriori with EM over Gauss-Hermite quadrature (Bock & Aitkin 1981) and
              a N(0, 1) prior on log a, the usual guard against runaway slopes on near-duplicate items.
              Discrimination a and the item information I(theta) = sum_c P_c'(theta)^2 / P_c(theta) say which
              items carry the measurement; a < 0.65 is "low" (Baker 2001) and a candidate to drop, which also
              makes the judge cheaper; a very high a next to another item's says the two are redundant. Sparse categories are merged first (a category needs >= 5 ratings).

Rows are turns, nested in sessions; the models treat them as independent respondents (stated in the report).

  python extra/psychometrics.py --run runs/v3_50 [--scales pri ip]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from _shared import chi2_sf, fmt, markdown_table, out_dir, pyplot, read_jsonl, run_path, write_json
from common import derive_seed

SUBSCALES = {
    "pri": {"anger": [1, 2], "negative_cognition": [3, 4, 5], "withdrawal": [6, 7]},
    "ip": {},
}
MIN_CATEGORY = 5
LOG_A_PRIOR_SD = 1.0   # N(0, 1) prior on log a (MAP): an item that is nearly a copy of another, or almost
                       # never leaves its floor category, otherwise drives a towards infinity


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def load_items(run_dir: Path, scale: str) -> tuple[np.ndarray, list[int]]:
    rows = []
    for f in sorted((run_dir / "scores").glob(f"*/{scale}.jsonl")):
        rows += [r for r in read_jsonl(f) if not r.get("parse_failed") and r.get("items")]
    if not rows:
        return np.zeros((0, 0)), []
    items = sorted({int(k) for r in rows for k in r["items"]})
    X = np.array([[round(float(r["items"][str(i)])) for i in items] for r in rows
                  if all(str(i) in r["items"] for i in items)], dtype=float)
    return X, items


# --------------------------------------------------------------------------- classical test theory


def cronbach_alpha(X: np.ndarray) -> float | None:
    k = X.shape[1]
    total_var = X.sum(axis=1).var(ddof=1)
    if k < 2 or total_var <= 0:
        return None
    return float(k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / total_var))


def item_total(X: np.ndarray) -> list[float | None]:
    out = []
    for j in range(X.shape[1]):
        rest = np.delete(X, j, axis=1).sum(axis=1)
        sd = X[:, j].std() * rest.std()
        out.append(float(np.corrcoef(X[:, j], rest)[0, 1]) if sd > 0 else None)
    return out


def parallel_analysis(X: np.ndarray, reps: int = 200, tag: str = "pa") -> dict:
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    observed = np.sort(np.linalg.eigvalsh(np.nan_to_num(R)))[::-1]
    rng = np.random.default_rng(derive_seed(tag, "parallel_analysis"))
    sims = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
                     for _ in range(reps)])
    threshold = np.percentile(sims, 95, axis=0)
    keep = 0
    for o, t in zip(observed, threshold):
        if o <= t:
            break
        keep += 1
    return {"eigenvalues": observed.tolist(), "random_95th": threshold.tolist(), "factors_retained": keep}


# --------------------------------------------------------------------------- factor analysis


def fa_em(S: np.ndarray, n: int, m: int, iters: int = 5000, tol: float = 1e-9) -> dict:
    """ML factor analysis of a correlation/covariance matrix S by EM. Returns loadings, uniquenesses, fit."""
    p = S.shape[0]
    vals, vecs = np.linalg.eigh(S)
    L = vecs[:, ::-1][:, :m] * np.sqrt(np.maximum(vals[::-1][:m], 1e-6))
    psi = np.maximum(np.diag(S - L @ L.T), 0.05)

    def loglik(L, psi):
        sigma = L @ L.T + np.diag(psi)
        logdet = np.linalg.slogdet(sigma)[1]
        return -0.5 * n * (p * math.log(2 * math.pi) + logdet + np.trace(np.linalg.solve(sigma, S)))

    ll = loglik(L, psi)
    for _ in range(iters):
        sigma = L @ L.T + np.diag(psi)
        beta = np.linalg.solve(sigma, L).T                        # m x p:  L' Sigma^-1
        ezz = np.eye(m) - beta @ L + beta @ S @ beta.T            # E[z z'] averaged over the sample
        L = S @ beta.T @ np.linalg.inv(ezz)
        psi = np.maximum(np.diag(S - L @ beta @ S), 1e-3)         # floor: a Heywood case is flagged below
        new_ll = loglik(L, psi)
        if abs(new_ll - ll) < tol:
            ll = new_ll
            break
        ll = new_ll
    sigma = L @ L.T + np.diag(psi)
    df = ((p - m) ** 2 - (p + m)) / 2
    stat = (n - 1 - (2 * p + 5) / 6 - 2 * m / 3) * (
        np.linalg.slogdet(sigma)[1] - np.linalg.slogdet(S)[1] + np.trace(np.linalg.solve(sigma, S)) - p)
    k_params = p * m + p - m * (m - 1) / 2
    return {"m": m, "loglik": float(ll), "bic": float(-2 * ll + k_params * math.log(n)),
            "chi2": float(stat), "df": df, "p": chi2_sf(stat, df) if df > 0 else None,
            "rmsea": math.sqrt(max(stat - df, 0) / (df * (n - 1))) if df > 0 else None,
            "loadings": orient(varimax(L) if m > 1 else L).tolist(), "uniqueness": psi.tolist(),
            "heywood": bool((psi <= 1.001e-3).any())}


def orient(L: np.ndarray) -> np.ndarray:
    """A factor's sign is arbitrary: flip each so its largest loading is positive."""
    return L * np.where(L[np.abs(L).argmax(axis=0), range(L.shape[1])] < 0, -1, 1)


def varimax(L: np.ndarray, iters: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Kaiser's varimax: the orthogonal rotation maximising the variance of squared loadings per factor."""
    p, m = L.shape
    R = np.eye(m)
    d = 0.0
    for _ in range(iters):
        LR = L @ R
        u, s, vt = np.linalg.svd(L.T @ (LR ** 3 - LR @ np.diag((LR ** 2).sum(axis=0)) / p))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R


# --------------------------------------------------------------------------- graded response model


def collapse(col: np.ndarray, min_count: int = MIN_CATEGORY) -> np.ndarray | None:
    """Merge sparse rating categories into their neighbour and relabel 0..C-1. None if fewer than 2 remain."""
    values = sorted(set(col.tolist()))
    groups = [[v] for v in values]
    count = lambda g: int(sum((col == v).sum() for v in g))
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i, g in enumerate(groups):
            if count(g) < min_count:
                j = i - 1 if i > 0 else i + 1
                groups[min(i, j)] = groups[min(i, j)] + groups[max(i, j)]
                del groups[max(i, j)]
                changed = True
                break
    if len(groups) < 2:
        return None
    mapping = {v: c for c, g in enumerate(groups) for v in g}
    return np.array([mapping[v] for v in col.tolist()])


def _grm_probs(params: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Category probabilities (len(theta) x C) from [log a, b_1, log(b_2 - b_1), ...]."""
    a = math.exp(params[0])
    b = np.cumsum(np.concatenate([[params[1]], np.exp(params[2:])]))
    star = sigmoid(a * (theta[:, None] - b[None, :]))
    star = np.hstack([np.ones((len(theta), 1)), star, np.zeros((len(theta), 1))])
    return np.clip(star[:, :-1] - star[:, 1:], 1e-12, 1)


def _m_step(params: np.ndarray, theta: np.ndarray, r: np.ndarray, steps: int = 25) -> np.ndarray:
    """Maximise sum_{q,c} r[q,c] log P_c(theta_q) + log prior(a) by gradient ascent with backtracking
    (numeric gradient; an item has at most 7 parameters)."""
    f = lambda x: float((r * np.log(_grm_probs(x, theta))).sum()) - 0.5 * (x[0] / LOG_A_PRIOR_SD) ** 2
    cur = f(params)
    for _ in range(steps):
        g = np.array([(f(params + e) - f(params - e)) / 2e-5 for e in np.eye(len(params)) * 1e-5])
        t = 1.0
        while t > 1e-6:
            cand = params + t * g / max(1.0, np.abs(g).max())
            val = f(cand)
            if val > cur:
                params, cur = cand, val
                break
            t /= 2
        else:
            break
    return params


def fit_grm(Y: list[np.ndarray], n_quad: int = 31, cycles: int = 100, tol: float = 1e-4) -> dict:
    """Bock-Aitkin EM. Y: one integer column (0..C_j-1) per item, all the same length."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    prior = weights / weights.sum()                               # N(0, 1) on the quadrature grid
    n = len(Y[0])
    params = []
    for y in Y:
        c = int(y.max()) + 1
        cum = np.array([(y >= k).mean() for k in range(1, c)])
        b = -np.log(np.clip(cum, 1e-3, 1 - 1e-3) / (1 - np.clip(cum, 1e-3, 1 - 1e-3)))
        b = np.maximum.accumulate(b + np.arange(len(b)) * 1e-3)
        params.append(np.concatenate([[0.0, b[0]], np.log(np.maximum(np.diff(b), 1e-2))]))
    prev = -math.inf
    for _ in range(cycles):
        logL = np.zeros((n, n_quad))
        probs = [_grm_probs(pj, nodes) for pj in params]
        for y, P in zip(Y, probs):
            logL += np.log(P[:, y]).T
        joint = logL + np.log(prior)[None, :]
        mx = joint.max(axis=1, keepdims=True)
        marg = mx[:, 0] + np.log(np.exp(joint - mx).sum(axis=1))
        post = np.exp(joint - marg[:, None])
        ll = float(marg.sum())
        for j, y in enumerate(Y):
            c = int(y.max()) + 1
            r = np.stack([post[y == k].sum(axis=0) for k in range(c)], axis=1)
            params[j] = _m_step(params[j], nodes, r)
        if ll - prev < tol:
            break
        prev = ll
    grid = np.linspace(-3, 3, 61)
    items = []
    for pj in params:
        a = math.exp(pj[0])
        b = np.cumsum(np.concatenate([[pj[1]], np.exp(pj[2:])]))
        P = _grm_probs(pj, grid)
        star = sigmoid(a * (grid[:, None] - b[None, :]))
        dstar = a * star * (1 - star)
        dstar = np.hstack([np.zeros((len(grid), 1)), dstar, np.zeros((len(grid), 1))])
        info = ((dstar[:, :-1] - dstar[:, 1:]) ** 2 / P).sum(axis=1)
        items.append({"a": a, "b": b.tolist(), "info": info.tolist(),
                      "peak_theta": float(grid[int(np.argmax(info))])})
    return {"loglik": ll, "theta_grid": grid.tolist(), "items": items,
            "test_info": np.sum([it["info"] for it in items], axis=0).tolist()}


# --------------------------------------------------------------------------- report


def analyse(X: np.ndarray, items: list[int], scale: str) -> dict:
    n, p = X.shape
    out = {"n": n, "items": items, "means": X.mean(axis=0).tolist(), "sds": X.std(axis=0, ddof=1).tolist(),
           "alpha": cronbach_alpha(X), "item_total": item_total(X), "subscales": {}}
    for name, idx in SUBSCALES.get(scale, {}).items():
        cols = [items.index(i) for i in idx if i in items]
        out["subscales"][name] = {"items": idx, "alpha": cronbach_alpha(X[:, cols]) if len(cols) > 1 else None}
    keep = X.std(axis=0) > 0
    Xk = X[:, keep]
    if Xk.shape[1] >= 2:
        R = np.corrcoef(Xk, rowvar=False)
        i, j = np.unravel_index(np.argmax(np.abs(R - np.eye(len(R)))), R.shape)
        ids = [it for it, k in zip(items, keep) if k]
        out["most_correlated_pair"] = {"items": [ids[i], ids[j]], "r": float(R[i, j])}
    if Xk.shape[1] >= 3 and n > Xk.shape[1] + 5:
        out["parallel_analysis"] = parallel_analysis(Xk, tag=scale)
        R = np.corrcoef(Xk, rowvar=False)
        out["factor_models"] = [fa_em(R, n, m) for m in range(1, 4)
                                if ((Xk.shape[1] - m) ** 2 - (Xk.shape[1] + m)) / 2 >= 0]
        out["fa_items"] = [i for i, k in zip(items, keep) if k]
    cols = [(i, collapse(X[:, j])) for j, i in enumerate(items)]
    cols = [(i, c) for i, c in cols if c is not None]
    if len(cols) >= 2:
        grm = fit_grm([c for _, c in cols])
        for (i, c), it in zip(cols, grm["items"]):
            it.update({"item": i, "categories": int(c.max()) + 1,
                       "flag": ("low" if it["a"] < 0.65 else "moderate" if it["a"] < 1.35 else
                                "high" if it["a"] < 1.7 else "very high" if it["a"] < 4 else
                                "very high: redundant or near-floor item?")})
        out["grm"] = grm
    return out


def report_lines(scale: str, r: dict) -> list[str]:
    lines = [f"## {scale.upper()} ({r['n']} rated turns, items {r['items']})", "",
             f"Cronbach alpha = {fmt(r['alpha'])}" + "".join(
                 f"; {k} alpha = {fmt(v['alpha'])}" for k, v in r["subscales"].items()), ""]
    rows = [{"item": i, "mean": fmt(m, 2), "sd": fmt(s, 2), "item-total r": fmt(t)}
            for i, m, s, t in zip(r["items"], r["means"], r["sds"], r["item_total"])]
    if "grm" in r:
        by = {it["item"]: it for it in r["grm"]["items"]}
        for row in rows:
            it = by.get(row["item"])
            row["GRM a"] = fmt(it["a"], 2) if it else "-"
            row["a flag"] = it["flag"] if it else "constant"
            row["info peak theta"] = fmt(it["peak_theta"], 1) if it else "-"
    lines += [markdown_table(rows, list(rows[0])), ""]
    if "most_correlated_pair" in r:
        pair = r["most_correlated_pair"]
        lines += [f"Most correlated item pair: {pair['items'][0]} and {pair['items'][1]} (r = {fmt(pair['r'])}). "
                  "A slope above about 4 usually means local dependence (two items asking the same thing) or an "
                  "item that almost never leaves its floor category, not a uniquely good item.", ""]
    if "parallel_analysis" in r:
        pa = r["parallel_analysis"]
        lines.append(f"Parallel analysis retains **{pa['factors_retained']}** factor(s) (eigenvalues "
                     f"{', '.join(fmt(e, 2) for e in pa['eigenvalues'][:4])} vs random 95th "
                     f"{', '.join(fmt(e, 2) for e in pa['random_95th'][:4])}).")
        lines += ["", markdown_table([{"factors": f["m"], "loglik": fmt(f["loglik"], 1), "BIC": fmt(f["bic"], 1),
                                       "chi2 (df)": f"{fmt(f['chi2'], 1)} ({f['df']:.0f})", "p": fmt(f["p"], 4),
                                       "RMSEA": fmt(f["rmsea"]), "Heywood": f["heywood"]}
                                      for f in r["factor_models"]],
                                     ["factors", "loglik", "BIC", "chi2 (df)", "p", "RMSEA", "Heywood"]), ""]
        two = next((f for f in r["factor_models"] if f["m"] == 2), None)
        if two:
            lines += ["Varimax loadings, 2 factors:", "", markdown_table(
                [{"item": i, "F1": fmt(l[0], 2), "F2": fmt(l[1], 2)} for i, l in zip(r["fa_items"], two["loadings"])],
                ["item", "F1", "F2"]), ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Reliability, factor structure and IRT for IP and PRI.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--scales", nargs="*", default=["pri", "ip"])
    args = ap.parse_args()

    run_dir = run_path(args.run)
    dest = out_dir(run_dir, "psychometrics")
    results, lines = {}, [f"# Psychometrics of the turn-level metrics - {run_dir.name}", ""]
    for scale in args.scales:
        X, items = load_items(run_dir, scale)
        if X.shape[0] < 20:
            lines += [f"## {scale.upper()}", "", f"Only {X.shape[0]} rated turns: skipped.", ""]
            continue
        results[scale] = analyse(X, items, scale)
        lines += report_lines(scale, results[scale])
    lines += ["Turns are nested in sessions but treated as independent respondents; the judge is the only "
              "rater, so this is the judge's measurement structure, not people's. The factor models use "
              "Pearson correlations of skewed 1-7 ratings, which understates loadings relative to polychoric "
              "correlations."]
    write_json(dest / "summary.json", results)
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt = pyplot()
    if plt is not None and any("grm" in r for r in results.values()):
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5), squeeze=False)
        for ax, (scale, r) in zip(axes[0], results.items()):
            if "grm" not in r:
                continue
            g = r["grm"]
            for it in g["items"]:
                ax.plot(g["theta_grid"], it["info"], label=f"item {it['item']}")
            ax.plot(g["theta_grid"], g["test_info"], "k--", label="test")
            ax.set_title(f"{scale.upper()} item information", fontsize=9)
            ax.set_xlabel("theta (latent intrusiveness / reactance)")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(dest / "item_information.png", dpi=150)
        plt.close(fig)
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
