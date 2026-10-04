"""Time to disclosure: Kaplan-Meier, log-rank and a Cox model (CPU only; needs label_seeker.py first).

Outcome  T = the seeker turn of the first disclosure (label_seeker state `disclosed`). A session that ends
         without one is right-censored at its last labelled seeker turn. Success Rate asks *whether* the
         need surfaced; T asks *how fast*, and censoring keeps short sessions honest.

Kaplan-Meier  S(t) = prod_{t_j <= t} (1 - d_j / n_j), Greenwood variance, log(-log) 95 % band; median T and
              the restricted mean RMST(tau) = integral_0^tau S(u) du (turns spent before disclosure, capped).
Log-rank      k-group test, (O - E)' V^-1 (O - E) ~ chi-square(k - 1), by arm and by resistance level.
Cox           h(t | x) = h0(t) exp(beta' x(t)), Breslow ties, Newton-Raphson. Covariates: arm (vs a
              reference arm), resistance level (vs low), and a time-varying x(t) = share of supporter turns
              so far that were deep (L2/L3), per 10 percentage points. A hazard ratio below 1 on depth means
              deeper probing SLOWS disclosure - COCOON's central claim, tested turn by turn.
              Standard errors are cluster-robust by profile (Lin & Wei 1989): sessions of one profile share
              a person.

Depth uses grounding.classify_rung on the supporter text for every arm. The logged `ladder_rung` is the
rung the pipeline PERMITTED (the monolithic listener writes a fixed L2), so it is not comparable across arms.

  python extra/survival.py --run runs/v3_50 [--reference base_instruct] [--tau 10]
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from _shared import chi2_sf, cluster_bootstrap, fmt, load_arm_dialogues, markdown_table, norm_sf, out_dir, \
    pyplot, run_path, write_json
from grounding import classify_rung
from label_seeker import load_labels

RESISTANCE = ("low", "medium", "high")


# --------------------------------------------------------------------------- data


def build_subjects(labels: dict[str, list[dict]], dialogues: dict[str, list[dict]]) -> list[dict]:
    """One subject per session: time, event, arm, resistance, profile, and depth share before each turn."""
    subjects = []
    for arm, rows in labels.items():
        sessions = {s["session_id"]: s for s in dialogues.get(arm, [])}
        by: dict[str, list[dict]] = {}
        for r in rows:
            by.setdefault(r["session_id"], []).append(r)
        for sid, rs in by.items():
            rs = sorted(rs, key=lambda x: x["seeker_turn"])
            ok = []
            for r in rs:
                if r.get("parse_failed"):
                    break
                ok.append(r)
            if not ok or sid not in sessions:
                continue
            first = next((r for r in ok if r["state"] == "disclosed"), None)
            time = first["seeker_turn"] if first else ok[-1]["seeker_turn"]
            turns = sessions[sid]["turns"]
            depth = []
            for r in ok[:time]:
                sup = [t for t in turns if t["role"] == "supporter" and t["turn_index"] < r["turn_index"]
                       and t.get("meta", {}).get("source") != "opener_pool"]
                deep = sum(classify_rung(t.get("text", "")) in ("L2", "L3") for t in sup)
                depth.append(deep / len(sup) if sup else 0.0)
            profile = sessions[sid].get("profile_snapshot", {})
            subjects.append({"session_id": sid, "arm": arm, "time": int(time), "event": int(first is not None),
                             "resistance": profile.get("resistance_level", "medium"),
                             "profile": sessions[sid].get("profile_id", sid), "deep_share": depth})
    return subjects


# --------------------------------------------------------------------------- Kaplan-Meier, log-rank


def kaplan_meier(times, events) -> list[dict]:
    times, events = np.asarray(times), np.asarray(events)
    out, s, gw = [], 1.0, 0.0
    for t in np.unique(times[events == 1]):
        n = int((times >= t).sum())
        d = int(((times == t) & (events == 1)).sum())
        s *= 1 - d / n
        gw += d / (n * (n - d)) if n > d else math.inf
        if 0 < s < 1 and math.isfinite(gw):
            se = math.sqrt(gw) / abs(math.log(s))
            lo, hi = s ** math.exp(1.96 * se), s ** math.exp(-1.96 * se)
        else:
            lo = hi = s
        out.append({"t": int(t), "at_risk": n, "events": d, "S": s, "lo": lo, "hi": hi})
    return out


def km_median(curve: list[dict]) -> int | None:
    return next((p["t"] for p in curve if p["S"] <= 0.5), None)


def rmst(curve: list[dict], tau: int) -> float:
    """Area under the KM step function on [0, tau]: S = 1 until the first event time."""
    area, prev_t, s = 0.0, 0, 1.0
    for p in curve:
        if p["t"] >= tau:
            break
        area += s * (p["t"] - prev_t)
        prev_t, s = p["t"], p["S"]
    return area + s * (tau - prev_t)


def logrank(times, events, groups) -> dict:
    times, events, groups = np.asarray(times), np.asarray(events), np.asarray(groups)
    labels = sorted(set(groups.tolist()))
    k = len(labels)
    if k < 2:
        return {"chi2": None, "df": 0, "p": None}
    O, E, V = np.zeros(k), np.zeros(k), np.zeros((k, k))
    for t in np.unique(times[events == 1]):
        at = times >= t
        n = at.sum()
        d = ((times == t) & (events == 1)).sum()
        ng = np.array([(at & (groups == g)).sum() for g in labels], dtype=float)
        dg = np.array([((times == t) & (events == 1) & (groups == g)).sum() for g in labels], dtype=float)
        O += dg
        E += ng * d / n
        if n > 1:
            frac = ng / n
            V += d * (n - d) / (n - 1) * (np.diag(frac) - np.outer(frac, frac))
    diff = (O - E)[:-1]
    stat = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return {"chi2": stat, "df": k - 1, "p": chi2_sf(stat, k - 1), "groups": labels,
            "observed": O.tolist(), "expected": E.tolist()}


# --------------------------------------------------------------------------- Cox


def cox_design(subjects: list[dict], arms: list[str], reference: str) -> tuple[list[str], list[dict]]:
    """Counting-process rows (t-1, t], one per subject per seeker turn, with the covariates at turn t."""
    names = [f"arm={a}" for a in arms if a != reference] + ["resistance=medium", "resistance=high",
                                                             "deep_share_per_10pp"]
    rows = []
    for i, s in enumerate(subjects):
        fixed = [float(s["arm"] == a) for a in arms if a != reference]
        fixed += [float(s["resistance"] == "medium"), float(s["resistance"] == "high")]
        for t in range(1, s["time"] + 1):
            share = s["deep_share"][t - 1] if t - 1 < len(s["deep_share"]) else 0.0
            rows.append({"subject": i, "t": t, "event": int(s["event"] and t == s["time"]),
                         "x": fixed + [10.0 * share], "cluster": s["profile"]})
    return names, rows


def cox_fit(rows: list[dict], n_cov: int, max_iter: int = 50) -> dict:
    """Breslow partial likelihood, Newton-Raphson with step halving, model and cluster-robust variance."""
    X = np.array([r["x"] for r in rows], dtype=float).reshape(len(rows), n_cov)
    t = np.array([r["t"] for r in rows])
    ev = np.array([r["event"] for r in rows])
    event_times = np.unique(t[ev == 1])
    risk = {u: np.where(t == u)[0] for u in event_times}       # rows are (u-1, u]: at risk at u iff t == u

    def loglik_grad_hess(beta):
        ll, g, H = 0.0, np.zeros(n_cov), np.zeros((n_cov, n_cov))
        for u in event_times:
            idx = risk[u]
            eta = X[idx] @ beta
            m = eta.max()
            w = np.exp(eta - m)
            s0 = w.sum()
            xbar = (w @ X[idx]) / s0
            dmask = ev[idx] == 1
            d = int(dmask.sum())
            ll += eta[dmask].sum() - d * (m + math.log(s0))
            g += X[idx][dmask].sum(axis=0) - d * xbar
            xc = X[idx] - xbar
            H -= d * (xc.T * w) @ xc / s0
        return ll, g, H

    beta = np.zeros(n_cov)
    ll0, g, H = loglik_grad_hess(beta)
    ll = ll0
    for _ in range(max_iter):
        step = np.linalg.lstsq(-H, g, rcond=None)[0]
        new_beta, scale = beta + step, 1.0
        new_ll = loglik_grad_hess(new_beta)[0]
        while new_ll < ll - 1e-9 and scale > 1e-4:
            scale /= 2
            new_beta = beta + scale * step
            new_ll = loglik_grad_hess(new_beta)[0]
        converged = abs(new_ll - ll) < 1e-8
        beta, ll = new_beta, new_ll
        _, g, H = loglik_grad_hess(beta)
        if converged:
            break
    info_inv = np.linalg.pinv(-H)

    # Score residuals per row (Breslow), summed by cluster: V_robust = I^-1 (sum_c U_c U_c') I^-1.
    U = np.zeros_like(X)
    for u in event_times:
        idx = risk[u]
        w = np.exp(X[idx] @ beta)
        s0 = w.sum()
        xbar = (w @ X[idx]) / s0
        d = int((ev[idx] == 1).sum())
        U[idx] += (ev[idx][:, None] == 1) * (X[idx] - xbar) - d * (w / s0)[:, None] * (X[idx] - xbar)
    clusters: dict = {}
    for r, u_row in zip(rows, U):
        clusters[r["cluster"]] = clusters.get(r["cluster"], 0) + u_row
    meat = sum(np.outer(v, v) for v in clusters.values()) if clusters else np.zeros((n_cov, n_cov))
    robust = info_inv @ meat @ info_inv
    return {"beta": beta, "se_model": np.sqrt(np.clip(np.diag(info_inv), 0, None)),
            "se_robust": np.sqrt(np.clip(np.diag(robust), 0, None)), "loglik": ll, "loglik_null": ll0,
            "n_events": int(ev.sum()), "n_rows": len(rows), "n_clusters": len(clusters)}


def cox_table(names: list[str], fit: dict) -> list[dict]:
    out = []
    for name, b, se in zip(names, fit["beta"], fit["se_robust"]):
        z = b / se if se > 0 else float("nan")
        out.append({"covariate": name, "beta": float(b), "hazard_ratio": math.exp(b),
                    "hr_ci_low": math.exp(b - 1.96 * se), "hr_ci_high": math.exp(b + 1.96 * se),
                    "se_robust": float(se), "p": 2 * norm_sf(abs(z)) if math.isfinite(z) else None})
    return out


# --------------------------------------------------------------------------- report


def plot_km(curves: dict[str, list[dict]], path, title: str) -> bool:
    plt = pyplot()
    if plt is None or not curves:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, curve in curves.items():
        ts = [0] + [p["t"] for p in curve]
        ss = [1.0] + [p["S"] for p in curve]
        ax.step(ts, ss, where="post", label=name)
    ax.set_xlabel("seeker turn")
    ax.set_ylabel("P(need not yet disclosed)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Survival analysis of time to disclosure.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--reference", default="base_instruct", help="reference arm for the Cox model")
    ap.add_argument("--tau", type=int, default=10, help="RMST horizon in seeker turns")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    run_dir = run_path(args.run)
    labels = load_labels(run_dir, args.arms)
    if not labels:
        raise SystemExit("no seeker-state labels: run extra/label_seeker.py first")
    subjects = build_subjects(labels, load_arm_dialogues(run_dir, list(labels)))
    arms = sorted({s["arm"] for s in subjects})
    reference = args.reference if args.reference in arms else arms[0]

    per_arm = {}
    curves = {}
    for arm in arms:
        sub = [s for s in subjects if s["arm"] == arm]
        curve = kaplan_meier([s["time"] for s in sub], [s["event"] for s in sub])
        curves[arm] = curve
        by_profile: dict = {}
        for s in sub:
            by_profile.setdefault(s["profile"], []).append(s)

        def rmst_stat(groups, tau=args.tau):
            flat = [s for g in groups for s in g]
            return rmst(kaplan_meier([s["time"] for s in flat], [s["event"] for s in flat]), tau)

        per_arm[arm] = {"sessions": len(sub), "events": sum(s["event"] for s in sub),
                        "median_turn": km_median(curve), "rmst": rmst(curve, args.tau),
                        "rmst_ci": cluster_bootstrap(list(by_profile.values()), rmst_stat, args.reps,
                                                     f"survival:{arm}"), "curve": curve}
    res_curves = {lvl: kaplan_meier([s["time"] for s in subjects if s["resistance"] == lvl],
                                    [s["event"] for s in subjects if s["resistance"] == lvl])
                  for lvl in RESISTANCE if any(s["resistance"] == lvl for s in subjects)}
    tests = {
        "logrank_arms": logrank([s["time"] for s in subjects], [s["event"] for s in subjects],
                                [s["arm"] for s in subjects]),
        "logrank_resistance": logrank([s["time"] for s in subjects], [s["event"] for s in subjects],
                                      [s["resistance"] for s in subjects]),
    }
    names, rows = cox_design(subjects, arms, reference)
    cox = None
    if sum(r["event"] for r in rows) >= len(names) + 1:
        fit = cox_fit(rows, len(names))
        lr = 2 * (fit["loglik"] - fit["loglik_null"])
        cox = {"reference_arm": reference, "table": cox_table(names, fit), "n_events": fit["n_events"],
               "n_clusters": fit["n_clusters"], "lr_vs_null": lr, "lr_df": len(names),
               "lr_p": chi2_sf(lr, len(names))}

    dest = out_dir(run_dir, "survival")
    write_json(dest / "summary.json", {"tau": args.tau, "arms": per_arm, "tests": tests, "cox": cox,
                                       "resistance_curves": res_curves})
    lines = [f"# Time to disclosure - {run_dir.name}", "",
             "T = seeker turn of the first disclosure; sessions without one are censored at their last turn. "
             f"RMST = mean turns before disclosure, capped at tau = {args.tau} (95 % CI: bootstrap over "
             "profiles).", "",
             markdown_table([{"arm": a, "sessions": r["sessions"], "disclosed": r["events"],
                              "median turn": r["median_turn"] if r["median_turn"] is not None else "not reached",
                              f"RMST({args.tau})": f"{fmt(r['rmst'], 2)} [{fmt(r['rmst_ci'][0], 2)}, "
                                                   f"{fmt(r['rmst_ci'][1], 2)}]"}
                             for a, r in per_arm.items()],
                            ["arm", "sessions", "disclosed", "median turn", f"RMST({args.tau})"]), ""]
    for key, label in (("logrank_arms", "arms"), ("logrank_resistance", "resistance levels")):
        t = tests[key]
        if t["chi2"] is not None:
            lines.append(f"- Log-rank across {label}: chi2 = {t['chi2']:.2f}, df = {t['df']}, p = {t['p']:.4f}")
    if cox:
        lines += ["", f"## Cox model (reference arm `{reference}`, resistance vs low; robust SE by profile)", "",
                  markdown_table([{"covariate": r["covariate"], "HR": fmt(r["hazard_ratio"]),
                                   "95 % CI": f"[{fmt(r['hr_ci_low'])}, {fmt(r['hr_ci_high'])}]",
                                   "p": fmt(r["p"], 4)} for r in cox["table"]],
                                 ["covariate", "HR", "95 % CI", "p"]), "",
                  f"Events {cox['n_events']}, profiles {cox['n_clusters']}. LR test vs the null model: "
                  f"{cox['lr_vs_null']:.2f} on {cox['lr_df']} df, p = {cox['lr_p']:.4f}. HR > 1 = faster "
                  "disclosure. Ties handled by Breslow's approximation."]
    else:
        lines += ["", "Cox model not fitted: fewer events than covariates."]
    lines += ["", "Disclosure is judged on a simulated seeker; transfer to people is out of scope."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_km(curves, dest / "km_arms.png", "Kaplan-Meier: time to disclosure by arm")
    plot_km(res_curves, dest / "km_resistance.png", "Kaplan-Meier: time to disclosure by resistance level")
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
