"""Disclosure as an absorbing Markov chain over seeker states (CPU only; needs label_seeker.py first).

Each session is the sequence of seeker states, one per seeker turn, cut at the first disclosure:
transient {guarded, opening, withdrawn}, absorbing {disclosed}. Per arm:

  P      transition matrix, posterior mean under a Jeffreys Dirichlet(1/2) prior per row (the MLE breaks on
         a row with no data), absorbing row fixed at e_disclosed; pi0 = first-turn state shares (MLE).
         A row with few observed transitions is mostly prior: the report lists them.
  Q      transient-to-transient block;  N = (I - Q)^-1  the fundamental matrix: N[i, j] = expected visits
         to j starting from i
  E[T]   expected seeker turn of the first disclosure, 1 + pi0_T . N 1  (model extrapolation: the chain is
         allowed to run past the end of a session)
  P(T <= H)  disclosure within the horizon H a session actually lasts: (pi0 P^(H-1))[disclosed]
  W      expected withdrawn turns before disclosure, pi0_T . N[:, withdrawn]

Tests (likelihood ratio, chi-square reference; positive cells only, so df counts only observed cells):
  homogeneity  one chain for all arms vs one per arm (Anderson & Goodman 1957)
  order        first-order vs second-order chain on the pooled data: is the Markov assumption tenable?
CIs: percentile bootstrap over profiles, the same resampling unit as report.py.

  python extra/markov.py --run runs/v3_50 [--horizon 10] [--reps 2000]
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter

import numpy as np

from _shared import chi2_sf, cluster_bootstrap, fmt, markdown_table, out_dir, pyplot, run_path, write_json
from label_seeker import STATES, load_labels, sequences

TRANSIENT = STATES[:3]
ABSORB = STATES.index("disclosed")
K = len(STATES)
PRIOR = 0.5


def truncate(seq: list[str]) -> list[str]:
    """Up to and including the first disclosure: after absorption the chain has nothing left to say."""
    return seq[: seq.index("disclosed") + 1] if "disclosed" in seq else list(seq)


def transition_counts(seqs: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    C = np.zeros((K, K))
    init = np.zeros(K)
    for seq in seqs:
        seq = truncate(seq)
        if not seq:
            continue
        init[STATES.index(seq[0])] += 1
        for a, b in zip(seq, seq[1:]):
            C[STATES.index(a), STATES.index(b)] += 1
    return C, init


def estimate(C: np.ndarray, init: np.ndarray, prior: float = PRIOR) -> tuple[np.ndarray, np.ndarray]:
    P = np.zeros((K, K))
    for i in range(K):
        if i == ABSORB:
            P[i, i] = 1.0
        else:
            P[i] = (C[i] + prior) / (C[i].sum() + K * prior)
    pi0 = init / init.sum() if init.sum() else np.full(K, 1.0 / K)   # one draw per session: MLE is fine
    return P, pi0


def absorbing_summary(P: np.ndarray, pi0: np.ndarray, horizon: int) -> dict:
    t_idx = [STATES.index(s) for s in TRANSIENT]
    Q = P[np.ix_(t_idx, t_idx)]
    N = np.linalg.inv(np.eye(len(t_idx)) - Q)
    steps = N.sum(axis=1)
    reach = pi0 @ np.linalg.matrix_power(P, max(0, horizon - 1))
    w = TRANSIENT.index("withdrawn")
    return {
        "p_disclosed_within_horizon": float(reach[ABSORB]),
        "expected_turn_of_disclosure": float(1.0 + pi0[t_idx] @ steps),
        "expected_withdrawn_turns": float(pi0[t_idx] @ N[:, w]),
        "fundamental_matrix": N.round(4).tolist(),
        "expected_steps_from": {s: float(v) for s, v in zip(TRANSIENT, steps)},
    }


def _g2(table: dict[tuple, Counter]) -> tuple[float, int]:
    """G^2 = 2 sum n ln(n / expected) for 'is the next state independent of the row group?', summed over
    strata. table: {(stratum, group): Counter(next_state)}. df = sum over strata of (groups-1)(dests-1),
    counting only groups and destinations that were observed in that stratum."""
    strata: dict = {}
    for (stratum, group), cnt in table.items():
        strata.setdefault(stratum, {})[group] = cnt
    g2, df = 0.0, 0
    for groups in strata.values():
        groups = {g: c for g, c in groups.items() if sum(c.values()) > 0}
        if len(groups) < 2:
            continue
        pooled = Counter()
        for c in groups.values():
            pooled.update(c)
        total = sum(pooled.values())
        dests = [d for d, n in pooled.items() if n > 0]
        for c in groups.values():
            n_g = sum(c.values())
            for d in dests:
                if c[d] > 0:
                    g2 += 2 * c[d] * math.log(c[d] / (n_g * pooled[d] / total))
        df += (len(groups) - 1) * (len(dests) - 1)
    return g2, df


def homogeneity_test(seqs_by_arm: dict[str, list[list[str]]]) -> dict:
    table: dict[tuple, Counter] = {}
    for arm, seqs in seqs_by_arm.items():
        for seq in seqs:
            seq = truncate(seq)
            for a, b in zip(seq, seq[1:]):
                table.setdefault((a, arm), Counter())[b] += 1
    g2, df = _g2(table)
    return {"G2": g2, "df": df, "p": chi2_sf(g2, df)}


def order_test(seqs: list[list[str]]) -> dict:
    """Second order: does the state before the current one change where the chain goes next?"""
    table: dict[tuple, Counter] = {}
    for seq in seqs:
        seq = truncate(seq)
        for h, a, b in zip(seq, seq[1:], seq[2:]):
            table.setdefault((a, h), Counter())[b] += 1
    g2, df = _g2(table)
    return {"G2": g2, "df": df, "p": chi2_sf(g2, df)}


def analyse_arm(seq_by_profile: dict[str, list[list[str]]], horizon: int, reps: int, tag: str) -> dict:
    seqs = [s for group in seq_by_profile.values() for s in group]
    C, init = transition_counts(seqs)
    P, pi0 = estimate(C, init)
    out = absorbing_summary(P, pi0, horizon)
    out.update({"sessions": len(seqs), "transitions": int(C.sum()), "P": P.round(4).tolist(),
                "sparse_rows": [s for s in TRANSIENT if C[STATES.index(s)].sum() < 5],
                "pi0": pi0.round(4).tolist(), "counts": C.astype(int).tolist(),
                "observed_disclosure_rate": statistics.fmean("disclosed" in s for s in seqs) if seqs else None})

    def stat(key: str):
        def f(groups: list[list[list[str]]]) -> float:
            Cb, ib = transition_counts([s for g in groups for s in g])
            return absorbing_summary(*estimate(Cb, ib), horizon)[key]
        return f

    clusters = list(seq_by_profile.values())
    for key in ("p_disclosed_within_horizon", "expected_turn_of_disclosure", "expected_withdrawn_turns"):
        out[key + "_ci"] = cluster_bootstrap(clusters, stat(key), reps, f"{tag}:{key}")
    return out


def plot_matrices(results: dict, path) -> bool:
    plt = pyplot()
    if plt is None or not results:
        return False
    arms = list(results)
    fig, axes = plt.subplots(1, len(arms), figsize=(3.2 * len(arms), 3.2), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        P = np.array(results[arm]["P"])
        ax.imshow(P, vmin=0, vmax=1, cmap="Blues")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(K), [s[:4] for s in STATES], fontsize=7)
        ax.set_yticks(range(K), [s[:4] for s in STATES], fontsize=7)
        ax.set_title(arm, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Absorbing Markov chain analysis of seeker states.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--horizon", type=int, default=None, help="default: median seeker turns per session")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    run_dir = run_path(args.run)
    labels = load_labels(run_dir, args.arms)
    if not labels:
        raise SystemExit("no seeker-state labels: run extra/label_seeker.py first")
    by_arm: dict[str, dict[str, list[list[str]]]] = {}
    lengths = []
    for arm, rows in labels.items():
        profile_of = {r["session_id"]: r.get("profile_id") or r["session_id"] for r in rows}
        groups: dict[str, list[list[str]]] = {}
        for sid, seq in sequences(rows).items():
            groups.setdefault(profile_of[sid], []).append(seq)
            lengths.append(len(seq))
        by_arm[arm] = groups
    horizon = args.horizon or (int(statistics.median(lengths)) if lengths else 10)

    results = {arm: analyse_arm(groups, horizon, args.reps, f"markov:{arm}") for arm, groups in by_arm.items()}
    flat = {arm: [s for g in groups.values() for s in g] for arm, groups in by_arm.items()}
    tests = {"homogeneity_across_arms": homogeneity_test(flat),
             "first_vs_second_order_pooled": order_test([s for seqs in flat.values() for s in seqs])}

    dest = out_dir(run_dir, "markov")
    write_json(dest / "summary.json", {"states": STATES, "horizon": horizon, "prior": PRIOR,
                                       "arms": results, "tests": tests})
    ci = lambda r, k: f"{fmt(r[k])} [{fmt(r[k + '_ci'][0])}, {fmt(r[k + '_ci'][1])}]"
    rows = [{"arm": a, "sessions": r["sessions"], "observed disclosure": fmt(r["observed_disclosure_rate"]),
             f"P(disclosed by turn {horizon})": ci(r, "p_disclosed_within_horizon"),
             "E[turn of disclosure]": ci(r, "expected_turn_of_disclosure"),
             "E[withdrawn turns]": ci(r, "expected_withdrawn_turns")} for a, r in results.items()]
    lines = [f"# Disclosure as an absorbing Markov chain - {run_dir.name}", "",
             f"States {', '.join(STATES)}; `disclosed` absorbs. Horizon H = {horizon} seeker turns. "
             f"Jeffreys prior {PRIOR} per transition cell. 95 % CIs: bootstrap over profiles.", "",
             markdown_table(rows, list(rows[0])) if rows else "(no data)", ""]
    h, o = tests["homogeneity_across_arms"], tests["first_vs_second_order_pooled"]
    lines += [f"- Same chain for every arm? G2 = {h['G2']:.2f}, df = {h['df']}, p = {h['p']:.4f}",
              f"- First-order Markov enough (vs second order)? G2 = {o['G2']:.2f}, df = {o['df']}, "
              f"p = {o['p']:.4f} (a small p means the previous state matters too)", ""]
    for arm, r in results.items():
        lines += [f"## {arm}", "", markdown_table(
            [{"from": s, **{t: f"{p:.2f}" for t, p in zip(STATES, row)}} for s, row in zip(STATES, r["P"])],
            ["from", *STATES]), ""]
        if r["sparse_rows"]:
            lines += [f"Rows with fewer than 5 observed transitions (mostly prior): {', '.join(r['sparse_rows'])}.", ""]
    lines += ["Seeker states come from a judge reading a simulated seeker; transfer to people is out of scope."]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if plot_matrices(results, dest / "transitions.png"):
        print(f"wrote {dest / 'transitions.png'}")
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
