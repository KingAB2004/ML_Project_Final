"""Checks for the extra analyses. Plain asserts, same style as tests/test_all.py; echo backend, no GPU.

Each estimator is checked against a case with a known answer (hand-computed, or simulated from known
parameters), and each model pass is run once against the echo backend to prove the wiring.

  python extra/test_extra.py            (all checks)
  python extra/test_extra.py survival   (only checks whose name contains 'survival')
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _shared                                   # noqa: E402  (puts src/ on the path)
import eig_probe                                 # noqa: E402
import label_seeker                              # noqa: E402
import lora_geometry as lg                       # noqa: E402
import markov                                    # noqa: E402
import memory_calibration as memcal              # noqa: E402
import need_sets                                 # noqa: E402
import psychometrics as psy                      # noqa: E402
import pvi                                       # noqa: E402
import scaling                                   # noqa: E402
import survival                                  # noqa: E402
import llm as llm_mod                            # noqa: E402
from judge import Judge                          # noqa: E402

CHECKS: list = []
TMP = Path(tempfile.mkdtemp(prefix="coccon_extra_tests_"))


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def echo(role: str):
    return llm_mod.LLM(role, backend="echo", cache_dir=TMP / "cache")


def toy_session(sid: str = "p000001-armX-s1", pid: str = "p000001", need: str = "I need to feel valued.",
                ptype: str = "job crisis", n_pairs: int = 4) -> dict:
    turns = [{"turn_index": 0, "role": "supporter", "text": "Hello, I have time if you want to talk.",
              "meta": {"source": "opener_pool"}}]
    for k in range(n_pairs):
        turns.append({"turn_index": 2 * k + 1, "role": "user", "text": f"Things are fine, I guess ({k})."})
        turns.append({"turn_index": 2 * k + 2, "role": "supporter",
                      "text": "It sounds like you might feel unseen?" if k % 2 else "Tell me more?"})
    turns.append({"turn_index": 2 * n_pairs + 1, "role": "user", "text": "Maybe."})
    return {"session_id": sid, "profile_id": pid, "turns": turns,
            "profile_snapshot": {"terminal_need": need, "feeling": "tired", "problem_type": ptype,
                                 "resistance_level": "medium",
                                 "need_chain": [{"depth": 0, "text": "work is a lot"},
                                                {"depth": 1, "text": "I need rest"},
                                                {"depth": 2, "text": need}]}}


# --------------------------------------------------------------------------- shared


@check("shared.chi2_sf_matches_tables")
def _():
    for x, df in ((3.841459, 1), (5.991465, 2), (18.307038, 10), (124.342, 100)):
        assert abs(_shared.chi2_sf(x, df) - 0.05) < 1e-4, (x, df, _shared.chi2_sf(x, df))
    assert abs(_shared.chi2_sf(0.0158, 1) - 0.9) < 1e-3
    assert abs(_shared.norm_sf(1.959964) - 0.025) < 1e-6


# --------------------------------------------------------------------------- seeker states, markov


@check("label_seeker.state_rule_and_sequences")
def _():
    s = label_seeker.state_of
    assert s({"disclosed": 6, "opening": 1, "guarded": 1, "pulling_back": 7}) == "disclosed"
    assert s({"disclosed": 2, "opening": 5, "guarded": 1, "pulling_back": 5}) == "withdrawn"
    assert s({"disclosed": 2, "opening": 6, "guarded": 1, "pulling_back": 5}) == "opening"
    assert s({"disclosed": 2, "opening": 2, "guarded": 6, "pulling_back": 2}) == "guarded"
    rows = [{"session_id": "a", "seeker_turn": 1, "state": "guarded"},
            {"session_id": "a", "seeker_turn": 3, "state": "opening"},
            {"session_id": "a", "seeker_turn": 2, "parse_failed": True}]
    assert label_seeker.sequences(rows) == {"a": ["guarded"]}, "a parse failure must end the sequence"


@check("label_seeker.echo_pass_labels_every_seeker_turn")
def _():
    session = toy_session()
    rows = label_seeker.label_session(echo("judge"), _shared.read_extra_prompt("seeker_state.md"), session)
    assert len(rows) == 5 and all(not r["parse_failed"] for r in rows)
    assert {r["state"] for r in rows} == {"guarded"}, "echo rates every item 2"


def simulate_chain(P, pi0, n, length, seed):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        s = rng.choice(4, p=pi0)
        seq = [markov.STATES[s]]
        while len(seq) < length and s != markov.ABSORB:
            s = rng.choice(4, p=P[s])
            seq.append(markov.STATES[s])
        seqs.append(seq)
    return seqs


P_TRUE = np.array([[0.6, 0.25, 0.1, 0.05], [0.2, 0.5, 0.1, 0.2], [0.4, 0.1, 0.45, 0.05], [0, 0, 0, 1.0]])
PI0 = np.array([0.7, 0.2, 0.1, 0.0])


@check("markov.estimator_recovers_the_transition_matrix")
def _():
    C, init = markov.transition_counts(simulate_chain(P_TRUE, PI0, 4000, 12, 1))
    P, _ = markov.estimate(C, init)
    assert np.abs(P - P_TRUE).max() < 0.03, np.abs(P - P_TRUE).max()


@check("markov.absorbing_formulas_match_simulation")
def _():
    H = 8
    out = markov.absorbing_summary(P_TRUE, PI0, H)
    seqs = simulate_chain(P_TRUE, PI0, 20000, 500, 2)
    first = [s.index("disclosed") + 1 if "disclosed" in s else math.inf for s in seqs]
    assert abs(out["p_disclosed_within_horizon"] - np.mean([t <= H for t in first])) < 0.015
    assert abs(out["expected_turn_of_disclosure"] - np.mean(first)) / np.mean(first) < 0.03
    withdrawn = np.mean([s.count("withdrawn") for s in seqs])
    assert abs(out["expected_withdrawn_turns"] - withdrawn) < 0.1


@check("markov.homogeneity_test_detects_a_different_chain")
def _():
    same = markov.homogeneity_test({"a": simulate_chain(P_TRUE, PI0, 400, 12, 3),
                                    "b": simulate_chain(P_TRUE, PI0, 400, 12, 4)})
    P2 = P_TRUE.copy()
    P2[0] = [0.2, 0.2, 0.1, 0.5]
    diff = markov.homogeneity_test({"a": simulate_chain(P_TRUE, PI0, 400, 12, 5),
                                    "b": simulate_chain(P2, PI0, 400, 12, 6)})
    assert same["p"] > 0.01 and diff["p"] < 1e-6, (same, diff)
    first_order = markov.order_test(simulate_chain(P_TRUE, PI0, 3000, 12, 7))
    assert first_order["p"] > 0.001, "data simulated from a first-order chain must not reject first order"


# --------------------------------------------------------------------------- survival


@check("survival.kaplan_meier_and_rmst_by_hand")
def _():
    curve = survival.kaplan_meier([1, 2, 2, 3, 4], [1, 1, 0, 1, 0])
    assert [round(p["S"], 6) for p in curve] == [0.8, 0.6, 0.3]
    assert abs(survival.rmst(curve, 4) - 2.7) < 1e-9
    assert survival.km_median(curve) == 3


@check("survival.logrank_null_and_alternative")
def _():
    t = [1, 2, 3, 4, 5, 6] * 20
    e = [1, 1, 0, 1, 1, 0] * 20
    null = survival.logrank(t + t, e + e, ["a"] * len(t) + ["b"] * len(t))
    assert null["chi2"] < 1e-9
    fast = [1, 1, 2, 2, 3, 1] * 20
    alt = survival.logrank(t + fast, e + [1] * len(fast), ["a"] * len(t) + ["b"] * len(fast))
    assert alt["p"] < 1e-6


@check("survival.cox_recovers_a_known_hazard_ratio")
def _():
    rng = np.random.default_rng(8)
    beta, rows = math.log(2.0), []
    for i in range(3000):
        x = float(rng.random() < 0.5)
        for t in range(1, 21):
            event = rng.random() < 1 - math.exp(-0.05 * math.exp(beta * x))
            rows.append({"subject": i, "t": t, "event": int(event), "x": [x], "cluster": i})
            if event:
                break
    fit = survival.cox_fit(rows, 1)
    assert abs(fit["beta"][0] - beta) < 0.12, fit["beta"]
    assert 0.5 < fit["se_robust"][0] / fit["se_model"][0] < 2.0


@check("survival.subjects_use_the_mechanical_rung_not_the_logged_cap")
def _():
    session = toy_session()
    labels = {"armX": label_seeker.label_session(echo("judge"), _shared.read_extra_prompt("seeker_state.md"),
                                                 session)}
    subj = survival.build_subjects(labels, {"armX": [session]})
    assert len(subj) == 1 and subj[0]["event"] == 0 and subj[0]["time"] == 5
    assert subj[0]["deep_share"][0] == 0.0, "only the opener precedes the first seeker turn, and it is excluded"


# --------------------------------------------------------------------------- need sets, EIG

WORDS = ("rest", "safety", "belonging", "respect", "control", "trust", "freedom", "comfort", "purpose",
         "recognition", "closeness", "fairness")


@check("need_sets.candidates_hold_the_truth_once_and_no_paraphrase")
def _():
    pool = [{"profile_id": f"p{i:06d}", "text": t, "problem_type": "job crisis" if i % 2 else "other"}
            for i, t in enumerate(["I need to feel valued.", "I need to feel valued at work.", "I want rest.",
                                   "I need safety.", "I need to belong.", "I need control.", "I want respect.",
                                   "I need to be heard.", "I want my family to trust me.", "I need a break."])]
    s = toy_session(pid="p999999")
    cands, truth = need_sets.candidate_list(s, pool, 6)
    assert len(cands) == 6 and cands[truth] == "I need to feel valued." and cands.count(cands[truth]) == 1
    assert "I need to feel valued at work." not in cands, "a near-paraphrase of the truth is not a distractor"
    assert need_sets.candidate_list(s, pool, 6) == (cands, truth), "deterministic per session"


@check("need_sets.temperature_mle_and_conformal_coverage")
def _():
    rng = np.random.default_rng(9)
    beta_true, K = 0.8, 6

    def draw(n):
        R = rng.integers(1, 8, size=(n, K)).astype(float)
        Y = [int(rng.choice(K, p=need_sets.softmax(r, beta_true))) for r in R]
        return R, Y

    R, Y = draw(3000)
    beta = need_sets.fit_temperature(R.tolist(), Y)
    assert abs(beta - beta_true) < 0.08, beta
    cal = [{"session_id": f"c{i}", "position": 1.0, "ratings": r.tolist(), "truth": y, "parse_failed": False,
            "candidates": list(range(K))} for i, (r, y) in enumerate(zip(*draw(1000)))]
    test = [{"session_id": f"t{i}", "position": 1.0, "ratings": r.tolist(), "truth": y, "parse_failed": False,
             "candidates": list(range(K))} for i, (r, y) in enumerate(zip(*draw(4000)))]
    out = need_sets.evaluate_position(cal, test)
    for key in ("LAC@0.1", "APS@0.1"):
        assert 0.875 <= out["sets"][key]["coverage"] <= 0.94, (key, out["sets"][key])
    assert out["sets"]["LAC@0.2"]["mean_size"] < out["sets"]["LAC@0.1"]["mean_size"]


@check("need_sets.echo_pass_scores_every_position")
def _():
    pool = [{"profile_id": f"p{i:06d}", "text": f"I need {w} in my life.", "problem_type": "x"}
            for i, w in enumerate(WORDS)]
    rows = need_sets.build_and_score(echo("analyzer"), [toy_session()], pool, 5, "test")
    assert len(rows) == len(need_sets.POSITIONS) and all(r["ratings"] == [2] * 5 for r in rows)


@check("eig_probe.entropy_choice_monotone_and_echo_wiring")
def _():
    assert abs(eig_probe.entropy_bits(np.full(8, 1 / 8)) - 3.0) < 1e-12
    stats = [{"L0": {"eig": 0.1, "pri": 0.1, "oracle_gain": 0}, "L1": {"eig": 0.3, "pri": 0.15, "oracle_gain": 0},
              "L2": {"eig": 0.6, "pri": 0.3, "oracle_gain": 0}, "L3": {"eig": 0.9, "pri": 0.6, "oracle_gain": 0}}] * 3
    pri = [eig_probe.policy_value(stats, eig_probe.choose(stats, lam))["pri"] for lam in np.linspace(0, 10, 50)]
    assert all(a >= b - 1e-12 for a, b in zip(pri, pri[1:])), "reactance of the choice cannot rise with lambda"
    pool = [{"profile_id": f"p{i:06d}", "text": f"I need {w} in my life.", "problem_type": "x"}
            for i, w in enumerate(WORDS)]
    gen = echo("generator")
    session = toy_session()
    decisions = eig_probe.generate_pass(gen, gen.view("simulator"), gen.view("analyzer"), [session], pool, 5,
                                        1.0, 2)
    gen.release()
    judge = Judge(llm=echo("judge"))
    eig_probe.judge_pass(judge, decisions, {session["session_id"]: session})
    judge.llm.release()
    st = eig_probe.move_stats(decisions[0])
    assert set(st) == {"original", *eig_probe.RUNG_ORDER}
    assert all(abs(v["eig"]) < 1e-9 for v in st.values()), "echo beliefs are uniform: no information"


# --------------------------------------------------------------------------- psychometrics


@check("psychometrics.alpha_by_hand_and_collapse")
def _():
    X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 4, 6], [5, 6, 7]], dtype=float)
    k, item_var, total_var = 3, X.var(axis=0, ddof=1).sum(), X.sum(axis=1).var(ddof=1)
    assert abs(psy.cronbach_alpha(X) - k / (k - 1) * (1 - item_var / total_var)) < 1e-12
    col = np.array([1] * 50 + [2] * 30 + [3] * 3 + [7] * 2)
    out = psy.collapse(col)
    assert out is not None and int(out.max()) == 1, "categories 3 and 7 are too sparse and merge downward"
    assert psy.collapse(np.ones(40)) is None


@check("psychometrics.factor_analysis_finds_two_factors")
def _():
    rng = np.random.default_rng(10)
    n = 2000
    F = rng.standard_normal((n, 2))
    L = np.array([[0.8, 0], [0.75, 0], [0.7, 0], [0, 0.8], [0, 0.75], [0, 0.7]])
    X = F @ L.T + rng.standard_normal((n, 6)) * np.sqrt(1 - (L ** 2).sum(axis=1))
    assert psy.parallel_analysis(X, reps=50)["factors_retained"] == 2
    R = np.corrcoef(X, rowvar=False)
    one, two = psy.fa_em(R, n, 1), psy.fa_em(R, n, 2)
    assert two["bic"] < one["bic"] and two["p"] > 0.001
    groups = np.abs(np.array(two["loadings"])).argmax(axis=1)
    assert len(set(groups[:3])) == 1 and len(set(groups[3:])) == 1 and groups[0] != groups[3]


@check("psychometrics.grm_recovers_discrimination_order")
def _():
    rng = np.random.default_rng(11)
    n, a_true = 1500, [0.5, 1.0, 1.8, 2.6]
    theta = rng.standard_normal(n)
    Y = []
    for a in a_true:
        b = np.array([-1.0, 0.0, 1.0])
        star = 1 / (1 + np.exp(-a * (theta[:, None] - b[None, :])))
        Y.append((rng.random((n, 1)) < star).sum(axis=1))
    fit = psy.fit_grm(Y, cycles=60)
    a_hat = [it["a"] for it in fit["items"]]
    assert a_hat == sorted(a_hat), a_hat
    assert all(abs(h - t) / t < 0.3 for h, t in zip(a_hat, a_true)), a_hat


# --------------------------------------------------------------------------- memory calibration


@check("memory_calibration.scoring_rules")
def _():
    p = np.array([0.1] * 50 + [0.9] * 50)
    y = np.array([0] * 45 + [1] * 5 + [1] * 45 + [0] * 5, dtype=float)
    d = memcal.brier_decomposition(p, y)
    assert abs(d["reliability"] - d["resolution"] + d["uncertainty"] - d["brier"]) < 1e-12
    assert abs(d["within_bin"]) < 1e-12 and d["ece"] < 1e-12
    assert memcal.auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0


@check("memory_calibration.irls_recovers_logistic_coefficients")
def _():
    rng = np.random.default_rng(12)
    X = np.column_stack([np.ones(5000), rng.standard_normal((5000, 2))])
    w_true = np.array([-0.5, 1.2, -0.8])
    y = (rng.random(5000) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
    assert np.abs(memcal.logistic_irls(X, y, l2=0.0) - w_true).max() < 0.12


@check("memory_calibration.echo_pass_labels_nodes")
def _():
    nodes = [{"text": "needs rest", "reference": ["I need rest", "I need to feel valued."]},
             {"text": "wants a car", "reference": ["I need rest"]}]
    memcal.label_nodes(echo("judge"), nodes)
    assert [n["real"] for n in nodes] == [0, 0], "echo rates 2/7: no match"


# --------------------------------------------------------------------------- scaling study (CPU parts)


@check("lora_geometry.lowrank_svd_matches_dense")
def _():
    rng = np.random.default_rng(3)
    A, B = rng.standard_normal((8, 50)), rng.standard_normal((40, 8))
    u, sig, v = lg.svd_lowrank(A, B, 2.0)
    dense = np.linalg.svd(2.0 * B @ A, compute_uv=False)[:8]
    assert np.allclose(sig, dense), (sig, dense)
    assert np.allclose(u @ np.diag(sig) @ v.T, 2.0 * B @ A)


@check("lora_geometry.spectrum_and_similarity_known_cases")
def _():
    st = lg.spectrum_stats(np.array([1.0, 1.0, 1.0, 1.0, 0, 0]))
    assert abs(st["effective_rank"] - 4) < 1e-9 and abs(st["stable_rank"] - 4) < 1e-9
    assert abs(lg.spectrum_stats(np.array([5.0, 0, 0]))["effective_rank"] - 1) < 1e-9
    q = np.linalg.qr(np.random.default_rng(0).standard_normal((30, 6)))[0]
    assert abs(lg.subspace_similarity(q, q, 4) - 1) < 1e-9
    assert lg.subspace_similarity(q[:, :3], q[:, 3:], 3) < 1e-9          # orthogonal subspaces
    curve = {0: 2.0, 1: 1.6, 2: 1.45, 4: 1.41, 8: 1.4, 32: 1.4}
    assert lg.rank_needed(curve) == 4                                     # 0.59 of 0.60 drop at k=4


@check("scaling.power_law_recovers_exponent")
def _():
    n = [0.36e9, 2.77e9, 6.53e9]
    fit = scaling.power_law(n, [5.0 * x ** -0.07 for x in n])
    assert abs(fit["b"] - 0.07) < 1e-9 and abs(fit["a"] - 5.0) < 1e-6 and fit["residual_df"] == 1


@check("pvi.summary_on_synthetic_turns")
def _():
    rows = [{"profile_id": f"p{i % 5}", "skipped": False, "n_tokens": 10, "thought_chars": 100 + i,
             "ladder_rung": "L1", "logp_with": -10.0, "logp_wo": -10.0 - math.log(2) * (1 if i % 4 else -1),
             "pvi_bits": 1.0 if i % 4 else -1.0} for i in range(40)]
    rows.append({"profile_id": "p0", "skipped": True})
    s = pvi.summarise(rows, 200, "test")
    assert s["n"] == 40 and s["skipped"] == 1
    assert abs(s["v_information"] - 0.5) < 1e-9 and abs(s["share_negative"] - 0.25) < 1e-9
    assert abs(s["reply_nll_with"] - 1.0) < 1e-9
    lo, hi = s["v_information_ci"]
    assert lo <= 0.5 <= hi


def main(argv: list[str]) -> int:
    wanted = [a for a in argv[1:] if not a.startswith("-")]
    selected = [(n, f) for n, f in CHECKS if not wanted or any(w in n for w in wanted)]
    failures = []
    print(f"running {len(selected)} extra checks\n")
    for name, fn in selected:
        llm_mod.release_all()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, traceback.format_exc(limit=4)))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    llm_mod.release_all()
    print("\n" + "=" * 72)
    print(f"{len(selected) - len(failures)} passed, {len(failures)} failed")
    for name, tb in failures:
        print(f"\n--- {name}\n{tb}")
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
