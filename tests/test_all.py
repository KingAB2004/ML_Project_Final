"""Component checks for the whole project. Plain asserts, no framework.

Run:  python tests/test_all.py            (all checks)
      python tests/test_all.py memory     (only checks whose name contains 'memory')

Everything runs against the `echo` backend: deterministic stubs, no GPU, no network, no model download. A
check that needs an optional dependency (numpy) reports SKIP rather than failing.

Each check is the smallest thing that fails if the logic breaks - one per row of PLAN.md Sec. 18, plus
cross-file consistency checks that catch the failure mode static reading misses: two files disagreeing
about a format.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "human_eval"))

import agents                                     # noqa: E402
import annotate                                   # noqa: E402
import build_sft                                  # noqa: E402
import common                                     # noqa: E402
import conformal                                  # noqa: E402
import counterfactual                             # noqa: E402
import convert_corpus                          # noqa: E402
import dialogue                                   # noqa: E402
import evaluate as evaluate_mod                   # noqa: E402
import filter as filter_mod                       # noqa: E402
import external                               # noqa: E402
import grounding                                  # noqa: E402
import judge as judge_mod                         # noqa: E402
import llm as llm_mod                             # noqa: E402
import memory as memory_mod                       # noqa: E402
import metrics                                    # noqa: E402
import pacing                                     # noqa: E402
import profiles                               # noqa: E402
import seeds                                  # noqa: E402
import report                                     # noqa: E402
import sessions as sessions_mod                   # noqa: E402
import agreement as agreement_mod                 # noqa: E402
import sample as sample_mod                       # noqa: E402

CHECKS: list[tuple[str, callable]] = []
TMP = Path(tempfile.mkdtemp(prefix="cocoon_new_tests_"))


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


class Skip(Exception):
    """Raised by a check whose optional dependency is missing."""


def echo(role: str) -> llm_mod.LLM:
    llm_mod.release_all()
    return llm_mod.LLM(role, backend="echo", cache_dir=TMP / "cache")


# --------------------------------------------------------------------------- fixtures


def fake_turns() -> list[dict]:
    return [
        {"turn_index": 0, "role": "supporter", "text": "Hey - how are things today?",
         "phase": "listening", "ladder_rung": "L0", "meta": {"source": "opener_pool"}},
        {"turn_index": 1, "role": "user", "text": "Work has been a lot. I'm fine though.",
         "phase": "listening", "disclosure_depth": 1, "meta": {}},
        {"turn_index": 2, "role": "supporter", "text": "That sounds like a heavy stretch.",
         "phase": "listening", "ladder_rung": "L0", "analysis": "worn down, minimising",
         "strategy": "reflect the load without probing", "meta": {}},
        {"turn_index": 3, "role": "user", "text": "I haven't really told anyone how bad the week was.",
         "phase": "listening", "disclosure_depth": 2, "meta": {}},
    ]


def fake_profile() -> dict:
    return {
        "profile_id": "p000001", "emotion": "worn down", "feeling": "running on empty",
        "terminal_need": "wants to matter beyond what they produce",
        "need_chain": [{"node_id": "nd000", "depth": 0, "text": "running on empty", "parent_id": None},
                       {"node_id": "nd001", "depth": 1, "text": "wants rest without guilt",
                        "parent_id": "nd000"},
                       {"node_id": "nd002", "depth": 2,
                        "text": "wants to matter beyond what they produce", "parent_id": "nd001"}],
        "memory": [{"text": "worked two weekends", "when_relative": "last fortnight", "salience": 0.8}],
        "persona_surface": {"style": "understated"},
        "persona_hidden": {"terminal_need": "wants to matter beyond what they produce",
                           "resistance_level": "medium", "disclosure_triggers": ["accurate reflection"],
                           "disclosure_blockers": ["quick advice"]},
        "resistance_level": "medium", "problem_type": "job crisis", "ambiguous": False,
    }


def fake_session(session_id: str = "p000001-s1", index: int = 1, gap: float | None = None) -> dict:
    return {"session_id": session_id, "profile_id": "p000001", "session_index": index,
            "prev_session_id": None if index == 1 else "p000001-s1", "gap_days_from_prev": gap,
            "profile_snapshot": fake_profile(), "stage_boundary_turn": 2, "turns": fake_turns(),
            "qc": {}, "annotations_complete": True}


def span(turn_index: int, quote: str, session_id: str = "p000001-s1") -> dict:
    return {"session_id": session_id, "turn_index": turn_index, "quote": quote,
            "char_start": -1, "char_end": -1}


# --------------------------------------------------------------------------- common


@check("common.seeds_are_deterministic_and_position_independent")
def _() -> None:
    a = common.derive_seed("p000007", "dialogue")
    b = common.derive_seed("p000007", "dialogue")
    c = common.derive_seed("p000007", "annotate")
    assert a == b, "same record and stage must give the same seed"
    assert a != c, "different stages must not collide"


@check("common.jsonl_roundtrip_and_resume_ids")
def _() -> None:
    path = TMP / "rt.jsonl"
    common.write_jsonl(path, [{"id": "a"}, {"id": "b"}])
    assert [r["id"] for r in common.read_jsonl(path)] == ["a", "b"]
    common.append_jsonl(path, {"id": "c"})
    assert common.existing_ids(path, "id") == {"a", "b", "c"}
    assert common.read_jsonl(TMP / "missing.jsonl") == [], "a missing file must read as empty, not raise"


@check("common.span_location_and_transcript_rendering")
def _() -> None:
    turns = fake_turns()
    found = common.locate_span(turns[1]["text"], "Work has been a lot")
    assert found == (0, 19), found
    assert common.locate_span(turns[1]["text"], "never said this") is None
    text = common.render_transcript(turns)
    assert text.startswith("supporter:") and "seeker:" in text
    mirrored = common.mirrored_history(turns, "supporter")
    assert mirrored[0]["role"] == "assistant" and mirrored[1]["role"] == "user"
    assert abs(common.normalize_scale(7.0) - 1.0) < 1e-9
    assert abs(common.normalize_scale(1.0)) < 1e-9


@check("common.prompt_templates_have_resolvable_placeholders")
def _() -> None:
    import re

    for path in sorted((ROOT / "prompts").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # single braces are placeholders, doubled braces are literal JSON in the examples
        names = set(re.findall(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})", text))
        kwargs = {n: "x" for n in names}
        try:
            common.fill(text, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{path.name} cannot be formatted: {exc}") from exc


# --------------------------------------------------------------------------- llm


@check("llm.echo_backend_returns_shape_correct_output")
def _() -> None:
    handle = echo("analyzer")
    try:
        out = handle.structured("Return JSON only: analyzer state", required=("emotional_state",))
        assert out["parse_failed"] is False and "implicit_needs" in out
        scale = handle.chat("[Question List]\n1. a\n2. b\n\n[Rating Scale]\n1: Strongly Disagree")
        assert "1:" in scale and "2:" in scale
    finally:
        handle.release()


@check("llm.residency_blocks_a_second_distinct_model")
def _() -> None:
    first = echo("generator")
    try:
        try:
            llm_mod.LLM("judge", backend="echo", cache_dir=TMP / "cache")
        except llm_mod.ResidencyError:
            pass
        else:
            raise AssertionError("a second distinct model was allowed to load (PLAN R2 violated)")
        view = first.view("simulator")
        assert view.spec["model"] == first.spec["model"], "a view must reuse the resident weights"
        assert view.use_adapter is False, "a non-supporter view must not use the supporter adapter"
    finally:
        first.release()
    again = echo("judge")          # allowed once the first is released
    again.release()


@check("llm.ollama_payload_is_built_correctly")
def _() -> None:
    assert "ollama" in llm_mod.BACKENDS
    spec = {"model": "Qwen/Qwen2.5-7B-Instruct", "ollama_tag": "qwen2.5:7b-instruct",
            "temperature": 0.7, "top_p": 0.9, "max_tokens": 256, "max_model_len": 4096}
    body = llm_mod.ollama_payload(spec, "sys", [{"role": "user", "content": "hi"}],
                                  {"temperature": 0.0})
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1]["content"] == "hi"
    assert body["options"]["temperature"] == 0.0, "a per-call override must win over the role default"
    assert body["options"]["num_predict"] == 256 and body["options"]["num_ctx"] == 4096
    no_system = llm_mod.ollama_payload(spec, None, [{"role": "user", "content": "hi"}])
    assert len(no_system["messages"]) == 1
    assert llm_mod.ollama_tag({"model": "org/Some-Model"}) == "some-model", "tag falls back to the id"
    roles = common.load_config("models")["roles"]
    missing = [r for r, sp in roles.items() if not sp.get("ollama_tag")]
    assert not missing, f"roles without an ollama_tag: {missing}"


@check("llm.json_extraction_survives_fences_and_prose")
def _() -> None:
    assert llm_mod.extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert llm_mod.extract_json('sure, here: {"a": {"b": 2}} hope that helps')["a"]["b"] == 2
    assert llm_mod.extract_json('{"a": "}"}')["a"] == "}", "a brace inside a string must not end the object"
    assert llm_mod.extract_json("no json here") is None
    assert llm_mod.extract_json("{broken") is None


@check("llm.pmap_keeps_input_order_and_runs_concurrently")
def _() -> None:
    import threading
    import time

    seen: set[int] = set()

    def slow(x: int) -> int:
        seen.add(threading.get_ident())
        time.sleep(0.05 * (5 - x))      # later items finish first
        return x * x

    handle = echo("generator")
    try:
        assert list(llm_mod.pmap(slow, range(5), handle)) == [0, 1, 4, 9, 16]
        assert len(seen) > 1 or int(common.cfg("parallel.workers", default=1)) <= 1, "expected worker threads"

        class Serial:
            backend_name = "transformers"
        seen.clear()
        assert list(llm_mod.pmap(slow, range(3), Serial())) == [0, 1, 4]
        assert seen == {threading.get_ident()}, "in-process backends must stay on the calling thread"
    finally:
        handle.release()


# --------------------------------------------------------------------------- judge


@check("judge.every_scale_prompt_parses_into_the_expected_item_count")
def _() -> None:
    expected = {"aels": 10, "crs": 10, "rac": 16, "basic": 6, "success": 1, "ip": 7, "pri": 7}
    for scale, count in expected.items():
        items = judge_mod.expected_items(scale)
        assert items == list(range(1, count + 1)), f"{scale}: got {items}"


@check("judge.score_parsing_rejects_partial_and_out_of_range")
def _() -> None:
    good = judge_mod.parse_scores("1: 5\n2: 6\n3: 7", [1, 2, 3])
    assert good == {1: 5, 2: 6, 3: 7}
    assert judge_mod.parse_scores("1: 5\n2: 6", [1, 2, 3]) is None, "a partial parse must fail"
    assert judge_mod.parse_scores("1: 9\n2: 6\n3: 7", [1, 2, 3]) is None, "9 is off the 1-7 scale"
    assert judge_mod.parse_scores("(1). 4  (2): 5", [1, 2]) == {1: 4, 2: 5}
    assert judge_mod.parse_scores("Question 1: Score - 4\n\nThe supporter approached it.", [1]) == {1: 4}
    assert judge_mod.parse_scores("Item 2: score 3\nQuestion 1: 6", [1, 2]) == {1: 6, 2: 3}


@check("judge.partial_answer_is_retried_once_then_rejected")
def _() -> None:
    class Scripted:
        """Answers from a script: a partial reply first, then whatever comes next."""
        spec = {"model": "scripted/judge"}

        def __init__(self, replies):
            self.replies, self.prompts = list(replies), []

        def chat(self, prompt, **_):
            self.prompts.append(prompt)
            return self.replies.pop(0)

    full = "\n".join(f"{i}: 2" for i in range(1, 8))
    llm = Scripted(["2: 3\n5: 4\n7: 3", full] + [full] * 6)   # extra replies for samples_per_item > 1
    res = judge_mod.Judge(llm=llm).score("ip", "s#1", "supporter: hi", fake_profile(), target_text="hi")
    assert not res.parse_failed and res.items[1] == 2, "a complete answer on the retry must be used"
    assert "did not score every question" in llm.prompts[1]
    llm = Scripted(["2: 3"] * 8)
    res = judge_mod.Judge(llm=llm).score("ip", "s#1", "supporter: hi", fake_profile(), target_text="hi")
    assert res.parse_failed, "a partial answer must never be filled in"


@check("judge.redactions_hide_resistance_and_the_need_from_ip_and_pri")
def _() -> None:
    profile = fake_profile()
    for scale in ("ip", "pri"):
        info, redacted = judge_mod.build_info(profile, scale)
        assert profile["terminal_need"] not in info, f"{scale} judge must not see the terminal need"
        assert profile["resistance_level"] not in info
        assert "terminal_need" in redacted and "resistance_level" in redacted
    info, _ = judge_mod.build_info(profile, "success")
    assert profile["terminal_need"] in info, "Success Rate is scored against the ground truth"


@check("judge.separation_guard_rejects_a_shared_model")
def _() -> None:
    models = common.load_config("models")
    original = models["roles"]["judge"]["model"]
    try:
        models["roles"]["judge"]["model"] = models["roles"]["generator"]["model"]
        try:
            judge_mod.assert_judge_separate()
        except judge_mod.JudgeSeparationError:
            pass
        else:
            raise AssertionError("a judge sharing the generator's model was accepted")
    finally:
        models["roles"]["judge"]["model"] = original
    judge_mod.assert_judge_separate()   # the shipped config must pass


# --------------------------------------------------------------------------- grounding


@check("grounding.span_validation_accepts_only_verbatim_quotes")
def _() -> None:
    turns = fake_turns()
    ok, _ = grounding.validate_span(span(1, "Work has been a lot"), turns)
    assert ok
    ok, why = grounding.validate_span(span(1, "work has been a LOT"), turns)
    assert not ok and "verbatim" in why, why
    bad_offsets = span(1, "Work has been a lot")
    bad_offsets.update({"char_start": 5, "char_end": 24})
    ok, why = grounding.validate_span(bad_offsets, turns)
    assert not ok and "offsets" in why, why
    ok, why = grounding.validate_span(span(2, "That sounds like a heavy stretch."), turns)
    assert not ok and "supporter" in why, "a supporter turn is not evidence about the seeker"
    ok, why = grounding.validate_span(span(99, "anything"), turns)
    assert not ok and "no turn" in why


@check("grounding.bare_or_malformed_spans_never_crash")
def _() -> None:
    turns = fake_turns()
    out = grounding.zero_unsupported_claims({
        "emotional_state": {"label": "tired", "confidence": 0.8,
                            "evidence_spans": ["Work has been a lot."]},          # bare string, verbatim
        "implicit_needs": [{"text": "to be heard", "confidence": 0.7,
                            "evidence_spans": ["never said this", 42, {"turn_index": "x", "quote": "a"}]}],
    }, turns, session_id="p000001-s1")
    good = out["emotional_state"]["evidence_spans"]
    assert good and good[0]["turn_index"] == 1, "a verbatim bare quote is located in its seeker turn"
    need = out["implicit_needs"][0]
    assert need["confidence"] == 0.0 and not need["evidence_spans"], "unlocatable or malformed spans ground nothing"


@check("grounding.unsupported_claims_drop_to_zero_confidence")
def _() -> None:
    turns = fake_turns()
    analyzer = {
        "emotional_state": {"label": "worn down", "confidence": 0.9,
                            "evidence_spans": [span(1, "Work has been a lot")]},
        "implicit_needs": [
            {"text": "wants rest", "depth": 1, "confidence": 0.8,
             "evidence_spans": [span(1, "I'm fine though")]},
            {"text": "wants recognition", "depth": 2, "confidence": 0.9,
             "evidence_spans": [span(1, "fabricated quote")]},
        ],
        "resistance_estimate": {"level": "medium", "confidence": 0.6, "evidence_spans": []},
    }
    out = grounding.zero_unsupported_claims(analyzer, turns, session_id="p000001-s1")
    assert out["emotional_state"]["confidence"] == 0.9
    assert out["implicit_needs"][0]["confidence"] == 0.8
    assert out["implicit_needs"][1]["confidence"] == 0.0, "a fabricated span must zero the claim"
    assert out["resistance_estimate"]["confidence"] == 0.0, "no span at all must zero the claim"
    assert out["grounding_report"]["invalid_spans"] >= 1


@check("grounding.draft_checks_catch_denial_advice_and_overreach")
def _() -> None:
    turns = fake_turns()
    rep = grounding.check_draft("Maybe you feel unseen because nobody notices your effort.", turns,
                                permitted_rung="L3",
                                forbidden_inferences=["you feel unseen because nobody notices your effort"])
    assert rep.has("denied_inference") and rep.mass() >= 1.0 - 1e-9
    rep = grounding.check_draft("You should just tell your manager you need a break.", turns,
                                permitted_rung="L1", phase="listening")
    assert rep.has("unsolicited_advice")
    rep = grounding.check_draft("You resent them for never asking how you are.", turns,
                                permitted_rung="L0")
    assert rep.has("unsupported_state_attribution") and rep.has("overreaching_depth")
    rep = grounding.check_draft("That sounds like a heavy stretch.", turns, permitted_rung="L0",
                                phase="listening")
    assert rep.ok, [v.to_dict() for v in rep.violations]
    assert grounding.classify_rung("It sounds like you might want rest. I could be off.") == "L2"
    assert grounding.classify_rung("How has this week been?") == "L1"
    assert grounding.classify_rung("That sounds heavy.") == "L0"
    viol = grounding.check_fabricated_facts("Your sister said the same thing.", turns, ["Your sister"])
    assert viol and viol[0].category == "fabricated_fact"


@check("grounding.denied_inference_carries_the_highest_weight")
def _() -> None:
    w = grounding.CATEGORY_WEIGHTS
    assert w["denied_inference"] == max(w.values()), "the decisive category must dominate the gate score"


# --------------------------------------------------------------------------- conformal


@check("conformal.risk_curve_is_monotone_in_lambda")
def _() -> None:
    scores = [i / 20 for i in range(21)]
    violations = [s > 0.6 for s in scores]
    curve = conformal.risk_curve(scores, violations, conformal.default_grid(21), delta=0.1)
    risks = [row["risk"] for row in curve]
    assert risks == sorted(risks), "a looser gate cannot lower empirical risk"
    releases = [row["release_rate"] for row in curve]
    assert releases == sorted(releases)


@check("conformal.lambda_hat_is_the_largest_safe_threshold")
def _() -> None:
    # 200 turns, IP rises with the score: everything above 0.6 is a violation.
    scores = [i / 199 for i in range(200)]
    ips = [0.9 if s > 0.6 else 0.1 for s in scores]
    cal = conformal.calibrate(scores, ips, alpha=0.20, tau=0.5, delta=0.1)
    assert not cal.vacuous
    safe = [r for r in cal.risk_curve if r["risk_ucb"] <= cal.alpha]
    assert cal.lambda_hat == max(r["lambda"] for r in safe)
    over = [r for r in cal.risk_curve if r["lambda"] > cal.lambda_hat]
    assert all(r["risk_ucb"] > cal.alpha for r in over), "a safer larger lambda was left unused"
    assert cal.releases(cal.lambda_hat) and not cal.releases(cal.lambda_hat + 0.01)


@check("conformal.impossible_budget_is_reported_as_vacuous")
def _() -> None:
    cal = conformal.calibrate([0.0] * 50, [0.9] * 50, alpha=0.01, tau=0.5, delta=0.1)
    assert cal.vacuous, "every turn violates, so no threshold can meet a 1% budget"
    assert cal.lambda_hat == min(cal.lambda_grid)


@check("conformal.bounds_are_valid_and_ordered")
def _() -> None:
    h = conformal.hoeffding_ucb(0.1, 300, 0.1)
    b = conformal.bentkus_ucb(0.1, 300, 0.1)
    hb = conformal.risk_ucb(0.1, 300, 0.1)
    assert 0.1 <= hb <= min(h, b) + 1e-12, (h, b, hb)
    assert hb >= 0.1, "an upper bound cannot fall below the empirical risk"
    assert conformal.hoeffding_ucb(0.5, 0, 0.1) == 1.0, "no calibration data means no guarantee"


@check("conformal.nonconformity_is_bounded_and_weighted")
def _() -> None:
    assert conformal.nonconformity(0.0, 0.0) == 0.0
    assert abs(conformal.nonconformity(1.0, 1.0) - 1.0) < 1e-9
    w = common.cfg("conformal.score_weights")
    mixed = conformal.nonconformity(1.0, 0.0)
    assert abs(mixed - w["ip_pred"]) < 1e-9
    sweep = conformal.sweep_alphas([i / 99 for i in range(100)],
                                   [0.9 if i > 60 else 0.1 for i in range(100)])
    lambdas = [row["lambda_hat"] for row in sweep]
    assert lambdas == sorted(lambdas), "a larger budget must not tighten the gate"


@check("conformal.calibration_artifact_roundtrips_with_its_distribution_id")
def _() -> None:
    cal = conformal.calibrate([0.1, 0.2, 0.3], [0.1, 0.1, 0.9], alpha=0.5,
                              distribution_id="own_generator_v1")
    path = TMP / "calibration.json"
    cal.save(path)
    back = conformal.Calibration.load(path)
    assert back.lambda_hat == cal.lambda_hat and back.distribution_id == "own_generator_v1"


# --------------------------------------------------------------------------- memory


@check("memory.ungrounded_proposals_are_refused")
def _() -> None:
    mem = memory_mod.NeedStateMemory("p000001")
    assert mem.propose("wants rest", [], 0.8) is None, "a claim with no span must not enter belief state"
    assert mem.nodes == {}
    assert any(t["to"] == "rejected" for t in mem.transitions), "the refusal must be recorded"
    node = mem.propose("wants rest without guilt", [span(1, "Work has been a lot")], 0.6, depth=1)
    assert node is not None and node.status == "hypothesis"


@check("memory.disconfirmation_blocks_reproposal_and_reopen_is_hard")
def _() -> None:
    mem = memory_mod.NeedStateMemory("p000001")
    node = mem.propose("wants recognition at work", [span(1, "Work has been a lot")], 0.6, depth=2)
    mem.disconfirm(node.node_id, span(3, "I haven't really told anyone how bad the week was."))
    assert mem.nodes[node.node_id].status == "disconfirmed"
    assert mem.nodes[node.node_id].blocked_from_reproposal
    assert mem.forbidden_inferences() == ["wants recognition at work"]
    assert mem.propose("wants recognition at work", [span(1, "Work has been a lot")], 0.9) is None
    assert mem.reopen(node.node_id, [span(1, "Work has been a lot")]) is None, "one span must not reopen"
    reopened = mem.reopen(node.node_id, [span(1, "Work has been a lot"), span(3, "I haven't really told")])
    assert reopened is not None and not reopened.blocked_from_reproposal


@check("memory.confirmation_requires_enough_spans")
def _() -> None:
    mem = memory_mod.NeedStateMemory("p000001")
    node = mem.propose("wants rest without guilt", [span(1, "Work has been a lot")], 0.5, depth=1)
    mem.confirm(node.node_id, span(1, "Work has been a lot"), explicit_affirmation=True)
    assert mem.nodes[node.node_id].status == "hypothesis", "one span cannot confirm"
    mem.confirm(node.node_id, span(3, "I haven't really told anyone"), explicit_affirmation=True)
    assert mem.nodes[node.node_id].status == "confirmed"
    assert mem.nodes[node.node_id].confidence >= 0.85


@check("memory.decay_touches_only_unconfirmed_hypotheses")
def _() -> None:
    mem = memory_mod.NeedStateMemory("p000001")
    hyp = mem.propose("wants rest", [span(1, "Work has been a lot")], 0.8, depth=1)
    conf = mem.propose("wants to be asked about", [span(3, "I haven't really told anyone")], 0.8, depth=2)
    mem.confirm(conf.node_id, span(3, "I haven't really told anyone"))
    mem.confirm(conf.node_id, span(1, "Work has been a lot"))
    before_conf = mem.nodes[conf.node_id].confidence
    mem.decay(gap_days=28.0)      # two half-lives at the default 14 days
    assert abs(mem.nodes[hyp.node_id].confidence - 0.2) < 1e-6, mem.nodes[hyp.node_id].confidence
    assert mem.nodes[conf.node_id].confidence == before_conf, "a tested link stays tested"
    mem.decay(gap_days=120.0)
    assert mem.nodes[hyp.node_id].dormant, "a long-decayed hypothesis must go dormant"


@check("memory.reinstated_concern_returns_at_reduced_confidence")
def _() -> None:
    mem = memory_mod.NeedStateMemory("p000001")
    node = mem.propose("wants rest", [span(1, "Work has been a lot")], 0.8, depth=1)
    mem.decay(gap_days=120.0)
    mem.reinstate(node.node_id, span(3, "I haven't really told anyone"))
    n = mem.nodes[node.node_id]
    assert n.status == "hypothesis" and not n.dormant and n.reinstated_count == 1
    assert n.confidence < 0.8, "resurfacing must not restore full confidence"


@check("memory.brief_reports_beliefs_gaps_and_forbidden_inferences")
def _() -> None:
    mem = memory_mod.NeedStateMemory("p000001")
    terminal = mem.propose("wants to matter beyond output", [span(3, "I haven't really told anyone")], 0.7,
                           depth=2)
    denied = mem.propose("wants praise from the boss", [span(1, "Work has been a lot")], 0.5, depth=2)
    mem.disconfirm(denied.node_id, span(1, "I'm fine though"))
    brief = mem.brief(open_questions=["who knows how bad the week was"])
    assert brief["best_terminal"]["node_id"] == terminal.node_id
    assert "wants praise from the boss" in brief["forbidden_inferences"]
    assert brief["open_questions"] == ["who knows how bad the week was"]
    text = mem.render_brief(brief["open_questions"])
    assert "DO NOT re-propose" in text and "open question" in text


@check("memory.state_is_a_fold_over_the_transition_log")
def _() -> None:
    log = TMP / "mem_log.jsonl"
    mem = memory_mod.NeedStateMemory("p000001", log_path=log)
    a = mem.propose("wants rest", [span(1, "Work has been a lot")], 0.6, depth=1)
    b = mem.propose("wants praise", [span(3, "I haven't really told anyone")], 0.6, depth=2)
    mem.disconfirm(b.node_id, span(1, "I'm fine though"))
    mem.confirm(a.node_id, span(3, "I haven't really told anyone"))
    mem.confirm(a.node_id, span(1, "Work has been a lot"))
    folded = memory_mod.NeedStateMemory.fold(common.read_jsonl(log))
    assert folded == mem.status_map(), (folded, mem.status_map())
    path = TMP / "mem_state.jsonl"
    mem.save(path)
    back = memory_mod.NeedStateMemory.load("p000001", path)
    assert back.status_map() == mem.status_map()
    assert back.new_node_id() not in mem.nodes, "loaded memory must not reissue an existing node id"


# --------------------------------------------------------------------------- ladder (E2 mechanism)


@check("ladder.caps_compose_in_the_strict_direction")
def _() -> None:
    assert pacing.at_most("L2") == ["L0", "L1", "L2"]
    assert pacing.cap("L3", "L1") == "L1" and pacing.cap("L0", "L2") == "L0"
    assert pacing.at_most("L0") == ["L0"]
    assert set(pacing.LADDER) == set(pacing.LADDER_DESCRIPTION)


@check("ladder.low_confidence_restricts_to_exploratory_moves")
def _() -> None:
    rung, reason = pacing.permitted_rung(analyzer_confidence=0.1,
                                         memory_brief={"best_terminal": {"status": "confirmed"}})
    assert rung == "L1" and "below floor" in reason
    rung, reason = pacing.permitted_rung(analyzer_confidence=0.9,
                                         memory_brief={"best_terminal": {"status": "confirmed"}})
    assert rung == "L3" and reason == ""


@check("ladder.naming_the_terminal_need_requires_a_confirmed_belief")
def _() -> None:
    rung, reason = pacing.permitted_rung(analyzer_confidence=0.9,
                                         memory_brief={"best_terminal": {"status": "hypothesis"}})
    assert rung == "L2" and "hypothesis" in reason
    rung, _ = pacing.permitted_rung(analyzer_confidence=0.9, memory_brief={})
    assert rung == "L2", "with nothing believed yet, L3 must stay closed"


# --------------------------------------------------------------------------- build_sft


@check("build_sft.target_sequence_roundtrips_in_both_arms")
def _() -> None:
    target = build_sft.render_target("worn down", "reflect the load", "That sounds heavy.")
    parsed = build_sft.parse_target(target)
    assert parsed == {"analysis": "worn down", "strategy": "reflect the load",
                      "response": "That sounds heavy."}
    plain = build_sft.render_target("a", "b", "That sounds heavy.", with_thoughts=False)
    assert "<analysis>" not in plain
    assert build_sft.parse_target(plain)["response"] == "That sounds heavy."
    truncated = "<analysis>worn</analysis>\n<strategy>reflect</strategy>\n<response>That sounds"
    assert build_sft.parse_target(truncated)["response"] == "That sounds", "truncation must still yield text"
    untagged = "Analysis: worn down.\nStrategy: reflect.\n\nReply: That sounds like a long week."
    assert build_sft.parse_target(untagged)["response"] == "That sounds like a long week.", \
        "an untagged analysis must never reach the seeker"
    assert build_sft.parse_target("Just a plain reply.")["response"] == "Just a plain reply."
    # v2 probe: tagged thoughts, no <response> tag, the prompt's label copied in front of the reply
    leaked = ("<analysis>They are stressed.</analysis> <strategy>Reassure.</strategy> \n\n"
              "what you say to them: It's okay to feel this way.")
    assert build_sft.parse_target(leaked)["response"] == "It's okay to feel this way."
    assert build_sft.parse_target("<response>what you say to them: Hi.</response>")["response"] == "Hi."
    for thoughts_only in ("Analysis: worn down.\n\nReply: ", "Analysis: worn down.\nStrategy: reflect.",
                          "<analysis>worn down</analysis>", "<analysis>cut off mid-thou"):
        assert build_sft.parse_target(thoughts_only)["response"] == "", thoughts_only


@check("evaluate.reactive_arm_runs_the_reactive_simulator")
def _() -> None:
    class Sim:
        systems: list = []

        def chat(self, prompt, system=None, **kw):
            self.systems.append(system)
            return f"seeker line {len(self.systems)}"

    class Pipe:
        def step(self, turns, **kw):
            return agents.TurnResult(text=f"supporter line {len(turns)}", permitted_rung="L0", analyzer={},
                                     strategist={}, critic={}, path="released", revisions=0)

    sim = Sim()
    evaluate_mod.run_session(Pipe(), sim, fake_profile(), "p-x-s1", 1, 0.0, 2, reactive=True)
    assert sim.systems[0] == dialogue.simulator_system(fake_profile(), reactive=True)



@check("evaluate.memory_arms_are_multi_session_corrective_and_advance_the_profile")
def _() -> None:
    corrective = dialogue.simulator_system(fake_profile(), corrective=True)
    plain = dialogue.simulator_system(fake_profile())
    assert corrective.startswith(plain) and "that's not really it" in corrective.lower()
    for arm in ("mem_none", "mem_summary", "mem_dense", "mem_event", "mem_needstate"):
        spec = evaluate_mod.resolve_arm(arm)
        assert spec["multi_session"] and spec["advance_profile"] and spec["simulator"] == "corrective", arm
        assert spec["architecture"] == "decomposed" and spec["supporter"] == "base" and not spec["gate"], arm

@check("build_sft.uses_the_profile_split_on_disk_so_test_profiles_never_train")
def _() -> None:
    pdir = TMP / "split_profiles"
    common.write_jsonl(pdir / "profiles_train.jsonl", [{"profile_id": "p000001"}])
    common.write_jsonl(pdir / "profiles_test.jsonl", [{"profile_id": "p000002"}])
    a = fake_session("p000001-s1"); a["profile_id"] = "p000001"
    b = fake_session("p000002-s1"); b["profile_id"] = "p000002"
    c = fake_session("p000003-s1"); c["profile_id"] = "p000003"
    sessions = TMP / "split_sessions.jsonl"
    common.write_jsonl(sessions, [a, b, c])
    stats = build_sft.build("with_thoughts", sessions, TMP / "split_sft", pdir)
    assert stats["counts"]["train"] > 0 and stats["counts"]["test"] > 0
    train_ids = {r["profile_id"] for r in common.read_jsonl(TMP / "split_sft" / "with_thoughts_train.jsonl")}
    assert train_ids == {"p000001"}, f"test profile leaked into train: {train_ids}"
    assert stats["unassigned_sessions"] == 1, "a profile with no split must not default to train"


@check("train.examples_use_the_chat_template_and_never_cut_the_target")
def _() -> None:
    import train

    class Tok:
        """Whitespace 'tokenizer' with a ChatML template, standing in for Qwen's."""
        eos_token = "<|im_end|>"

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
            return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    row = {"system": "be kind", "input": "seeker: hi", "target": "<response>hello</response>"}
    enc = train.encode_example(Tok(), row, seq_len=64)
    n_target = len("<response>hello</response><|im_end|>".split())
    assert enc["labels"][-n_target:] == enc["input_ids"][-n_target:], "the target must carry the loss"
    assert all(l == -100 for l in enc["labels"][:-n_target]), "the prompt must be masked"
    prompt = Tok().apply_chat_template([{"role": "system", "content": "be kind"},
                                        {"role": "user", "content": "seeker: hi"}])
    assert enc["input_ids"][:len(prompt.split())] == prompt.split(), "training must use the inference template"
    assert train.encode_example(Tok(), row, seq_len=4) is None, "an example that does not fit is dropped, not cut"


@check("model_output_fields_are_coerced_not_trusted")
def _() -> None:
    assert common.to_int("2 - intermediate need") == 2 and common.to_int(None, 1) == 1
    assert common.to_int("7", lo=0, hi=3) == 3 and common.to_int(True, 5) == 5
    assert annotate.text_field(["reflect", "slow down"]) == "reflect; slow down"
    assert annotate.text_field("  phrase ") == "phrase" and annotate.text_field(None) == ""


@check("build_sft.both_arms_train_on_the_same_turns")
def _() -> None:
    session = fake_session()
    session["turns"].append({"turn_index": 4, "role": "supporter", "text": "No annotation on this one.",
                             "phase": "listening", "meta": {}})
    with_t = [r["turn_index"] for r in build_sft.examples_from_session(session, True)]
    without = [r["turn_index"] for r in build_sft.examples_from_session(session, False)]
    assert with_t == without == [2], (with_t, without)
    assert build_sft.gap_statement(14.2, 2).startswith("It has been about 14 days")
    assert build_sft.gap_statement(None, 1) == ""


@check("build_sft.examples_match_inference_shape_and_skip_the_opener")
def _() -> None:
    session = fake_session(gap=None)
    rows = build_sft.examples_from_session(session)
    assert rows and all(r["turn_index"] != 0 for r in rows), "the opener is not a training target"
    assert "[memory]" in rows[0]["input"], "the memory block must be in training inputs too"
    assert rows[0]["system"] == build_sft.SUPPORTER_SYSTEM
    followup = fake_session("p000001-s2", index=2, gap=21.0)
    rows2 = build_sft.examples_from_session(followup)
    assert "[elapsed]" in rows2[0]["input"], "a follow-up session must state the elapsed interval"
    plain = build_sft.examples_from_session(session, with_thoughts=False)
    assert plain and "<analysis>" not in plain[0]["target"]
    assert not hasattr(build_sft, "pacing_rows_from_session"), \
        "the pacing-teacher seed belongs to Enhancement 5, which the modified SOP drops"


@check("profiles.splits_are_disjoint_by_profile")
def _() -> None:
    rows = [{"profile_id": f"p{i:06d}"} for i in range(100)]
    splits = profiles.split_profiles(rows)
    ids = {name: {r["profile_id"] for r in group} for name, group in splits.items()}
    assert set(ids) == {"train", "val", "test", "calibration"}
    seen: set[str] = set()
    for name, group in ids.items():
        assert not (group & seen), f"{name} overlaps an earlier split"
        seen |= group
    assert len(seen) == 100, "every profile must land in exactly one split"
    assert ids["calibration"], "the conformal calibration slice must be non-empty and disjoint"
    again = profiles.split_profiles(rows)
    assert {r["profile_id"] for r in again["test"]} == ids["test"], "splits must be reproducible"


@check("profiles.resistance_strata_and_chain_validation")
def _() -> None:
    plan = profiles.resistance_plan(8)
    assert plan.count("medium") == 4 and plan.count("low") == 2 and plan.count("high") == 2, plan
    ok, _ = profiles.check_chain(fake_profile()["need_chain"])
    assert ok
    too_short, why = profiles.check_chain(fake_profile()["need_chain"][:2])
    assert not too_short and "2 nodes" in why
    trivial = [{"text": "running on empty", "depth": 0}, {"text": "wants rest", "depth": 1},
               {"text": "running on empty", "depth": 2}]
    ok, why = profiles.check_chain(trivial)
    assert not ok and "paraphrases" in why, why


# --------------------------------------------------------------------------- filter and annotate


@check("filter.structural_checks_catch_the_failures_that_matter")
def _() -> None:
    good = fake_session()
    good["turns"] += [
        {"turn_index": 4, "role": "supporter", "text": "What made this week heavier than usual?",
         "phase": "listening", "ladder_rung": "L1", "meta": {}},
        {"turn_index": 5, "role": "user", "text": "Deadlines stacked up and nobody noticed.",
         "phase": "listening", "disclosure_depth": 2, "meta": {}},
    ]
    assert filter_mod.structural_problems(good) == [], filter_mod.structural_problems(good)

    short = fake_session()
    short["turns"] = short["turns"][:2]
    assert any("minimum" in p for p in filter_mod.structural_problems(short))

    role_break = fake_session()
    role_break["turns"][2]["role"] = "user"
    assert any("role repeats" in p for p in filter_mod.structural_problems(role_break))

    repeated = fake_session()
    repeated["turns"][3]["text"] = repeated["turns"][1]["text"]
    assert any("repeats turn" in p for p in filter_mod.structural_problems(repeated))

    leaky = fake_session()
    leaky["turns"][1]["text"] = "I want to matter beyond what I produce, honestly."
    assert any("recites the hidden need" in p for p in filter_mod.structural_problems(leaky))

    chinese = fake_session()
    chinese["turns"][1]["text"] = "我很累，不想说话。"
    assert any("non-English" in p for p in filter_mod.structural_problems(chinese))

    crisis = fake_session()
    crisis["turns"][1]["text"] = "Some days I think about suicide, but I'm fine."
    assert filter_mod.crisis_flags(crisis) == ["suicide"]


@check("filter.aels_thresholds_and_tail_pruning")
def _() -> None:
    items = {i: 6.0 for i in range(1, 11)}
    ok, _ = filter_mod.aels_verdict(items, 6.0)
    assert ok
    ok, why = filter_mod.aels_verdict(items, 4.0)
    assert not ok and "mean" in why
    low = dict(items)
    low[4] = 2.0
    ok, why = filter_mod.aels_verdict(low, 6.0)
    assert not ok and "items below" in why

    s1 = fake_session("p1-s1", 1); s1["qc"] = {"verdict": "keep"}
    s2 = fake_session("p1-s2", 2); s2["qc"] = {"verdict": "reject"}
    s3 = fake_session("p1-s3", 3); s3["qc"] = {"verdict": "keep"}
    kept = filter_mod.prune_broken_tails([s1, s2, s3])
    assert [s["session_id"] for s in kept] == ["p1-s1"], "a broken middle session truncates the sequence"


@check("annotate.depths_propagate_and_degeneracy_is_detected")
def _() -> None:
    session = fake_session()
    records = [{"turn_index": 2, "analysis": "worn down", "strategy": "reflect the load",
                "ladder_rung": "L1", "user_disclosure_depth": 2, "parse_failed": False}]
    out = annotate.apply_annotations(session, records)
    assert out["turns"][2]["ladder_rung"] == "L1" and out["annotations_complete"]
    assert out["turns"][3]["disclosure_depth"] == 2, "later seeker turns inherit the depth reached"
    degenerate = annotate.vocabulary_report([{"strategy": "reflect"} for _ in range(20)])
    assert degenerate["degenerate"] and degenerate["n_clusters"] == 1
    varied = annotate.vocabulary_report([{"strategy": s} for s in
                                         ["reflect the load", "ask about the weekend shifts",
                                          "name the guilt gently", "check what rest would mean",
                                          "sit with the tiredness", "invite the untold part"]])
    assert not varied["degenerate"]


# --------------------------------------------------------------------------- dialogue and sessions


@check("dialogue.session_generation_alternates_and_opens_proactively")
def _() -> None:
    handle = echo("generator")
    try:
        session = dialogue.generate_session(fake_profile(), "p000001-s1", handle, handle,
                                            n_supporter_turns=4)
    finally:
        handle.release()
    turns = session["turns"]
    assert turns[0]["role"] == "supporter", "the system must open in a proactive setting"
    roles = [t["role"] for t in turns]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles
    assert [t["turn_index"] for t in turns] == list(range(len(turns)))
    assert {t["phase"] for t in turns} <= {"listening", "suggestion"}
    assert filter_mod.structural_problems(session) == [], filter_mod.structural_problems(session)
    rungs = [dialogue.rung_schedule(i, 10) for i in range(10)]
    assert rungs == sorted(rungs, key=lambda r: pacing.RUNG_INDEX[r]), rungs


@check("dialogue.a_turn_that_copies_an_earlier_line_is_resampled_once")
def _() -> None:
    turns = fake_turns()
    calls = []

    def scripted(replies):
        def chat(prompt, **kw):
            calls.append(kw.get("use_cache", True))
            return replies.pop(0)
        return chat

    copied = "seeker: That sounds like a heavy stretch."          # the supporter's line, relabelled
    out = dialogue.fresh_turn(scripted([copied, "It was. I didn't sleep much."]), "p", turns, "seeker")
    assert out == "It was. I didn't sleep much.", out
    assert calls == [True, False], "the retry must bypass the cache, or it returns the same copy"
    calls.clear()
    assert dialogue.fresh_turn(scripted(["Something new."]), "p", turns, "seeker") == "Something new."
    assert calls == [True], "a fresh turn must not trigger a retry"


@check("dialogue.simulator_prompt_carries_the_hidden_need_and_resistance")
def _() -> None:
    system = dialogue.simulator_system(fake_profile())
    assert "wants to matter beyond what they produce" in system
    assert "medium" in system and "quick advice" in system
    reactive = dialogue.simulator_system(fake_profile(), reactive=True)
    assert "wants to matter beyond what they produce" not in reactive, \
        "the reactive baseline prompt omits the need, as upstream does"


@check("sessions.gaps_are_log_uniform_in_range_and_integrity_is_checked")
def _() -> None:
    lo, hi = common.cfg("corpus.gap_days_range")
    gaps = [sessions_mod.sample_gap_days(f"p{i}-s2") for i in range(200)]
    assert all(lo <= g <= hi for g in gaps)
    assert min(gaps) < 7 and max(gaps) > 21, "the distribution must produce both short and long gaps"
    assert "weeks" in sessions_mod.gap_phrase(21.0) and "yesterday" in sessions_mod.gap_phrase(1.0)

    s1 = fake_session("p1-s1", 1)
    s2 = fake_session("p1-s2", 2, gap=14.0)
    s2["profile_snapshot"] = {**fake_profile(), "advancement": {"applied": "intensified"}}
    assert sessions_mod.integrity_check([s1, s2]) == []
    bad = fake_session("p1-s2", 2, gap=-3.0)
    bad["profile_snapshot"] = {**fake_profile(), "advancement": {"applied": "resolved"}}
    problems = sessions_mod.integrity_check([s1, bad])
    assert any("non-positive gap" in p for p in problems)
    missing = fake_session("p1-s2", 2, gap=5.0)
    assert any("no advancement transition" in p for p in sessions_mod.integrity_check([s1, missing]))


@check("sessions.advancement_keeps_node_ids_stable_unless_displaced")
def _() -> None:
    handle = echo("generator")
    try:
        prev = fake_session()
        kept = sessions_mod.advance_profile(handle, fake_profile(), prev, 10.0, "intensified")
        displaced = sessions_mod.advance_profile(handle, fake_profile(), prev, 10.0, "displaced")
    finally:
        handle.release()
    old_ids = [n["node_id"] for n in fake_profile()["need_chain"]]
    assert [n["node_id"] for n in kept["need_chain"]] == old_ids, "an evolving need keeps its identity"
    assert [n["node_id"] for n in displaced["need_chain"]] != old_ids, "a new concern gets new ids"
    assert kept["advancement"]["applied"] == "intensified"
    assert kept["persona_hidden"]["terminal_need"] == kept["terminal_need"], "hidden view must stay in sync"


@check("sessions.truncated_advanced_chain_keeps_the_previous_one")
def _() -> None:
    class Short:
        spec = {"model": "stub"}

        def structured(self, prompt, **kw):
            return {"need_chain": [{"depth": 0, "text": "only one node"}], "parse_failed": False}

    adv = sessions_mod.advance_profile(Short(), fake_profile(), fake_session(), 10.0, "intensified")
    assert [n["text"] for n in adv["need_chain"]] == [n["text"] for n in fake_profile()["need_chain"]]
    assert adv["terminal_need"] == fake_profile()["terminal_need"], "Success ground truth must stay terminal"
    assert adv["advancement"]["chain_kept_from_previous"]


@check("corpus.malformed_profile_json_is_rejected_or_coerced_never_a_crash")
def _() -> None:
    class Bad:
        spec = {"model": "stub"}
        backend_name = "echo"

        def structured(self, prompt, **kw):
            return {"need_chain": ["a bare string", {"depth": "2 - deep", "text": "x"}], "feeling": "meh",
                    "persona_hidden": ["not", "a", "dict"], "memory": "not a list", "parse_failed": False}

    prof = profiles.build_profile(Bad(), {"seed_id": "sd000001", "situation_text": "s"}, "low", False)
    assert [n["depth"] for n in prof["need_chain"]] == [0, 2] and prof["memory"] == []
    ok, why = profiles.check_chain(prof["need_chain"])
    assert not ok and "2 nodes" in why, why
    p = {**fake_profile(), "memory": ["worked late", {"text": "a fight", "when_relative": "today"}]}
    p["persona_hidden"] = {**p["persona_hidden"], "disclosure_triggers": "patience"}
    system = dialogue.simulator_system(p)
    assert "Things that make you open up: patience\n" in system, "a string is one phrase, not its letters"
    assert "worked late" in system and "a fight" in system

    class Echo:
        def __init__(self, reply):
            self.reply = reply

        def chat(self, prompt, **kw):
            return self.reply

    assert not seeds.sustainability_screen(Echo("VERDICT: yes|no\nREASON: ..."), "s")[0], "an echoed template is no pass"
    assert seeds.sustainability_screen(Echo("VERDICT: yes\nREASON: deep enough"), "s")[0]
    assert profiles.check_chain(fake_profile()["need_chain"], Echo("RESTATEMENT: yes|no\nREACHABLE: yes"))[0]


# --------------------------------------------------------------------------- agents


@check("agents.fallback_is_a_bare_reflection")
def _() -> None:
    text = agents.reflective_fallback(fake_turns())
    assert "?" not in text and "you should" not in text.lower()
    assert "you haven't really told anyone how bad the week was" in text, text
    assert agents.reflective_fallback([]) != ""
    turns = fake_turns() + [{"turn_index": 5, "role": "user", "meta": {},
                             "text": "I was exhausted and my boss noticed. Thanks for checking in!"}]
    text = agents.reflective_fallback(turns)
    assert "you were exhausted" in text and "Thanks" not in text, "one clause, second person, no echo"


@check("agents.critic_scores_off_the_scale_are_retried_and_unrecorded_denials_ignored")
def _() -> None:
    assert agents.valid_item_scores({str(i): 0 for i in range(1, 8)}) is None, "template zeros are invalid"
    assert agents.valid_item_scores({str(i): "3" for i in range(1, 8)})["7"] == 3.0
    assert agents.valid_item_scores({"1": 2}) is None, "every item must be scored"

    class Scripted:
        spec = {"model": "scripted"}

        def __init__(self, replies):
            self.replies = list(replies)

        def structured(self, prompt, **_):
            return self.replies.pop(0)

    zeros = {"item_scores": {str(i): 0 for i in range(1, 8)}, "violations": ["denied_inference"]}
    fine = {"item_scores": {str(i): 1 for i in range(1, 8)}, "violations": ["denied_inference"]}
    critic = agents.Critic(Scripted([zeros, fine]), calibration=None, gate=True)
    verdict = critic.review("That sounds like a lot to carry.", fake_turns(), {}, "reflect", "L1")
    assert not verdict.parse_failed and verdict.ip_pred == 0.0, "the retry's valid scores are used"
    assert all(v["category"] != "denied_inference" for v in verdict.grounding_violations), \
        "nothing was denied, so a model-reported denied_inference must not veto the turn"
    assert verdict.decision == "release"


@check("grounding.a_seeker_quote_under_the_wrong_turn_number_is_relocated")
def _() -> None:
    turns = fake_turns()
    good, errors = grounding.validated_spans([{"turn_index": 2, "quote": "Work has been a lot."}], turns)
    assert good and good[0].turn_index == 1, "verbatim seeker words cited under a supporter turn"
    good, errors = grounding.validated_spans([{"turn_index": 1, "quote": "a heavy stretch"}], turns)
    assert not good, "the supporter's own words never ground a claim about the seeker"


@check("agents.critic_gate_uses_the_calibrated_threshold")
def _() -> None:
    handle = echo("critic")
    try:
        strict = conformal.Calibration(alpha=0.1, tau=0.5, delta=0.1, n_calibration=10, lambda_hat=0.0,
                                       lambda_grid=[0.0, 1.0], risk_curve=[])
        loose = conformal.Calibration(alpha=0.1, tau=0.5, delta=0.1, n_calibration=10, lambda_hat=1.0,
                                      lambda_grid=[0.0, 1.0], risk_curve=[])
        turns = fake_turns()
        v_loose = agents.Critic(handle, calibration=loose, gate=True).review(
            "That sounds heavy.", turns, {}, "reflect", "L0")
        assert v_loose.decision == "release"
        v_strict = agents.Critic(handle, calibration=strict, gate=True).review(
            "That sounds heavy.", turns, {}, "reflect", "L0")
        assert v_strict.decision in ("revise", "fallback"), v_strict.decision
        ungated = agents.Critic(handle, calibration=strict, gate=False).review(
            "You should tell your manager you need a break.", turns, {}, "advise", "L0")
        assert ungated.decision == "release", "the ungated cell releases everything by design"
        assert ungated.nonconformity > 0
    finally:
        handle.release()


@check("agents.denied_inference_forces_revision_even_below_threshold")
def _() -> None:
    handle = echo("critic")
    try:
        mem = memory_mod.NeedStateMemory("p000001")
        node = mem.propose("wants praise from the boss", [span(1, "Work has been a lot")], 0.6, depth=2)
        mem.disconfirm(node.node_id, span(1, "I'm fine though"))
        loose = conformal.Calibration(alpha=0.1, tau=0.5, delta=0.1, n_calibration=10, lambda_hat=1.0,
                                      lambda_grid=[0.0, 1.0], risk_curve=[])
        verdict = agents.Critic(handle, calibration=loose, gate=True).review(
            "I still think you want praise from the boss.", fake_turns(), {}, "name it", "L3", memory=mem)
        assert verdict.decision != "release", "a denied inference must never be released"
        assert any(v["category"] == "denied_inference" for v in verdict.grounding_violations)
    finally:
        handle.release()


@check("agents.pipeline_step_produces_an_inspectable_turn")
def _() -> None:
    handle = echo("generator")
    try:
        mem = memory_mod.NeedStateMemory("p000001")
        views = {r: handle.view(r) for r in ("analyzer", "strategist", "critic", "generator")}
        pipe = agents.AgentPipeline(views["generator"], memory=mem, gate=False, views=views)
        result = pipe.step(fake_turns(), session_id="p000001-s1", phase="listening")
    finally:
        handle.release()
    assert result.text.strip()
    assert result.permitted_rung in pacing.LADDER
    assert "grounding_report" in result.analyzer
    assert result.strategist["requested_rung"] in pacing.at_most(result.permitted_rung)
    assert result.path in ("released", "revised", "fallback")
    assert isinstance(result.memory_writes, list)
    assert not hasattr(result, "readiness"), "readiness is Enhancement 5 and was removed"


@check("agents.memory_writes_refuse_ungrounded_claims")
def _() -> None:
    handle = echo("generator")
    try:
        mem = memory_mod.NeedStateMemory("p000001")
        pipe = agents.AgentPipeline(handle, memory=mem, gate=False)
        analyzer = {"implicit_needs": [
            {"text": "wants rest", "depth": 1, "confidence": 0.7,
             "evidence_spans": [span(1, "Work has been a lot")]},
            {"text": "wants a new job", "depth": 2, "confidence": 0.0, "evidence_spans": []},
        ]}
        writes = pipe.update_memory(analyzer, "p000001-s1")
    finally:
        handle.release()
    actions = {w["claim"]: w["action"] for w in writes}
    assert actions["wants rest"] == "proposed"
    assert actions["wants a new job"] == "refused"
    assert len(mem.nodes) == 1


# --------------------------------------------------------------------------- metrics and counterfactual


@check("metrics.turn_sampling_is_stratified_by_position")
def _() -> None:
    session = fake_session()
    session["turns"] = ([{"turn_index": 0, "role": "supporter", "text": "open",
                          "meta": {"source": "opener_pool"}}]
                        + [{"turn_index": i, "role": "user" if i % 2 else "supporter",
                            "text": f"t{i}", "meta": {}} for i in range(1, 17)])
    idx = metrics.sample_turn_indices(session)
    assert idx and len(set(idx)) == len(idx), idx
    assert all(next(t for t in session["turns"] if t["turn_index"] == i)["role"] == "supporter"
               for i in idx)
    assert 0 not in idx, "the scripted opener is not scored"


@check("metrics.aggregates_report_components_and_failures")
def _() -> None:
    ip_rows = [{"normalized": 0.2, "mean": 2.2, "items": {"1": 2, "2": 2, "3": 5}, "judge_model": "J",
                "context_redactions": ["terminal_need", "resistance_level"], "parse_failed": False},
               {"normalized": 0.4, "mean": 3.4, "items": {"1": 3, "2": 3, "3": 4}, "judge_model": "J",
                "context_redactions": ["terminal_need"], "parse_failed": False},
               {"parse_failed": True, "normalized": 0, "mean": 0, "items": {}, "judge_model": "J",
                "context_redactions": []}]
    agg = metrics.aggregate(ip_rows, "ip")
    assert agg["n"] == 2 and agg["parse_failures"] == 1
    assert abs(agg["mean_normalized"] - 0.3) < 1e-9
    assert abs(agg["item3_mean"] - 4.5) < 1e-9
    pri_rows = [{"normalized": 0.5, "mean": 4.0, "judge_model": "J", "context_redactions": [],
                 "parse_failed": False,
                 "items": {"1": 5, "2": 3, "3": 6, "4": 4, "5": 5, "6": 2, "7": 1}}]
    pri = metrics.aggregate(pri_rows, "pri")
    assert abs(pri["anger_mean"] - 4.0) < 1e-9 and abs(pri["negative_cognition_mean"] - 5.0) < 1e-9


@check("metrics.published_dimensions_match_their_item_groupings")
def _() -> None:
    def row(items: dict) -> dict:
        return {"items": {str(k): v for k, v in items.items()}, "normalized": 0.5, "mean": 4.0,
                "judge_model": "J", "context_redactions": [], "parse_failed": False}

    crs = metrics.crs_dimensions([row({1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 2, 7: 6, 8: 2, 9: 2, 10: 6})])
    assert abs(crs["affective_improvement"] - 6.0) < 1e-9
    assert abs(crs["negative_helper"] - 2.0) < 1e-9, "only items 6, 8 and 9 are negatively worded"
    assert abs(crs["negative_helper_all_items_reversed"] - 2.0) < 1e-9, "7 and 10 must be reverse-scored"

    rac = metrics.rac_dimensions([row({i: 5 for i in range(1, 17)})])
    assert abs(rac["supportiveness"] - 5.0) < 1e-9 and abs(rac["management"] - 5.0) < 1e-9
    assert set(metrics.RAC_SUPPORTIVENESS) & set(metrics.RAC_MANAGEMENT) == set(), "dimensions overlap"

    basic = metrics.basic_breakdown([row({1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7})])
    assert basic["basic_avg_100"] == 100.0, basic
    assert basic["fluency"] == 7.0 and basic["skillfulness_100"] == 100.0
    worst = metrics.basic_breakdown([row({i: 1 for i in range(1, 7)})])
    assert worst["basic_avg_100"] == 0.0
    assert judge_mod.expected_items("basic") == [1, 2, 3, 4, 5, 6], \
        "the basic scale must carry all six published metrics"
    agg = metrics.aggregate([row({i: 5 for i in range(1, 7)})], "basic")
    assert "basic_avg_100" in agg, "aggregate() must expose the published breakdown"


@check("convert_corpus.external_corpora_become_response_only_sft_rows")
def _() -> None:
    esconv = [{"dialog": [{"speaker": "seeker", "text": "I can't sleep."},
                          {"speaker": "supporter", "text": "That sounds exhausting."},
                          {"speaker": "seeker", "text": "It is."},
                          {"speaker": "supporter", "text": "What has the week been like?"}]}]
    path = TMP / "esconv.json"
    common.write_json(path, esconv)
    stats = convert_corpus.convert("esconv", path, out_dir=TMP / "sft")
    assert stats["dialogues_kept"] == 1 and stats["annotations"] is False
    rows = common.read_jsonl(TMP / "sft" / "corpus_esconv_train.jsonl") + \
        common.read_jsonl(TMP / "sft" / "corpus_esconv_val.jsonl")
    assert rows, "no training rows produced"
    assert all("<analysis>" not in r["target"] for r in rows), \
        "an unannotated corpus must train response-only, like our w/o thoughts arm"
    assert all(r["system"] == build_sft.SUPPORTER_SYSTEM for r in rows), "same prompt shape as our arms"
    assert build_sft.parse_target(rows[0]["target"])["response"]

    extes = [{"content": [{"User": "I moved cities."}, {"AI": "That is a big change."},
                          {"User": "Yeah."}, {"AI": "What has been hardest?"}]}]
    path = TMP / "extes.json"
    common.write_json(path, extes)
    stats = convert_corpus.convert("extes", path, out_dir=TMP / "sft")
    assert stats["dialogues_kept"] == 1, stats
    assert convert_corpus.normalize_role("Seeker") == "user"
    assert convert_corpus.normalize_role("counselor") == "supporter"
    assert convert_corpus.normalize_role("narrator") is None


@check("evaluate.arms_resolve_their_own_adapter")
def _() -> None:
    spec = evaluate_mod.resolve_arm("corpus_esconv")
    assert spec["supporter"] == "corpus_esconv"
    assert "adapter" in spec, "an arm must carry its adapter path so no config edit is needed per arm"
    arms = common.load_config("arms")
    supporters = {a.get("supporter", arms["defaults"]["supporter"]) for a in arms["arms"].values()}
    missing = [s for s in supporters if s not in arms["adapters"]]
    assert not missing, f"supporters with no entry in the adapters map: {missing}"
    extes_arm = evaluate_mod.resolve_arm("corpus_ours_extes_profiles")
    assert extes_arm["eval_set_spec"]["in_distribution"] is False


@check("counterfactual.pri_is_a_paired_difference_with_a_ci")
def _() -> None:
    rows = []
    for k in range(5):
        rows.append({"session_id": "s", "turn_index": 2, "condition": "factual", "rollout_index": k,
                     "reactance_score": 0.6, "parse_failed": False, "judge_model": "J"})
        rows.append({"session_id": "s", "turn_index": 2, "condition": "control", "rollout_index": k,
                     "reactance_score": 0.2, "parse_failed": False, "judge_model": "J"})
    out = counterfactual.aggregate_pri(rows, resamples=200)
    assert abs(out["pri"] - 0.4) < 1e-9 and out["n_pairs"] == 5
    assert out["ci_low"] <= out["pri"] <= out["ci_high"]
    assert out["n_turns_scored"] == 1
    unpaired = counterfactual.aggregate_pri([rows[0]], resamples=50)
    assert unpaired["pri"] is None, "an unpaired rollout cannot yield an attribution"


# --------------------------------------------------------------------------- report


@check("report.guards_refuse_missing_n_shared_judges_and_ood_guarantees")
def _() -> None:
    try:
        report.guard_sample_size({"mean": 0.5}, "test")
    except report.ReportGuardError:
        pass
    else:
        raise AssertionError("a cell without n was accepted")

    models = common.load_config("models")
    try:
        report.guard_judge_separation(models["roles"]["generator"]["model"], "test")
    except report.ReportGuardError:
        pass
    else:
        raise AssertionError("a judge sharing the generator's model was accepted")
    report.guard_judge_separation(models["roles"]["judge"]["model"], "test")

    try:
        report.guard_distribution_language("we guarantee the bound here", False, "test")
    except report.ReportGuardError:
        pass
    else:
        raise AssertionError("an out-of-distribution guarantee claim was accepted")
    report.guard_distribution_language("we guarantee the bound here", True, "test")
    assert "NOT YET MEASURED" in report.guard_agreement("ip", None, "test")
    assert report.guard_agreement("aels", None, "test") == ""


@check("report.two_by_two_effects_and_holm_are_correct")
def _() -> None:
    eff = report.two_by_two({"mono_ungated": 0.40, "mono_gated": 0.45,
                             "dec_ungated": 0.42, "dec_gated": 0.60})
    assert abs(eff["effect_decomposition"] - 0.085) < 1e-9, eff
    assert abs(eff["effect_gate"] - 0.115) < 1e-9, eff
    assert abs(eff["interaction"] - 0.13) < 1e-9, eff
    assert "error" in report.two_by_two({"mono_ungated": 0.4})
    corrected = report.holm({"a": 0.001, "b": 0.04, "c": 0.9}, alpha=0.05)
    assert corrected["a"]["significant"] and not corrected["c"]["significant"]
    diff = report.paired_bootstrap_diff({"p1": 0.6, "p2": 0.7, "p3": 0.8},
                                        {"p1": 0.4, "p2": 0.5, "p3": 0.6}, resamples=200)
    assert abs(diff["diff"] - 0.2) < 1e-9 and diff["n"] == 3
    assert report.paired_bootstrap_diff({"p1": 1.0}, {"p9": 1.0})["n"] == 0
    table = report.markdown_table([{"a": 1, "b": 2}], ["a", "b"])
    assert table.count("|") == 9 and "---" in table


# --------------------------------------------------------------------------- evaluate wiring


@check("evaluate.arms_resolve_over_defaults_and_memory_kinds_map")
def _() -> None:
    spec = evaluate_mod.resolve_arm("cellD_dec_gated")
    assert spec["architecture"] == "decomposed" and spec["gate"] == "conformal"
    assert spec["memory"] == "needstate", "defaults must fill switches the arm does not name"
    assert spec["eval_set_spec"]["in_distribution"] is True
    ood = evaluate_mod.resolve_arm("corpus_ours_extes_profiles")
    assert ood["eval_set_spec"]["in_distribution"] is False
    try:
        evaluate_mod.resolve_arm("no_such_arm")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown arm was accepted")
    handle = echo("generator")
    try:
        kinds = {"needstate": memory_mod.NeedStateMemory, "none": type(None)}
        assert isinstance(evaluate_mod.make_memory("needstate", "p1", handle), memory_mod.NeedStateMemory)
        for kind in ("none", "summary", "dense", "event"):
            mem = evaluate_mod.make_memory(kind, "p1", handle)
            assert hasattr(mem, "render_brief") and hasattr(mem, "brief")
            assert mem.forbidden_inferences() == []
    finally:
        handle.release()


@check("evaluate.baseline_memories_share_the_needstate_surface")
def _() -> None:
    from baselines.memory_dense import DenseMemory
    from baselines.memory_event import EventMemory
    from baselines.memory_none import NoMemory
    from baselines.memory_summary import SummaryMemory

    session = fake_session()
    dense = DenseMemory("p1")
    dense.ingest_session(session)
    text = dense.render_brief(query="how bad was the week")
    assert "retrieval" in text and "seeker turn" in text
    event = EventMemory("p1")
    event.ingest_session(session)
    event.advance_clock(14.0)
    assert "event level" in event.render_brief(query="week")
    assert NoMemory("p1").render_brief().startswith("[memory]")
    summ = SummaryMemory("p1")
    summ.update_from_session(session)
    assert "running summary" in summ.render_brief()
    for mem in (dense, event, NoMemory("p1"), summ):
        assert mem.propose("x", [span(1, "Work has been a lot")], 0.5) is None, \
            "baselines must not silently accept belief writes"


@check("evaluate.monolithic_listener_matches_the_pipeline_contract")
def _() -> None:
    from baselines.listener_mono import MonolithicListener

    handle = echo("generator")
    try:
        mono = MonolithicListener(handle, memory=memory_mod.NeedStateMemory("p1"), gate=False)
        result = mono.step(fake_turns(), session_id="p1-s1")
    finally:
        handle.release()
    assert isinstance(result, agents.TurnResult)
    assert result.text.strip() and result.permitted_rung in pacing.LADDER
    assert result.memory_writes == [], "the monolithic arm writes no inspectable analyzer state"


@check("sessions.memory_corpus_trains_on_the_brief_the_generator_saw")
def _() -> None:
    from baselines.listener_mono import MonolithicListener

    need = {"text": "wants to matter beyond what they produce", "depth": 2, "confidence": 0.6,
            "evidence_spans": [span(1, "Work has been a lot")]}

    def writer(handle):
        w = sessions_mod.memory_writer(handle, "p000001")
        w.analyze = lambda turns, sid, block: {"emotional_state": {}, "implicit_needs": [need]}
        return w

    handle = echo("generator")
    try:
        w = writer(handle)
        session = dialogue.generate_session(fake_profile(), "p000001-s2", handle, handle, n_supporter_turns=4,
                                            memory_fn=sessions_mod.turn_brief(w, "p000001-s2"))
        plain = dialogue.generate_session(fake_profile(), "p000001-s1", handle, handle, n_supporter_turns=3)
        replayed = sessions_mod.replay(writer(handle), json.loads(json.dumps(session)))
        mono = MonolithicListener(handle, memory=w.memory, gate=False, memory_writer=w)
        writes = mono.step(fake_turns(), session_id="p1-s3").memory_writes
    finally:
        handle.release()
    sup = [t for t in session["turns"] if t["role"] == "supporter" and t["turn_index"]]
    blocks = [t["memory_block"] for t in sup]
    assert "first session" in blocks[0] and need["text"] in blocks[1], "brief before the turn, writes after"
    assert all("memory_block" not in t for t in plain["turns"]), "without memory_fn generation is unchanged"
    assert [t.get("memory_block") for t in replayed["turns"]] == [t.get("memory_block") for t in session["turns"]], \
        "a replayed (resumed) session must rebuild the same briefs"
    for t in sup:
        t["analysis"], t["strategy"] = "worn down", "reflect"
    inputs = [r["input"] for r in build_sft.examples_from_session(session)]
    assert all(b in i and dialogue.MEMORY_NOTE.strip() not in i for b, i in zip(blocks, inputs)), \
        "training input = the brief alone, without the generator-only note"
    assert writes and writes[0]["action"] in ("proposed", "blocked"), "memory_writer must write the memory"


# --------------------------------------------------------------------------- human evaluation


@check("human_eval.agreement_statistics_behave_on_known_inputs")
def _() -> None:
    assert abs(agreement_mod.weighted_kappa([1, 3, 5, 7], [1, 3, 5, 7]) - 1.0) < 1e-9
    disagree = agreement_mod.weighted_kappa([1, 1, 1, 7], [7, 7, 7, 1])
    assert disagree is not None and disagree < 0
    assert agreement_mod.weighted_kappa([], []) is None
    assert abs(agreement_mod.spearman_rho([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9
    assert abs(agreement_mod.spearman_rho([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-9
    alpha = agreement_mod.krippendorff_alpha([[5, 5, 5], [2, 2, 2], [6, 6, 6], [1, 1, 1]])
    assert alpha is not None and alpha > 0.99
    assert agreement_mod.mean_score({"1": 4, "2": None}) == 4.0
    assert sample_mod.decile(0.0) == 0 and sample_mod.decile(0.99) == 9 and sample_mod.decile(1.0) == 9


@check("human_eval.sampling_is_stratified_and_blinded")
def _() -> None:
    rows = [{"item_id": f"s{i}#2", "arm": f"arm{i % 2}", "resistance_level": ["low", "medium", "high"][i % 3],
             "judge_score": (i % 10) / 10, "judge_decile": i % 10, "dialogue_prefix": "p",
             "target_turn": "t", "metric": "ip"} for i in range(60)]
    picked = sample_mod.stratify(rows, 12)
    assert len(picked) == 12 and len({p["item_id"] for p in picked}) == 12
    assert len({p["arm"] for p in picked}) == 2, "both arms must appear"
    out = TMP / "forms"
    sample_mod.write_sheets(picked, out)
    sheets = [common.read_json(out / f"rater_{r}.json") for r in ("a", "b")]
    assert all("arm" not in item for sheet in sheets for item in sheet), "sheets must be blind to arm"
    assert [i["item_id"] for i in sheets[0]] != [i["item_id"] for i in sheets[1]], \
        "raters must see a different order"
    assert set(sheets[0][0]["scores"]) == {str(i) for i in range(1, 8)}


@check("external.mapped_sets_flag_what_they_cannot_supply")
def _() -> None:
    extes = external.map_extes([{"scene": "moved city for work", "emotion_type": "lonely"},
                                {"description": "argument with my sister"}])
    assert len(extes) == 2
    assert all(r["terminal_need"] == "" for r in extes)
    assert all(r["success_rate_applicable"] is False for r in extes), \
        "Success Rate must not be scored against a need we invented"
    assert "resistance_level" in extes[0]["generated_fields"]
    assert {r["resistance_level"] for r in extes} <= set(external.RESISTANCE_CYCLE)
    mem = external.map_es_memeval([{"persona": {"summary": "lonely after a move"},
                                    "sessions": ["s1", "s2"], "target_need": "belonging"},
                                   {"persona": {}, "sessions": []}])
    assert mem[0]["success_rate_applicable"] is True and mem[0]["n_reference_sessions"] == 2
    assert mem[1]["success_rate_applicable"] is False, "a row without an annotated need cannot be scored"


@check("pipeline.plan_covers_every_stage_and_arm")
def _() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_all

    arms = ["cellC_dec_ungated", "cellD_dec_gated", "gate_alpha_005", "mem_dense"]
    run_dir = TMP / "runrun"
    plan = run_all.build_plan("smoke", "echo", run_dir, arms, train=False, only=None)
    stages = [stage for stage, _, _ in plan]
    for expected in ("check", "corpus", "sft", "calib", "eval", "score", "human", "report"):
        assert expected in stages, f"stage {expected} missing from the plan"
    assert "train" not in stages, "training must be opt-in"
    assert stages.index("corpus") < stages.index("sft") < stages.index("calib") < stages.index("eval")
    assert stages.index("eval") < stages.index("score") < stages.index("report")

    evals = [cmd for stage, cmd, _ in plan if stage == "eval"]
    assert len(evals) == len(arms), "every arm must be rolled out"
    scores = [cmd for stage, cmd, _ in plan if stage == "score"]
    assert len(scores) == 2 * len(arms), "each arm needs the instruments and the counterfactual PRI"

    calibs = [cmd for stage, cmd, _ in plan if stage == "calib"]
    assert any("evaluate.py" in " ".join(c) for c in calibs), "the gate needs ungated dialogues first"
    assert any("--alpha" in c and "0.05" in c for c in calibs), "an arm's own alpha must be calibrated"
    assert run_all.calibration_name(None) == "calibration.json"
    assert run_all.calibration_name(0.05) == "calibration_alpha_0.05.json"

    with_train = run_all.build_plan("full", None, run_dir, arms, train=True, only=None)
    trained = [" ".join(c) for st, c, _ in with_train if st == "train"]
    for arm in ("with_thoughts", "wo_thoughts"):
        assert any(f"--arm {arm} " in t for t in trained), f"SFT arm {arm} must be trained"
    subset = run_all.build_plan("pilot", "echo", run_dir, arms, train=False, only=["report"])
    assert [st for st, _, _ in subset] == ["report"]
    smoke_eval = [c for st, c, _ in plan if st == "eval"][0]
    assert "--limit" in smoke_eval and "--backend" in smoke_eval, "scale and backend must reach the stages"


@check("scope.enhancement_5_is_absent")
def _() -> None:
    """The modified SOP has four enhancements. Nothing may reintroduce the pacing machinery."""
    import importlib

    for gone in ("readiness", "fit_readiness", "teacher_rollouts", "baselines.pacing_fixed",
                 "baselines.pacing_greedy"):
        try:
            importlib.import_module(gone)
        except ImportError:
            continue
        raise AssertionError(f"module `{gone}` is Enhancement 5 and should have been removed")
    for gone in ("AdaptivePolicy", "FixedRatioPolicy", "GreedyPolicy", "PrivilegedTeacher",
                 "build_distillation_dataset", "rung_entropy", "make_policy", "rung_from_readiness"):
        assert not hasattr(pacing, gone), f"pacing.{gone} is Enhancement 5 and should have been removed"
    assert "readiness" not in (ROOT / "src" / "agents.py").read_text(encoding="utf-8")
    assert "pacing_distill" not in (ROOT / "src" / "train.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- cross-file consistency


@check("consistency.ip_items_match_between_the_judge_and_the_critic")
def _() -> None:
    import re

    def items(text: str) -> list[str]:
        block = text.split("[Rating Scale]")[0] if "[Rating Scale]" in text else text
        block = block.split("Return JSON only")[0]
        return [re.sub(r"\s+", " ", m.strip().lower().rstrip("."))
                for m in re.findall(r"(?m)^\s*\d\.\s+(.*)$", block)]

    metric_items = items(common.read_prompt("metric_ip.md"))
    critic_items = items(common.read_prompt("agent_critic.md"))
    assert len(metric_items) == 7 and len(critic_items) == 7, (len(metric_items), len(critic_items))
    for a, b in zip(metric_items, critic_items):
        core_a = a.replace("the supporter ", "").replace("the supporter's ", "")
        core_b = b.replace("the supporter ", "").replace("the supporter's ", "")
        assert memory_mod.similarity(core_a, core_b) > 0.6, f"IP item drift:\n  judge: {a}\n  critic: {b}"


@check("consistency.configs_reference_only_existing_files_and_roles")
def _() -> None:
    arms = common.load_config("arms")
    models = common.load_config("models")
    for name, spec in arms["arms"].items():
        merged = {**arms["defaults"], **spec}
        assert merged["eval_set"] in arms["eval_sets"], f"{name} names an unknown eval set"
        assert merged["memory"] in ("none", "summary", "dense", "event", "needstate"), name
        assert merged["architecture"] in ("monolithic", "decomposed"), name
        assert "pacing" not in merged, f"{name} still carries an Enhancement 5 pacing switch"
    for role in ("generator", "simulator", "analyzer", "strategist", "critic", "supporter", "judge"):
        assert role in models["roles"], f"configs/models.yaml is missing role '{role}'"
    judge_family = models["roles"]["judge"]["model"].split("/")[0].lower()
    gen_family = models["roles"]["generator"]["model"].split("/")[0].lower()
    assert judge_family != gen_family, "the judge must come from a different model family"
    assert models["backend"] in ("echo", "ollama", "transformers", "vllm")


@check("consistency.every_prompt_file_referenced_in_code_exists")
def _() -> None:
    import re

    names = set()
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "human_eval").glob("*.py")):
        names |= set(re.findall(r"read_prompt\(\s*[\"']([^\"']+)[\"']", path.read_text(encoding="utf-8")))
    names |= set(judge_mod.SCALE_PROMPTS.values())
    missing = [n for n in sorted(names) if not (ROOT / "prompts" / n).exists()]
    assert not missing, f"prompts referenced but absent: {missing}"


@check("consistency.config_knobs_read_by_code_exist_in_default_yaml")
def _() -> None:
    import re

    sentinel = object()
    missing = []
    for path in list((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for key in re.findall(r"cfg\(\s*\"([a-z0-9_.]+)\"", text):
            if common.cfg(key, default=sentinel) is sentinel:
                missing.append(f"{path.name}:{key}")
    assert not missing, f"config keys read but not defined: {sorted(set(missing))}"


@check("consistency.scale_prompts_all_expose_the_placeholders_the_judge_fills")
def _() -> None:
    for scale, filename in judge_mod.SCALE_PROMPTS.items():
        text = common.read_prompt(filename)
        assert "{info}" in text and "{diag}" in text, f"{filename} is missing info/diag"
        if scale in ("ip", "pri"):
            assert "{target}" in text, f"{filename} must mark the single turn being scored"
        assert "[Response Format]" in text and "Question number: Score" in text, filename


# --------------------------------------------------------------------------- runner


def main(argv: list[str]) -> int:
    wanted = [a for a in argv[1:] if not a.startswith("-")]
    selected = [(n, f) for n, f in CHECKS if not wanted or any(w in n for w in wanted)]
    failures, skips = [], []
    print(f"running {len(selected)} checks against the echo backend\n")
    for name, fn in selected:
        llm_mod.release_all()
        try:
            fn()
        except Skip as exc:
            skips.append((name, str(exc)))
            print(f"SKIP {name} ({exc})")
        except AssertionError as exc:
            failures.append((name, "assertion", str(exc) or "assertion failed",
                             traceback.format_exc(limit=3)))
            print(f"FAIL {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, type(exc).__name__, str(exc), traceback.format_exc(limit=4)))
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    llm_mod.release_all()

    print("\n" + "=" * 72)
    print(f"{len(selected) - len(failures) - len(skips)} passed, {len(failures)} failed, "
          f"{len(skips)} skipped")
    if skips:
        print("\nskipped (optional dependency missing):")
        for name, why in skips:
            print(f"  - {name}: {why}")
    if failures:
        print("\nissues found:")
        for name, kind, message, tb in failures:
            print(f"\n--- {name} [{kind}]\n{message}\n{tb}")
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
