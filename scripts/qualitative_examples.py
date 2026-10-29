"""Qualitative examples for the report, chosen by fixed rules (stated with each section), not by hand. (CPU only)

    python scripts/qualitative_examples.py --out temp_results/22_qualitative_examples
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "runs/baselines_1000"


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def scores(run: Path, arm: str, scale: str) -> dict[str, float]:
    return {r["target"]: r["normalized"] for r in jl(run / "scores" / arm / f"{scale}.jsonl")
            if r.get("normalized") is not None}


def cut(text: str, n: int = 320) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def prev_user(turns: list[dict], idx: int) -> str:
    return next((t["text"] for t in reversed(turns) if t["turn_index"] < idx and t["role"] == "user"), "")


def gate_examples() -> list[str]:
    out = ["## 1. The gate at work (cell D, four agents + conformal gate) - SOP Enhancement 2",
           "Rule: the first 3 revised turns and the first 2 fallback turns, in session-id order, among turns the judge "
           "scored. Only the final reply is logged, not the rejected drafts; the critic's last feedback is shown.", ""]
    ip = scores(B, "cellD_dec_gated", "ip")
    want = {"revised": 3, "fallback": 2}
    for s in sorted(jl(B / "dialogues/cellD_dec_gated.jsonl"), key=lambda s: s["session_id"]):
        arts = {a["turn_index"]: a for a in s.get("agent_artifacts", [])}
        for t in s["turns"]:
            path = t.get("meta", {}).get("path")
            key = f"{s['session_id']}#{t['turn_index']}"
            if path in want and want[path] > 0 and key in ip:
                want[path] -= 1
                fb = (arts.get(t["turn_index"], {}).get("critic") or {}).get("feedback", "")
                out += [f"**{key}** - {path}, {t['meta'].get('revisions', 0)} revision(s), judge IP {ip[key]:.3f}",
                        f"- seeker: {cut(prev_user(s['turns'], t['turn_index']))}",
                        f"- supporter (sent): {cut(t['text'])}",
                        f"- critic's last feedback: {cut(fb, 240) or '(none)'}", ""]
        if not any(want.values()):
            break
    return out


def steering_examples() -> list[str]:
    out = ["## 2. Steering vs its control on the same context - added comparison (activation steering)",
           "Rule: the steered and unsteered arms share session seeds, so the first supporter reply of a session answers "
           "the same seeker turn; the first 4 sessions (by id) where both first replies were judge-scored.", ""]
    run = ROOT / "runs/steering"
    a = {s["session_id"]: s for s in jl(run / "dialogues/steer_mono_L12_c0.jsonl")}
    b = {s["session_id"]: s for s in jl(run / "dialogues/steer_mono_L12_c0.2.jsonl")}
    ipa, ipb = scores(run, "steer_mono_L12_c0", "ip"), scores(run, "steer_mono_L12_c0.2", "ip")
    n = 0
    for sid in sorted(set(a) & set(b)):
        ta, tb = (next(t for t in x["turns"] if t["role"] == "supporter" and t["turn_index"] > 0) for x in (a[sid], b[sid]))
        k = f"{sid}#{ta['turn_index']}"
        if prev_user(a[sid]["turns"], ta["turn_index"]) != prev_user(b[sid]["turns"], tb["turn_index"]) \
                or k not in ipa or k not in ipb:
            continue
        out += [f"**{sid}**", f"- seeker: {cut(prev_user(a[sid]['turns'], ta['turn_index']))}",
                f"- unsteered (IP {ipa[k]:.3f}): {cut(ta['text'])}",
                f"- steered L12 x0.2 (IP {ipb[k]:.3f}): {cut(tb['text'])}", ""]
        n += 1
        if n == 4:
            break
    return out


def sycophancy_examples() -> list[str]:
    out = ["## 3. Risky plans: where fine-tuning and steering go along - added comparison (LLM psychology probes)",
           "Rule: the 3 risky-plan probes with the largest SFT-minus-base sycophancy score (items 1-3, 0-1), with all "
           "three replies.", ""]
    d = ROOT / "runs/psych/sycophancy"
    rows = {name: {r["probe_id"]: r for r in jl(d / f"scores_{name}.jsonl") if r.get("syco")} for name in ("base", "sft", "steered")}
    syc = lambda r: st.mean((r["syco"][str(i)] - 1) / 6 for i in (1, 2, 3))
    probes = [p for p in rows["sft"] if p.startswith("risky_plan") and all(p in rows[n] for n in rows)]
    for p in sorted(probes, key=lambda p: syc(rows["sft"][p]) - syc(rows["base"][p]), reverse=True)[:3]:
        out += [f"**{p}** - seeker: {cut(rows['base'][p]['turns'][1]['text'])}"]
        out += [f"- {n} (sycophancy {syc(rows[n][p]):.2f}): {cut(rows[n][p]['text'])}" for n in ("base", "sft", "steered")]
        out.append("")
    return out


def resistance_examples() -> list[str]:
    out = ["## 4. Turns that provoked resistance (cell C, counterfactual PRI) - SOP Enhancement 3",
           "Rule: the 3 supporter turns with the largest factual-minus-control reactance (mean over rollouts): the "
           "seeker's simulated reply to the real turn vs to a neutral reflective control in the same place.", ""]
    rolls = defaultdict(lambda: defaultdict(list))
    for r in jl(B / "scores/cellC_dec_ungated/pri_rollouts.jsonl"):
        if r.get("reactance_score") is not None:
            rolls[(r["session_id"], r["turn_index"])][r["condition"]].append(r)
    gap = {k: st.mean(x["reactance_score"] for x in v["factual"]) - st.mean(x["reactance_score"] for x in v["control"])
           for k, v in rolls.items() if v["factual"] and v["control"]}
    for k in sorted(gap, key=gap.get, reverse=True)[:3]:
        f, c = rolls[k]["factual"][0], rolls[k]["control"][0]
        out += [f"**{k[0]}#{k[1]}** - reactance factual minus control {gap[k]:+.3f}",
                f"- supporter (real): {cut(f['supporter_text'])}", f"- seeker replied: {cut(f['user_reply'])}",
                f"- control turn: {cut(c['supporter_text'])}", f"- seeker replied: {cut(c['user_reply'])}", ""]
    return out


def memory_examples() -> list[str]:
    out = ["## 5. Need-state memory across sessions (mem_needstate) - SOP Enhancement 4",
           "Rule: among people with a session 2, the 2 with the highest and the 2 with the lowest session-2 Success "
           "(ties by id): what the memory handed session 2, and the supporter's first reply there.", ""]
    run = ROOT / "runs/memory_v2"
    suc = scores(run, "mem_needstate", "success")
    by = defaultdict(dict)
    for s in jl(run / "dialogues/mem_needstate.jsonl"):
        by[s["profile_id"]][s["session_index"]] = s
    cand = sorted((suc[p[2]["session_id"]], pid) for pid, p in by.items() if 1 in p and 2 in p and p[2]["session_id"] in suc)
    for score, pid in cand[-2:][::-1] + cand[:2]:
        s1, s2 = by[pid][1], by[pid][2]
        first = next(t for t in s2["turns"] if t["role"] == "supporter" and t["turn_index"] > 0)
        out += [f"**{pid}** - session 2 Success {score:.3f}, {s2.get('gap_days_from_prev') or 0:.0f} days after session 1",
                f"- memory brief after session 1: {cut(s1.get('memory_brief_after', ''), 400)}",
                f"- session 2, seeker: {cut(prev_user(s2['turns'], first['turn_index']))}",
                f"- session 2, supporter: {cut(first['text'])}", ""]
    return out


def language_drift() -> list[str]:
    import re

    cjk = re.compile(r"[\u4e00-\u9fff]")
    out = ["## 6. Data note: Qwen sometimes switches to Chinese",
           "Example 4's control reply above is in Chinese: Qwen2.5 (simulator and supporter) drifts into Chinese now and "
           "then. Share of turns with Chinese characters in the main multi-session arms, and of simulated seeker "
           "replies in the counterfactual rollouts:", "", "arm | dialogue turns | supporter turns | rollout seeker replies"]
    arms = [(B, "cellA_mono_ungated"), (B, "cellB_mono_gated"), (B, "cellC_dec_ungated"), (B, "cellD_dec_gated"),
            (ROOT / "runs/memory_v2", "mem_none"), (ROOT / "runs/memory_v2", "mem_needstate"),
            (ROOT / "runs/steering", "steer_mono_L12_c0"), (ROOT / "runs/steering", "steer_mono_L12_c0.2"),
            (ROOT / "runs/sft_agents", "sft_dec_gated_mem")]
    for run, arm in arms:
        turns = [t for s in jl(run / "dialogues" / f"{arm}.jsonl") for t in s["turns"]]
        if not turns:
            continue
        sup = [t for t in turns if t["role"] == "supporter"]
        rolls = jl(run / "scores" / arm / "pri_rollouts.jsonl")
        share = lambda xs, key: f"{sum(bool(cjk.search(x.get(key) or '')) for x in xs) / len(xs):.2%}" if xs else "n/a"
        out.append(f"{arm} | {share(turns, 'text')} | {share(sup, 'text')} | {share(rolls, 'user_reply')}")
    out += ["", "Rare: at most about 1 % of dialogue turns and under 3 % of rollout replies in any arm. The judge "
            "scores such turns as they are, so it adds a little noise to every arm rather than favouring one.", ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/22_qualitative_examples"))
    out = Path(ap.parse_args().out)
    lines = ["# Qualitative examples", "", "Chosen by the fixed rule given in each section (scripts/qualitative_examples.py), "
             "so they illustrate the numbers rather than argue for them. Judge scores are 0-1; IP: higher = more "
             "intrusive.", ""]
    for part in (gate_examples, steering_examples, sycophancy_examples, resistance_examples, memory_examples,
                 language_drift):
        lines += part()
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
