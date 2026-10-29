"""Activation steering: the comparison approach to our pipeline.

Our approach controls intrusiveness with structure: four agents, a conformal gate on the critic, need-state memory.
Steering controls it inside the model: the one-call listener (cell A) is left as it is, and while it writes a reply
a fixed direction is added to its residual stream that pushes the reply away from intrusive and toward gentle.
No training, no extra calls, no gate.

Method (contrastive activation addition with an instruction contrast, as in ActAdd / CAA):
  build   N training-split contexts (data/sft/wo_thoughts_train.jsonl: the listener's exact prompt, sessions 1-4,
          training people only). The untuned 7B reads each one twice, once told to reply gently and once told to
          reply intrusively (the IP scale items as instructions). The residual stream at the last prompt token,
          at every layer, gives v[L] = mean(gentle) - mean(intrusive). Also stored: the mean residual norm per
          layer (the steering strength is a fraction of it) and d', how well v[L] separates held-out pairs
          (a hint about layers, not the choice).
  sweep   scripts/steering.sh: each LAYER:COEF on 20 calibration people, multi-session (2-4 sessions), scored for
          judge IP and Success. Coef 0 is the unsteered control on the same people and session seeds.
  select  rule fixed before the sweep: among settings with a fallback share <= 5 % and Success >= control - 0.03,
          the lowest mean IP. Writes LAYER:COEF to <run>/selected.txt.
  final   the chosen setting and the coef-0 control on the 100 test people, multi-session, all scores.
  report  steering vs its control vs cells A-D and step 12, by-person bootstrap CIs.

    python src/steering.py build  --out runs/steering/steering_vectors.pt
    python src/steering.py select --run runs/steering --arm steer_calib
    python src/steering.py report --run runs/steering --arm steer_mono --out temp_results/15_activation_steering
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from build_sft import SUPPORTER_SYSTEM
from common import DATA, ROOT, read_jsonl

STYLES = {
    "gentle": ("In this reply, stay with exactly what they have said: reflect it back simply and leave them room. "
               "Do not interpret, do not name feelings, needs or causes they have not expressed, do not give "
               "advice, and do not press for more."),
    "intrusive": ("In this reply, go beyond what they have said: tell them the deeper feeling or motive you think "
                  "is behind it, name the need or cause they have not admitted, give advice, and press them for "
                  "the details they seem reluctant to share."),
}


def build(n: int, out: Path, seed: int) -> None:
    import torch

    from llm import LLM

    if out.exists():   # the path is part of the reply cache key: never rebuild in place
        print(f"{out} exists, kept")
        return
    rows = random.Random(seed).sample(read_jsonl(DATA / "sft" / "wo_thoughts_train.jsonl"), n)
    llm = LLM("supporter", backend="transformers")
    model, tok = llm.backend.model, llm.backend.tokenizer
    acts = {k: [] for k in STYLES}
    try:
        for i, r in enumerate(rows):
            for k, extra in STYLES.items():
                msgs = [{"role": "system", "content": f"{SUPPORTER_SYSTEM} {extra}"},
                        {"role": "user", "content": r["input"]}]
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                enc = tok(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states
                acts[k].append(torch.stack([h[0, -1].float().cpu() for h in hs]))
            if i % 50 == 0:
                print(f"{i}/{n} contexts")
    finally:
        model = None   # drop the last reference, or release() cannot free the GPU
        llm.release()
    g, b = torch.stack(acts["gentle"]), torch.stack(acts["intrusive"])     # [n, layers + 1, hidden]
    vectors = g.mean(0) - b.mean(0)
    resid_norm = torch.cat([g, b]).norm(dim=-1).mean(0)
    # d': direction from the first half, separation measured on the second half
    h = n // 2
    u = g[:h].mean(0) - b[:h].mean(0)
    u = u / u.norm(dim=-1, keepdim=True)
    pg, pb = (g[h:] * u).sum(-1), (b[h:] * u).sum(-1)
    dprime = (pg.mean(0) - pb.mean(0)) / ((pg.var(0) + pb.var(0)) / 2).sqrt()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, "resid_norm": resid_norm, "dprime_heldout": dprime, "n": n, "seed": seed,
                "styles": STYLES, "source": "data/sft/wo_thoughts_train.jsonl"}, out)
    lines = ["layer  |v|      |h|      |v|/|h|  heldout d'"]
    lines += [f"{L:5d}  {vectors[L].norm():7.2f}  {resid_norm[L]:7.2f}  {vectors[L].norm() / resid_norm[L]:.3f}    "
              f"{dprime[L]:.2f}" for L in range(1, len(vectors))]
    (out.parent / "vector_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _mean(path: Path) -> tuple[float | None, int]:
    vals = [x["normalized"] for x in read_jsonl(path) if x.get("normalized") is not None] if path.exists() else []
    return (st.mean(vals) if vals else None), len(vals)


def arm_stats(run: Path, tag: str) -> dict:
    sessions = read_jsonl(run / "dialogues" / f"{tag}.jsonl")
    sup = [t for s in sessions for t in s["turns"]
           if t["role"] == "supporter" and t.get("meta", {}).get("source") != "opener_pool"]
    paths = Counter(t.get("meta", {}).get("path") for t in sup)
    out = {"tag": tag, "sessions": len(sessions), "turns": len(sup),
           "fallback": paths["fallback"] / max(1, len(sup)),
           "words": st.mean(len(t["text"].split()) for t in sup) if sup else None}
    for scale in ("ip", "success", "pri"):
        out[scale], out[f"n_{scale}"] = _mean(run / "scores" / tag / f"{scale}.jsonl")
    cf = run / "scores" / tag / "pri_summary.json"
    if cf.exists():
        s = json.loads(cf.read_text())
        out["cf_pri"], out["cf_ci"] = s.get("pri"), (s.get("ci_low"), s.get("ci_high"))
    return out


def select(run: Path, arm: str) -> str:
    tags = sorted(p.stem for p in (run / "dialogues").glob(f"{arm}_L*_c*.jsonl"))
    stats = [arm_stats(run, t) for t in tags]
    control = next((s for s in stats if s["tag"].endswith("_c0")), None)
    if control is None or control["success"] is None:
        raise SystemExit(f"no scored coef-0 control among {tags}")
    ok = [s for s in stats if not s["tag"].endswith("_c0") and s["ip"] is not None
          and s["fallback"] <= 0.05 and s["success"] >= control["success"] - 0.03]
    lines = ["tag  sessions  IP  Success  fallback  words"]
    lines += [f"{s['tag']}  {s['sessions']}  {s['ip']}  {s['success']}  {s['fallback']:.3f}  {s['words']}"
              for s in stats]
    if not ok:
        lines.append("no steered setting kept Success within 0.03 of the control with <= 5 % fallback")
        (run / "selection.txt").write_text("\n".join(lines) + "\n")
        raise SystemExit("\n".join(lines))
    best = min(ok, key=lambda s: s["ip"])
    _, layer, coef = best["tag"].rsplit("_", 2)
    choice = f"{layer[1:]}:{coef[1:]}"
    lines.append(f"selected {best['tag']} -> --steer {choice}")
    (run / "selection.txt").write_text("\n".join(lines) + "\n")
    (run / "selected.txt").write_text(choice + "\n")
    print("\n".join(lines))
    return choice


def report(run: Path, arm: str, out: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from recovery_cost import diff_ci, mean_by_person

    tags = sorted(p.stem for p in (run / "dialogues").glob(f"{arm}_L*_c*.jsonl"))
    control = next(t for t in tags if t.endswith("_c0"))
    steered = [t for t in tags if t != control]
    others = [(ROOT / "runs/baselines_1000", c) for c in
              ("cellA_mono_ungated", "cellB_mono_gated", "cellC_dec_ungated", "cellD_dec_gated")]
    others.append((ROOT / "runs/sft_agents", "sft_dec_gated_mem"))
    rows = [(run, t) for t in [control] + steered] + [(r, c) for r, c in others if (r / "dialogues" / f"{c}.jsonl").exists()]
    lines = [__doc__.split("\n\n")[0], "",
             "arm  sessions  IP  PRI  Success  counterfactual PRI [95% CI]  fallback  words"]
    for r, t in rows:
        s = arm_stats(r, t)
        f = lambda x: "n/a" if x is None else f"{x:.3f}"
        cf = (f"{s['cf_pri']:+.4f} [{s['cf_ci'][0]:+.4f}, {s['cf_ci'][1]:+.4f}]" if s.get("cf_pri") is not None
              else "n/a")
        lines.append(f"{t}  {s['sessions']}  {f(s['ip'])}  {f(s['pri'])}  {f(s['success'])}  {cf}  "
                     f"{s['fallback']:.3f}  {f(s['words'])}")
    lines += ["", "differences, by-person bootstrap 95% CI (b - a)"]
    pairs = [((run, control), (run, t)) for t in steered]
    pairs += [((run, t), (r, c)) for t in steered for r, c in others if (r / "dialogues" / f"{c}.jsonl").exists()]
    for (ra, a), (rb, b) in pairs:
        for scale in ("ip", "pri", "success"):
            if not ((ra / "scores" / a / f"{scale}.jsonl").exists() and (rb / "scores" / b / f"{scale}.jsonl").exists()):
                continue
            d, lo, hi = diff_ci(mean_by_person(ra, a, scale), mean_by_person(rb, b, scale))
            lines.append(f"   {b} - {a}  {scale:8s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines += ["", "Steering and its control run on the transformers backend (nf4); cells A-D ran on Ollama, so "
              "compare steering with its own control first. Cells and step 12 use different people seeds, so "
              "those comparisons are unpaired."]
    out.mkdir(parents=True, exist_ok=True)
    for f in ("vector_stats.txt", "selection.txt"):
        if (run / f).exists():
            (out / f).write_text((run / f).read_text())
    (out / "steering_comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Activation steering: build the vectors, pick a setting, report.")
    ap.add_argument("cmd", choices=["build", "select", "report"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="runs/steering")
    ap.add_argument("--arm", default="steer_calib")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.n, Path(args.out or "runs/steering/steering_vectors.pt"), args.seed)
    elif args.cmd == "select":
        select(Path(args.run), args.arm)
    else:
        report(Path(args.run), args.arm, Path(args.out or ROOT / "temp_results/15_activation_steering"))


if __name__ == "__main__":
    main()
