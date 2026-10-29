"""Judge-judge agreement: do the headline numbers depend on which open model judges them? (CPU only)

The same sampled turns and sessions of the key arms are scored by the main judge (Mistral-Nemo-12B, scores/) and by a
second judge from another family (Llama-3.1-8B, role judge_fallback, scores_judge2/; scripts/second_judge_104.sh).
Reported: per-item agreement (Spearman rho over the turns both scored), each arm's mean under both judges, the rank
agreement of the arms (Kendall tau), and whether the key contrasts keep their sign. This checks the judge choice, not
the judge against people - judge-human agreement still needs human raters (human_eval/).

    python scripts/judge_agreement.py --out temp_results/21_judge_judge_agreement
"""
import argparse
import json
import statistics as st
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = [("runs/baselines_1000", "cellA_mono_ungated"), ("runs/baselines_1000", "cellB_mono_gated"),
        ("runs/baselines_1000", "cellC_dec_ungated"), ("runs/baselines_1000", "cellD_dec_gated"),
        ("runs/memory_v2", "mem_none"), ("runs/memory_v2", "mem_needstate"),
        ("runs/steering", "steer_mono_L12_c0"), ("runs/steering", "steer_mono_L12_c0.2"),
        ("runs/sft_agents", "sft_dec_gated_mem")]
CONTRASTS = [("gate, one call (B - A)", "cellB_mono_gated", "cellA_mono_ungated"),
             ("gate, four agents (D - C)", "cellD_dec_gated", "cellC_dec_ungated"),
             ("decomposition, ungated (C - A)", "cellC_dec_ungated", "cellA_mono_ungated"),
             ("need-state memory (needstate - none)", "mem_needstate", "mem_none"),
             ("steering (c0.2 - c0)", "steer_mono_L12_c0.2", "steer_mono_L12_c0"),
             ("fine-tuned Generator (step 12 - D)", "sft_dec_gated_mem", "cellD_dec_gated")]


def load(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = (json.loads(l) for l in open(path))
    return {r["target"]: r["normalized"] for r in rows if r.get("normalized") is not None}


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):            # average ranks over ties (scores are coarse, ties are common)
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def kendall(a: list[float], b: list[float]) -> float:
    pairs = list(combinations(range(len(a)), 2))
    s = sum((a[i] - a[j]) * (b[i] - b[j]) > 0 for i, j in pairs) - sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i, j in pairs)
    return s / len(pairs) if pairs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/21_judge_judge_agreement"))
    out = Path(ap.parse_args().out)
    lines = [__doc__.split("\n\n")[0], ""]
    means = {}
    for scale in ("ip", "pri", "success"):
        lines += [f"== {scale}", "arm  n both  Mistral mean  Llama mean  Spearman rho (per item)"]
        all_a, all_b = [], []
        for run, arm in ARMS:
            a = load(ROOT / run / "scores" / arm / f"{scale}.jsonl")
            b = load(ROOT / run / "scores_judge2" / arm / f"{scale}.jsonl")
            both = sorted(set(a) & set(b))
            if not both:
                continue
            xa, xb = [a[k] for k in both], [b[k] for k in both]
            all_a += xa
            all_b += xb
            means[(scale, arm)] = (st.mean(xa), st.mean(xb))
            lines.append(f"{arm}  {len(both)}  {st.mean(xa):.3f}  {st.mean(xb):.3f}  {spearman(xa, xb):.3f}")
        if all_a:
            arms = [arm for _, arm in ARMS if (scale, arm) in means]
            tau = kendall([means[(scale, x)][0] for x in arms], [means[(scale, x)][1] for x in arms])
            lines += [f"pooled: {len(all_a)} items, Spearman rho {spearman(all_a, all_b):.3f}; "
                      f"arm ranking Kendall tau {tau:.3f} over {len(arms)} arms", ""]
    lines += ["== key contrasts: difference under each judge (same sign = the conclusion does not hang on the judge;",
              "   'near 0' = one of the two differences is under 0.005, too small to have a sign)",
              "contrast  scale  Mistral  Llama  agree"]
    for label, b, a in CONTRASTS:
        for scale in ("ip", "pri", "success"):
            if (scale, a) in means and (scale, b) in means:
                dm = means[(scale, b)][0] - means[(scale, a)][0]
                dl = means[(scale, b)][1] - means[(scale, a)][1]
                agree = "near 0" if min(abs(dm), abs(dl)) < 0.005 else ("yes" if dm * dl > 0 else "NO")
                lines.append(f"{label}  {scale}  {dm:+.3f}  {dl:+.3f}  {agree}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_agreement.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
