"""Step 12: the fine-tuned Generator inside the four-agent gated pipeline, against cell D (same pipeline, untuned
Generator) and the fine-tuned one-call model with memory (sft7b_multi_mem). 100 test people, 2-4 sessions each.
By-person bootstrap 95% CIs; the arms use different session seeds, so differences are unpaired.
BACKEND CAVEAT: step 12 ran on transformers (nf4), cells C and D on Ollama (Q4 GGUF). The backend alone moves IP: the
untuned one-call listener scores 0.085 on transformers (steer_mono_L12_c0) against 0.068 on Ollama (cell A), +0.017.
So most of step 12's IP gap to cell D may be the backend, not the fine-tuned Generator; no untuned four-agent gated arm
was run on transformers to separate the two.

    python scripts/step12_compare.py --out temp_results/20_step12_finetuned_generator
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from recovery_cost import diff_ci, mean_by_person  # noqa: E402
from steering import arm_stats  # noqa: E402

ARMS = [(ROOT / "runs/sft_agents", "sft_dec_gated_mem"), (ROOT / "runs/baselines_1000", "cellD_dec_gated"),
        (ROOT / "runs/baselines_1000", "cellC_dec_ungated"), (ROOT / "runs/sft7b_multi", "sft7b_multi_mem")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/20_step12_finetuned_generator"))
    out = Path(ap.parse_args().out)
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    lines = ["\n\n".join(__doc__.split("\n\n")[:2]), "", "arm  sessions  IP  PRI  Success  counterfactual PRI [95% CI]  fallback  words"]
    for run, arm in ARMS:
        s = arm_stats(run, arm)
        cf = f"{s['cf_pri']:+.4f} [{s['cf_ci'][0]:+.4f}, {s['cf_ci'][1]:+.4f}]" if s.get("cf_pri") is not None else "n/a"
        lines.append(f"{arm}  {s['sessions']}  {f(s['ip'])}  {f(s['pri'])}  {f(s['success'])}  {cf}  "
                     f"{s['fallback']:.3f}  {f(s['words'])}")
    lines += ["", "differences (step 12 - other), by-person bootstrap 95% CI"]
    (rb, b), others = ARMS[0], ARMS[1:]
    for ra, a in others:
        for scale in ("ip", "pri", "success"):
            if (ra / "scores" / a / f"{scale}.jsonl").exists():
                d, lo, hi = diff_ci(mean_by_person(ra, a, scale), mean_by_person(rb, b, scale))
                lines.append(f"   {b} - {a}  {scale:8s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out.mkdir(parents=True, exist_ok=True)
    (out / "step12_comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
