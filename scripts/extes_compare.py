"""Training corpus x test people: ours vs ExTES (SOP Section IV, ExTES as second corpus and strongest reactive
baseline). CPU only.

Untuned Qwen2.5-7B fine-tuned (QLoRA, reply-only targets) on our corpus or on ExTES sampled to 15k examples; one
session per person, 100 people per cell, one-call listener, no gate, no memory. All four cells judged by Mistral-Nemo on
10.10.10.104: the ours-on-our-people cell was first judged on the lab and is re-judged there (runs/scaling_eval_104),
because the judge's serving machine shifts IP (summaries/25_judge_machine_check). By-person bootstrap 95% CIs;
cells use different seeds, so differences are unpaired.

    python scripts/extes_compare.py --out temp_results/16_corpus_comparison__ours_vs_ExTES
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from recovery_cost import diff_ci, mean_by_person  # noqa: E402
from steering import arm_stats  # noqa: E402

OURS_OURS = ROOT / "runs/scaling_eval_104/qwen2.5_7B"
CELLS = {("ours", "our people"): (OURS_OURS, "sft_wo_thoughts"),
         ("ExTES", "our people"): (ROOT / "runs/corpus_extes", "corpus_extes"),
         ("ours", "ExTES people"): (ROOT / "runs/corpus_extes", "corpus_ours_extes_profiles"),
         ("ExTES", "ExTES people"): (ROOT / "runs/corpus_extes", "corpus_extes_extes_profiles")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/16_corpus_comparison__ours_vs_ExTES"))
    out = Path(ap.parse_args().out)
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    lines = [__doc__.split("\n\n")[0], "", __doc__.split("\n\n")[1], "",
             "corpus  people        sessions  IP     PRI    Success  counterfactual PRI"]
    for (c, p), (run, arm) in CELLS.items():
        s = arm_stats(run, arm)
        cf = f"{s['cf_pri']:+.4f} [{s['cf_ci'][0]:+.4f}, {s['cf_ci'][1]:+.4f}]" if s.get("cf_pri") is not None else "n/a"
        lines.append(f"{c:6s}  {p:12s}  {s['sessions']:8d}  {f(s['ip'])}  {f(s['pri'])}  {f(s['success'])}    {cf}")
    lines += ["", "ExTES-trained minus ours-trained, same people (b - a)"]
    for people in ("our people", "ExTES people"):
        (ra, a), (rb, b) = CELLS[("ours", people)], CELLS[("ExTES", people)]
        for scale in ("ip", "pri", "success"):
            d, lo, hi = diff_ci(mean_by_person(ra, a, scale), mean_by_person(rb, b, scale))
            lines.append(f"   {people:12s} {scale:8s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines += ["", "The counterfactual PRI of the ours-on-our-people cell is the lab's (not re-run on 10.10.10.104)."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "extes_2x2.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
