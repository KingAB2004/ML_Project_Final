"""Which headline contrasts survive a correction for multiple comparisons? (CPU only)

Every main claim of the project is a difference between two arms on IP, PRI or Success. Each gets a bootstrap p-value
over people (two-sided; 10,000 resamples) and the whole family is Holm-corrected at 0.05. Arms that share people and
session seeds (steering vs its control, DPO vs SFT) are resampled paired; all others independently, as in the reports.

    python scripts/multiple_comparisons.py --out temp_results/23_multiple_comparisons
"""
import argparse
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recovery_cost import mean_by_person  # noqa: E402

B = "runs/baselines_1000"
# (SOP part, label, (run, arm b), (run, arm a), paired): the difference is b - a
CONTRASTS = [
    ("Enh. 2 / Outcome 2 (2 x 2)", "gate, one call: B - A", (B, "cellB_mono_gated"), (B, "cellA_mono_ungated"), False),
    ("Enh. 2 / Outcome 2 (2 x 2)", "gate, four agents: D - C", (B, "cellD_dec_gated"), (B, "cellC_dec_ungated"), False),
    ("Enh. 2 / Outcome 2 (2 x 2)", "decomposition, ungated: C - A", (B, "cellC_dec_ungated"), (B, "cellA_mono_ungated"), False),
    ("Enh. 2 / Outcome 2 (2 x 2)", "decomposition, gated: D - B", (B, "cellD_dec_gated"), (B, "cellB_mono_gated"), False),
    ("Outcome 1 (reactive baseline)", "proactive vs reactive seeker: base - reactive",
     (B, "base_instruct"), (B, "reactive_baseline"), False),
    ("Enh. 4 / Outcome 4 (memory)", "need-state - no memory", ("runs/memory_v2", "mem_needstate"), ("runs/memory_v2", "mem_none"), False),
    ("Enh. 4 / Outcome 4 (memory)", "summary - no memory", ("runs/memory_v2", "mem_summary"), ("runs/memory_v2", "mem_none"), False),
    ("Enh. 4 / Outcome 4 (memory)", "dense - no memory", ("runs/memory_v2", "mem_dense"), ("runs/memory_v2", "mem_none"), False),
    ("Enh. 2 (step 12)", "fine-tuned Generator: step 12 - D (backends differ, see 20_step12)",
     ("runs/sft_agents", "sft_dec_gated_mem"), (B, "cellD_dec_gated"), False),
    ("Backend check", "transformers vs Ollama, untuned one call: steering control - cell A",
     ("runs/steering", "steer_mono_L12_c0"), (B, "cellA_mono_ungated"), False),
    ("Section IV (ExTES)", "ExTES- vs ours-trained, our people",
     ("runs/corpus_extes", "corpus_extes"), ("runs/scaling_eval_104/qwen2.5_7B", "sft_wo_thoughts"), False),
    ("Section IV (ExTES)", "ExTES- vs ours-trained, ExTES people",
     ("runs/corpus_extes", "corpus_extes_extes_profiles"), ("runs/corpus_extes", "corpus_ours_extes_profiles"), False),
    ("Added: steering", "steering: c0.2 - c0", ("runs/steering", "steer_mono_L12_c0.2"), ("runs/steering", "steer_mono_L12_c0"), True),
    ("Added: DPO", "DPO - SFT", ("runs/preference", "pref_mono_dpo"), ("runs/preference", "pref_mono_sft"), True),
]
SCALES = ("ip", "pri", "success")


def flat_mean(by: dict, people: list) -> float:
    return st.mean(v for p in people for v in by[p])


def boot_p(a: dict, b: dict, paired: bool, reps: int = 10000) -> tuple[float, float, float, float]:
    """b - a: point, 95% CI, two-sided bootstrap p (share of resamples on the other side of 0, doubled)."""
    rng = random.Random(0)
    pa, pb = list(a), list(b)
    point = flat_mean(b, pb) - flat_mean(a, pa)
    ds = []
    for _ in range(reps):
        if paired:
            draw = [rng.choice(pa) for _ in pa]
            ds.append(flat_mean(b, draw) - flat_mean(a, draw))
        else:
            ds.append(flat_mean(b, [rng.choice(pb) for _ in pb]) - flat_mean(a, [rng.choice(pa) for _ in pa]))
    ds.sort()
    p = 2 * min(sum(d <= 0 for d in ds), sum(d >= 0 for d in ds)) / reps
    return point, ds[int(.025 * reps)], ds[int(.975 * reps)], min(1.0, max(p, 1 / reps))


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/23_multiple_comparisons"))
    out = Path(ap.parse_args().out)
    rows = []
    for part, label, (rb, b), (ra, a), paired in CONTRASTS:
        for scale in SCALES:
            fa, fb = ROOT / ra / "scores" / a / f"{scale}.jsonl", ROOT / rb / "scores" / b / f"{scale}.jsonl"
            if not (fa.exists() and fb.exists()):
                continue
            ma, mb = mean_by_person(ROOT / ra, a, scale), mean_by_person(ROOT / rb, b, scale)
            if paired:
                common = sorted(set(ma) & set(mb))
                ma, mb = {p: ma[p] for p in common}, {p: mb[p] for p in common}
            rows.append((part, label, scale, paired, *boot_p(ma, mb, paired)))
    adj = holm([r[7] for r in rows])
    lines = [__doc__.split("\n\n")[0], "",
             f"{len(rows)} tests, Holm-corrected at 0.05. 'yes' = the difference survives the correction.", "",
             "SOP part | contrast | scale | paired | difference [95% CI] | p | Holm p | survives"]
    for r, q in zip(rows, adj):
        part, label, scale, paired, d, lo, hi, p = r
        lines.append(f"{part} | {label} | {scale} | {'yes' if paired else 'no'} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{p:.4f} | {q:.4f} | {'yes' if q < 0.05 else 'no'}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "multiple_comparisons.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06], holm([0.01, 0.04, 0.03])
    main()
