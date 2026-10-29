"""What each way of lowering intrusiveness costs per supporter turn. (CPU only)

Cost is counted from the pipeline's structure and each turn's logged revisions, so it does not depend on which machine
ran the arm: the one-call listener makes 1 model call per turn (+1 critic call when gated, +2 per revision); the
four-agent pipeline makes 4 (Analyzer, Strategist, Generator, Critic - the critic scores even when ungated) +2 per
revision. Steering adds no call (a vector added inside the forward pass) and DPO none (it changes the weights).
Simulator, judge and memory-upkeep calls are not counted; nor are re-asks after unparseable output (rare).
Benefit is the drop in judge IP from the baseline named in each row.

    python scripts/cost_per_turn.py --out temp_results/24_cost_per_turn
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steering import arm_stats  # noqa: E402

B = "runs/baselines_1000"
# (label, run, arm, architecture, baseline label, baseline run, baseline arm)
SYSTEMS = [
    ("cell A: one call, ungated", B, "cellA_mono_ungated", "mono", None, None, None),
    ("cell B: one call + gate", B, "cellB_mono_gated", "mono", "cell A", B, "cellA_mono_ungated"),
    ("cell C: four agents, ungated", B, "cellC_dec_ungated", "dec", "cell A", B, "cellA_mono_ungated"),
    ("cell D: four agents + gate", B, "cellD_dec_gated", "dec", "cell A", B, "cellA_mono_ungated"),
    ("step 12: four agents + gate, fine-tuned Generator", "runs/sft_agents", "sft_dec_gated_mem", "dec",
     "cell A", B, "cellA_mono_ungated"),
    ("steering control: one call (transformers)", "runs/steering", "steer_mono_L12_c0", "mono", None, None, None),
    ("steering: one call + vector", "runs/steering", "steer_mono_L12_c0.2", "mono",
     "steering control", "runs/steering", "steer_mono_L12_c0"),
    ("SFT control: fine-tuned one call", "runs/preference", "pref_mono_sft", "mono", None, None, None),
    ("DPO: fine-tuned one call after DPO", "runs/preference", "pref_mono_dpo", "mono",
     "SFT control", "runs/preference", "pref_mono_sft"),
]


def calls(run: str, arm: str, arch: str) -> tuple[float, float]:
    per_turn, gated = [], False
    for line in open(ROOT / run / "dialogues" / f"{arm}.jsonl"):
        s = json.loads(line)
        gated = s.get("arm_spec", {}).get("gate") == "conformal"
        for t in s["turns"]:
            if t["role"] != "supporter" or t.get("meta", {}).get("source") == "opener_pool":
                continue
            rev = int(t.get("meta", {}).get("revisions") or 0)
            per_turn.append((4 if arch == "dec" else 1 + gated) + 2 * rev)
    return st.mean(per_turn), max(per_turn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/24_cost_per_turn"))
    out = Path(ap.parse_args().out)
    lines = [__doc__.split("\n\n")[0], "",
             "system | model calls per turn (mean, max) | words per reply | IP | Success | vs | IP drop | extra calls | "
             "IP drop per extra call"]
    for label, run, arm, arch, base_label, brun, barm in SYSTEMS:
        if not (ROOT / run / "dialogues" / f"{arm}.jsonl").exists():
            lines.append(f"{label} | not run yet")
            continue
        mean_c, max_c = calls(run, arm, arch)
        s = arm_stats(ROOT / run, arm)
        row = f"{label} | {mean_c:.2f} ({max_c}) | {s['words']:.1f} | {s['ip']:.3f} | {s['success']:.3f}"
        if base_label and (ROOT / brun / "dialogues" / f"{barm}.jsonl").exists():
            b = arm_stats(ROOT / brun, barm)
            extra = mean_c - calls(brun, barm, "mono")[0]
            drop = b["ip"] - s["ip"]
            per = f"{drop / extra:+.4f}" if extra > 0.01 else "no extra call"
            row += f" | {base_label} | {drop:+.3f} | {extra:+.2f} | {per}"
        lines.append(row)
    lines += ["", "IP drop > 0 means less intrusive than the baseline. Cells A-D ran on Ollama, with cell A as their "
              "baseline. Step 12, steering and DPO ran on transformers: steering and DPO are compared with their own "
              "controls; step 12 with cell A, so its row mixes the backend change with the fine-tuned Generator."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost_per_turn.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
