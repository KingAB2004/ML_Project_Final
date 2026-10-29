"""Does the judge's serving machine change the scores? (CPU only)

The same Mistral-Nemo weights (Ollama digest daf673741712) judged on the lab (Ollama 0.35 at check time) and on
10.10.10.104 (Ollama 0.18), on identical items: cell A's turn-level IP (also against its original 7 Oct lab
scores), session-level dependence for cell A and the SFT one-call arm, and the sycophancy probes for the base model.
Files under runs/judgecheck/<machine>/ (scores_judgecheck = same layout as runs/psych and scores/).

    python scripts/judge_machine_check.py --out temp_results/25_judge_machine_check
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from judge_agreement import spearman  # noqa: E402

J = ROOT / "runs/judgecheck"


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def by_target(path: Path) -> dict[str, float]:
    return {r["target"]: r["normalized"] for r in jl(path) if r.get("normalized") is not None}


def subscales(rows: list[dict], key: str, groups: dict) -> dict[str, float]:
    rows = [r for r in rows if r.get(key)]
    return {name: st.mean(st.mean((r[key][str(i)] - 1) / 6 for i in items) for r in rows) for name, items in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp_results/25_judge_machine_check"))
    out = Path(ap.parse_args().out)
    lines = [__doc__.split("\n\n")[0], "", "== turn-level IP, cell A (same 1,240 sampled turns)",
             "judge instance  n  mean IP  Spearman rho vs original"]
    orig = by_target(ROOT / "runs/baselines_1000/scores/cellA_mono_ungated/ip.jsonl")
    for label, path in [("lab, 7 Oct (original)", None), ("lab, 23 Oct", J / "lab/scores/cellA_mono_ungated/ip.jsonl"),
                        ("10.10.10.104, 23 Oct", J / "104/scores/cellA_mono_ungated/ip.jsonl")]:
        x = orig if path is None else by_target(path)
        if not x:
            lines.append(f"{label}  not available")
            continue
        both = sorted(set(x) & set(orig))
        rho = spearman([x[k] for k in both], [orig[k] for k in both]) if path else 1.0
        lines.append(f"{label}  {len(x)}  {st.mean(x.values()):.3f}  {rho:.3f}")
    dep = {"reliance": (1, 2), "outward": (3, 4), "autonomy": (5,)}
    lines += ["", "== session-level dependence (same sessions: first 15 test people of each arm)",
              "arm  judge machine  reliance  outward  autonomy"]
    for arm, lab, other in [("cellA_mono_ungated", J / "lab/dependence/cellA_mono_ungated.jsonl",
                             ROOT / "runs/psych/dependence/cellA_mono_ungated.jsonl"),
                            ("pref_mono_sft", ROOT / "runs/psych_lab_judge/dependence/pref_mono_sft.jsonl",
                             J / "104/dependence/pref_mono_sft.jsonl")]:
        for machine, path in (("lab", lab), ("10.10.10.104", other)):
            rows = jl(path)
            if rows:
                s = subscales(rows, "items", dep)
                lines.append(f"{arm}  {machine}  " + "  ".join(f"{v:.3f}" for v in s.values()))
    syc = {"sycophancy": (1, 2, 3), "balance": (4, 5), "harshness": (6, 7)}
    lines += ["", "== sycophancy probes, base model (same 48 replies)", "judge machine  sycophancy  balance  harshness  IP"]
    for machine, path in (("lab", J / "lab/sycophancy/scores_base.jsonl"),
                          ("10.10.10.104", ROOT / "runs/psych/sycophancy/scores_base.jsonl")):
        rows = jl(path)
        if rows:
            s = subscales(rows, "syco", syc)
            ip = st.mean(r["ip_norm"] for r in rows if r.get("ip_norm") is not None)
            lines.append(f"{machine}  " + "  ".join(f"{v:.3f}" for v in s.values()) + f"  {ip:.3f}")
    lines += ["", "Reading: a scale is safe to compare across arms judged on different machines only if the same items "
              "score alike here. All psychology-probe numbers in summaries/19 are from 10.10.10.104 alone."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_machine_check.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
