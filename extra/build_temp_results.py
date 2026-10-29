"""Collect every comparison so far into temp_results/, one folder per question, tables as .txt and the
matching figures as .png. Reads only what the pipeline and the analyses already wrote; no model calls.
Rerun after new results land (missing runs are listed as pending, never invented):

    python extra/build_temp_results.py
"""
from __future__ import annotations

import json
import shutil
import statistics as st
from collections import defaultdict
from pathlib import Path

from _shared import cluster_bootstrap, pyplot

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "v3_full1000"
RUNS = ROOT / "runs"
OUT = ROOT / "temp_results"
SESSION_SCALES = ("success", "aels", "basic", "crs", "rac")
TURN_SCALES = ("ip", "pri")
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")          # dataviz reference slots 1-4

# arm label -> (run dir, arm name)
ARMS = {
    "untuned 7B, one call (base_instruct)": (RUNS / "baselines_1000", "base_instruct"),
    "untuned 7B, reactive seeker (reactive_baseline)": (RUNS / "baselines_1000", "reactive_baseline"),
    "fine-tuned 0.5B, with thoughts": (RUNS / "scaling_eval/qwen2.5_0.5B", "sft_with_thoughts"),
    "fine-tuned 0.5B, without thoughts": (RUNS / "scaling_eval/qwen2.5_0.5B", "sft_wo_thoughts"),
    "fine-tuned 3B, with thoughts": (RUNS / "scaling_eval/qwen2.5_3B", "sft_with_thoughts"),
    "fine-tuned 3B, without thoughts": (RUNS / "scaling_eval/qwen2.5_3B", "sft_wo_thoughts"),
    "fine-tuned 7B, with thoughts": (RUNS / "scaling_eval/qwen2.5_7B", "sft_with_thoughts"),
    "fine-tuned 7B, without thoughts": (RUNS / "scaling_eval/qwen2.5_7B", "sft_wo_thoughts"),
    "fine-tuned Qwen3-4B, with thoughts": (RUNS / "scaling_eval/qwen3_4B", "sft_with_thoughts"),
    "fine-tuned Qwen3-4B, without thoughts": (RUNS / "scaling_eval/qwen3_4B", "sft_wo_thoughts"),
    "cell A: untuned, one call, gate off": (RUNS / "baselines_1000", "cellA_mono_ungated"),
    "cell B: untuned, one call, gate on": (RUNS / "baselines_1000", "cellB_mono_gated"),
    "cell C: untuned, four agents, gate off": (RUNS / "baselines_1000", "cellC_dec_ungated"),
    "cell D: untuned, four agents, gate on": (RUNS / "baselines_1000", "cellD_dec_gated"),
    "untuned 7B, four agents + need-state memory (flat)": (RUNS / "memory_1000", "mem_needstate"),
    "fine-tuned 7B (with thoughts), no memory": (RUNS / "sft7b_multi", "sft7b_multi_nomem"),
    "fine-tuned 7B (with thoughts) + need-state memory (tree)": (RUNS / "sft7b_multi", "sft7b_multi_mem"),
}


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def pid(target: str) -> str:
    return target.split("-")[0]


def sessions(label: str) -> dict[str, dict]:
    run, arm = ARMS[label]
    return {s["session_id"]: s for s in jl(run / "dialogues" / f"{arm}.jsonl")}


def score_rows(label: str, scale: str) -> list[dict]:
    run, arm = ARMS[label]
    return [r for r in jl(run / "scores" / arm / f"{scale}.jsonl") if not r.get("parse_failed")]


def mean_ci(rows: list[dict], key=lambda r: r["normalized"]) -> str:
    if not rows:
        return "pending"
    by = defaultdict(list)
    for r in rows:
        by[pid(r["target"])].append(key(r))
    lo, hi = cluster_bootstrap(list(by.values()), lambda cs: st.mean(v for c in cs for v in c), reps=1000)
    return f"{st.mean(key(r) for r in rows):.3f} [{lo:.3f}, {hi:.3f}]"


def table(header: list[str], rows: list[list], title: str = "", notes: list[str] = ()) -> str:
    cells = [header] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))
    out = ([title, "=" * len(title)] if title else []) + [line(header), "  ".join("-" * x for x in w)]
    out += [line(r) for r in cells[1:]]
    return "\n".join(out + [""] + [f"- {n}" for n in notes]) + "\n"


def write(folder: str, name: str, text: str) -> None:
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def copy(folder: str, *paths: Path) -> None:
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    for p in paths:
        if p.exists():      # reports become .txt so every table in the folder opens the same way
            shutil.copy2(p, d / (p.stem + ".txt" if p.suffix == ".md" else p.name))


def arm_row(label: str, scales=("success", "aels", "ip", "pri")) -> list:
    n = len(sessions(label))
    return [label, n or "pending"] + [mean_ci(score_rows(label, s)) for s in scales]


# --------------------------------------------------------------------------- folders


def f01_dataset() -> None:
    f = "01_dataset_quality__ours_vs_ExTES_vs_ESConv"
    dq = json.loads((RES / "dataset/data_quality.json").read_text())
    t8 = dq["table8"]
    rows = [[k, v["n_descriptions"], f"{v['self_bleu_2']:.3f}", f"{v['self_bleu_4']:.3f}", f"{v['distinct_2']:.3f}",
             f"{v['shannon_entropy_bits']:.2f}", f"{v['mean_words']:.0f}"] for k, v in t8.items()]
    write(f, "table8_profile_diversity.txt", table(
        ["dataset", "n", "self-BLEU-2 (lower=more diverse)", "self-BLEU-4", "Distinct-2", "entropy bits", "words"],
        rows, "Profile / user-description diversity (paper Table 8), same code for every dataset",
        ["Ours beats ExTES on all four measures; ESConv is more diverse than both on these measures."]))
    t2 = dq.get("table2") or {}
    write(f, "table2_dialogue_quality.txt", "Dialogue quality on the paper's judge scales (Table 2)\n"
          + ("\n".join(f"{k}: {v}" for k, v in t2.items() if v) or "pending: scores queued on the lab\n"))
    stats = json.loads((RES / "dataset/dataset_stats.json").read_text())
    write(f, "dataset_stats.txt", json.dumps({k: v for k, v in stats.items()}, indent=1))


def f02_training() -> None:
    f = "02_finetuning__validation_loss_by_model_size"
    rows = []
    for size in ("0.5B", "3B", "7B"):
        for arm in ("with_thoughts", "wo_thoughts"):
            stt = json.loads((RES / f"training/{size}/trainer_state_{arm}.json").read_text())
            ev = [round(x["eval_loss"], 3) for x in stt["log_history"] if "eval_loss" in x]
            rows.append([f"Qwen2.5-{size}", arm] + ev + [min(ev)])
    for arm in ("with_thoughts", "wo_thoughts"):
        cks = sorted((RUNS / f"scale_qwen3_4B/adapter_{arm}").glob("checkpoint-*/trainer_state.json"),
                     key=lambda p: int(p.parent.name.split("-")[1]))
        if cks:
            ev = [round(x["eval_loss"], 3) for x in json.loads(cks[-1].read_text())["log_history"] if "eval_loss" in x]
            rows.append(["Qwen3-4B", arm] + ev + ["-"] * (3 - len(ev)) + [min(ev)])
        else:
            rows.append(["Qwen3-4B", arm, "pending", "", "", ""])
    write(f, "validation_loss_by_epoch.txt", table(
        ["model", "arm", "epoch 1", "epoch 2", "epoch 3", "best"], rows,
        "Validation loss per epoch (QLoRA r32, 18,821 train / 2,608 val examples)",
        ["Best checkpoint was epoch 2 for every Qwen2.5 run; epoch 3 overfits.",
         "Qwen3-4B rows fill in as its training finishes ('-' = epoch not reached yet)."]))
    copy(f, *(RES / "analyses/scaling").glob("training_curves*.png"), RES / "analyses/scaling/scaling_loss.png")
    copy(f, RES / "analyses/scaling/report.md")


def f03_pvi() -> None:
    f = "03_information_in_thoughts__PVI"
    s = json.loads((RES / "analyses/pvi/summary.json").read_text())["sizes"]
    rows = [[k, v["n"], f"{v['v_information']:.2f} [{v['v_information_ci'][0]:.2f}, {v['v_information_ci'][1]:.2f}]",
             f"{v['share_negative']:.1%}", f"{v['reply_nll_with']:.3f}", f"{v['reply_nll_wo']:.3f}"] for k, v in s.items()]
    write(f, "pvi_by_size.txt", table(
        ["size", "turns", "bits per turn [95% CI]", "PVI < 0", "reply NLL with thoughts", "without"], rows,
        "Usable information the annotated thoughts carry about the reply (validation turns)"))
    bs = json.loads((RES / "analyses/pvi/pvi_by_session.json").read_text())
    rows = [[size, k, r["turns"], r["profiles"], f"{r['nll_with']:.3f}", f"{r['nll_wo']:.3f}", f"{r['pvi_bits']:.2f}"]
            for size, d in bs.items() for k, r in d.items()]
    write(f, "pvi_by_session.txt", table(
        ["size", "session", "turns", "people", "NLL with", "NLL without", "bits added"], rows,
        "The same, split by session number (teacher-forced on corpus sessions, not live)"))
    copy(f, *(RES / "analyses/pvi").glob("*.png"), RES / "analyses/pvi/report.md")


def f04_lora() -> None:
    f = "04_lora_geometry__how_the_update_uses_its_rank"
    s = json.loads((RES / "analyses/lora_geometry/summary.json").read_text())
    rows = []
    for size, d in s["sizes"].items():
        for arm in ("with_thoughts", "wo_thoughts"):
            a = d[arm]
            rows.append([size, arm] + [f"{a[k]:.2f}" if isinstance(a.get(k), float) else a.get(k, "")
                                       for k in ("fro", "stable_rank", "effective_rank", "energy_at_4")])
    write(f, "geometry.txt", table(["size", "arm", "norm", "stable rank", "effective rank (of 32)", "energy top-4"],
                                   rows, "Shape of the LoRA updates"))
    tr = s["truncation"]
    ks = sorted({int(k) for v in tr.values() for k in v})
    write(f, "rank_truncation.txt", table(["size|arm"] + [f"k={k}" for k in ks],
                                          [[n] + [f"{v.get(str(k), float('nan')):.3f}" for k in ks] for n, v in tr.items()],
                                          "Validation loss when every update is cut to rank k"))
    copy(f, *(RES / "analyses/lora_geometry").glob("*.png"), RES / "analyses/lora_geometry/report.md")


def f05_single() -> None:
    f = "05_single_session_test__finetuned_vs_untuned"
    labels = [l for l in ARMS if l.startswith(("untuned 7B, one call", "untuned 7B, reactive", "fine-tuned 0.5B",
                                               "fine-tuned 3B", "fine-tuned 7B, with", "fine-tuned 7B, without",
                                               "fine-tuned Qwen3"))]
    write(f, "scores.txt", table(["supporter", "sessions", "Success [95% CI]", "AELS", "IP (lower better)",
                                  "PRI observed (lower better)"], [arm_row(l) for l in labels],
                                 "100 test people, one session each, no memory, no gate",
                                 ["CIs: bootstrap over people.", "'pending' = not run or not scored yet."]))
    copy(f, RES / "analyses/scaling/scaling_test.png", RES / "analyses/scaling/thoughts_gain.png")


def f06_gate() -> None:
    f = "06_architecture_and_gate__2x2_cells"
    labels = [l for l in ARMS if l.startswith("cell ")]
    write(f, "cells.txt", table(["cell", "sessions", "Success [95% CI]", "AELS", "IP", "PRI observed"],
                                [arm_row(l) for l in labels],
                                "2 x 2: one call vs four agents, gate off vs on (untuned 7B, 2-4 sessions per person)",
                                ["Cells A/B never wrote memory (no Analyzer in the one-call supporter).",
                                 "Cells C/D: paused for the fine-tuned 7B runs; resume automatically."]))
    sweep = json.loads((RES / "baselines_1000/conformal/alpha_sweep.json").read_text())
    write(f, "gate_calibration.txt", table(["alpha", "lambda-hat", "release rate", "risk", "risk upper bound"],
                                           [[r["alpha"], r["lambda_hat"], f"{r['release_rate']:.0%}",
                                             f"{r['risk']:.3f}", f"{r['risk_ucb']:.3f}"] for r in sweep],
                                           "Conformal gate: threshold per risk budget alpha (delta 0.1, tau 0.20)"))
    copy(f, *(RES / "baselines_1000/extra/baseline_plots").glob("*.png"))


def f07_multi() -> None:
    f = "07_multi_session__finetuned_7B_with_and_without_memory"
    labels = ["fine-tuned 7B (with thoughts), no memory", "fine-tuned 7B (with thoughts) + need-state memory (tree)",
              "untuned 7B, four agents + need-state memory (flat)", "cell A: untuned, one call, gate off"]
    write(f, "overall.txt", table(["system", "sessions", "Success [95% CI]", "AELS", "IP", "PRI observed"],
                                  [arm_row(l) for l in labels],
                                  "100 test people, 2-4 linked sessions each",
                                  ["Fine-tuned runs and the untuned memory run: the person changes between sessions"
                                   " and the corrective seeker; cell A: same need, proactive seeker.",
                                   "Cell A's memory was never written, so it is effectively no memory."]))
    by = {}
    for l in labels:
        S, rows = sessions(l), score_rows(l, "success")
        groups = defaultdict(lambda: defaultdict(list))
        for r in rows:
            s = S.get(r["target"])
            if not s:
                continue
            i, g = s["session_index"], s.get("gap_days_from_prev") or 0
            groups["session"][i].append(r["normalized"])
            groups["gap"]["1st session" if i == 1 else "<=7 d" if g <= 7 else "7-21 d" if g <= 21 else ">21 d"].append(r["normalized"])
            if i > 1:
                groups["change"][(s["profile_snapshot"].get("advancement") or {}).get("applied") or "none"].append(r["normalized"])
        by[l] = groups
    for kind, keys in (("session", [1, 2, 3, 4]), ("gap", ["1st session", "<=7 d", "7-21 d", ">21 d"]),
                       ("change", ["intensified", "resolved", "displaced", "none"])):
        rows = [[l] + [f"{st.mean(by[l][kind][k]):.3f} (n={len(by[l][kind][k])})" if by[l][kind].get(k) else "-"
                       for k in keys] for l in labels]
        write(f, f"success_by_{kind}.txt", table(["system"] + [str(k) for k in keys], rows, f"Success by {kind}"))
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for color, l in zip(COLORS, labels):
        pts = [(k, st.mean(v)) for k, v in sorted(by[l]["session"].items())]
        if pts:
            ax.plot(*zip(*pts), color=color, lw=2, marker="o", ms=8, label=l)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("session with this person")
    ax.set_ylabel("Success (0-1)")
    ax.set_ylim(0.30, 0.45)          # a fixed, honest range: differences here are a few hundredths
    ax.set_title("Success by session, multi-session systems")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / f / "success_by_session.png", dpi=160, facecolor="#ffffff")
    plt.close(fig)


def f08_memory() -> None:
    f = "08_need_state_memory__untuned_flat_run"
    copy(f, RES / "memory_1000/extra/memory_eval/success_by_gap.png", RES / "memory_1000/extra/memory_eval/report.md")
    s = json.loads((RES / "memory_1000/extra/memory_eval/summary.json").read_text())["arms"]["mem_needstate"]
    write(f, "memory_eval_summary.txt", json.dumps(s, indent=1))


def f09_person() -> None:
    f = "09_one_person_ablation__p000394"
    rows = []
    for l in ARMS:
        succ = {r["target"].rsplit("-s", 1)[1]: round(r["normalized"] * 6 + 1) for r in score_rows(l, "success")
                if pid(r["target"]) == "p000394"}
        aels = [r["normalized"] for r in score_rows(l, "aels") if r["target"].startswith("p000394") and r["target"].endswith("-s1")]
        if succ:
            rows.append([l] + [succ.get(str(i), "") for i in (1, 2, 3, 4)] + [f"{aels[0]:.2f}" if aels else "-"])
    write(f, "ablation.txt", table(["system", "S1", "S2", "S3", "S4", "AELS (S1)"], rows,
                                   "Test person p000394: Success on the judge's 1-7 scale per session",
                                   ["Single-session systems have no S2-S4."]))


def f10_v2() -> None:
    f = "10_v2_reference__50_profile_run"
    text = (ROOT / "results/v2/README.md").read_text()
    keep = [b for b in text.split("\n## ") if b.startswith(("3.", "4."))]
    write(f, "v2_baselines_2x2_and_finetune.txt", "\n## ".join([""] + keep).strip() + "\n")


FOLDERS = (f01_dataset, f02_training, f03_pvi, f04_lora, f05_single, f06_gate, f07_multi, f08_memory, f09_person,
           f10_v2)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for fn in FOLDERS:
        fn()
    lines = ["temp_results: every comparison so far, one folder per question (rebuilt by extra/build_temp_results.py)",
             ""]
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        q = d.name.split("__", 1)
        lines.append(f"{d.name}/   {q[1].replace('_', ' ') if len(q) > 1 else ''}")
        lines += [f"    {p.name}" for p in sorted(d.iterdir())]
    (OUT / "00_README.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
