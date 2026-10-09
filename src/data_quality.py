"""Data-quality evaluation of our corpus next to ESConv and ExTES, in the shape of the baseline paper's
Tables 2, 6 and 8.

  Table 8 (CPU)   user-description diversity: self-BLEU-2 / -4 (lower = more diverse), Distinct-2, Shannon
                  entropy of the word distribution (bits). The same number of descriptions per dataset (both
                  D-2 and entropy grow with corpus size). Ours: the natural-language parts of each profile
                  (emotion, feeling, memory events, need chain, terminal need); ExTES: `description`;
                  ESConv: `situation` (human-written).
  Table 2 (GPU)   dialogue quality on the judge scales (six basic metrics, CRS Aff / Neg, RAC Sup / Man) by our
                  Mistral-Nemo judge, on a sample of each dataset's dialogues (scripts/sop_queue2.sh step 0).
  Table 6 (GPU)   the same dialogues scored by ESC-RANK (Zhao et al. 2024; InternLM2-chat-7B + per-dimension
                  LoRA, 0-4 per dimension), the scorer the paper used, so these are the numbers most directly
                  comparable to the paper's (scripts/escrank_score.py, same queue step).

  python extra/data_quality.py samples      # dialogue samples for the GPU steps -> runs/data_quality/dialogues/
  python extra/data_quality.py table        # Table 8 now + Tables 2 / 6 from whatever GPU scores exist
Writes results/v3_full1000/dataset/data_quality.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from _shared import ROOT, read_jsonl, write_json, write_jsonl

sys.path.insert(0, str(ROOT / "src"))
from convert_corpus import turns_from_esconv, turns_from_extes  # noqa: E402

RUN = ROOT / "runs" / "data_quality"
OUT = ROOT / "results" / "v3_full1000" / "dataset"
SETS = ("ours", "extes", "esconv")
N_DESCRIPTIONS = 1000
N_DIALOGUES = {"ours": 200, "extes": 100, "esconv": 100}
BASIC = ("fluency", "diversity", "empathy", "information", "humanoid", "skillfulness")

# The paper's values (GPT-4o rows of Table 2, Table 6, Table 8), for reference only: other judge, other language
PAPER = {
    "table2": {"esconv": [76.0, 65.9, 75.9, 62.1, 71.1, 69.6, 5.12, 1.74, 6.03, 5.22],
               "extes": [91.8, 81.3, 92.5, 81.1, 89.2, 90.3, 5.56, 1.48, 6.79, 6.03],
               "cocoon": [93.1, 83.1, 95.7, 86.1, 89.5, 93.3, 5.84, 1.21, 6.90, 6.20]},
    "table6": {"esconv": [72.3, 55.8, 74.0, 56.5, 50.8, 69.5], "extes": [75.0, 75.0, 75.0, 75.0, 70.3, 74.8],
               "cocoon": [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]},
    "table8": {"cocoon": [0.760, 0.287, 0.476, 10.17], "extes": [0.875, 0.678, 0.323, 7.89],
               "esconv": [0.747, 0.540, 0.508, 8.24]},
}


# --------------------------------------------------------------------------- Table 8
def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def descriptions(which: str) -> list[str]:
    if which == "ours":
        out = []
        for p in read_jsonl(ROOT / "data" / "profiles" / "profiles.jsonl"):
            chain = " ".join(n.get("text", "") for n in p.get("need_chain") or [] if isinstance(n, dict))
            memory = " ".join(m.get("text", "") for m in p.get("memory") or [] if isinstance(m, dict))
            out.append(" ".join(x for x in (p.get("emotion", ""), p.get("feeling", ""), memory, chain,
                                            p.get("terminal_need", "")) if x))
        return out
    raw = json.loads((ROOT / "data" / "raw" / f"{which}.json").read_text())
    key = "description" if which == "extes" else "situation"
    return [r[key].strip() for r in raw if isinstance(r.get(key), str) and r[key].strip()]


def sentence_bleu(hyp: list[str], refs: list[list[str]], n: int) -> float:
    """BLEU-n of one hypothesis against many references: clipped n-gram precisions, uniform weights,
    epsilon smoothing for empty orders, brevity penalty against the closest reference length."""
    if not hyp:
        return 0.0
    logs = []
    for k in range(1, n + 1):
        grams = Counter(tuple(hyp[i:i + k]) for i in range(len(hyp) - k + 1))
        if not grams:
            logs.append(math.log(1e-9))
            continue
        best: Counter = Counter()
        for r in refs:
            for g, c in Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1)).items():
                if c > best[g]:
                    best[g] = c
        clipped = sum(min(c, best[g]) for g, c in grams.items())
        logs.append(math.log(max(clipped, 0.1) / sum(grams.values())))
    closest = min((len(r) for r in refs), key=lambda L: (abs(L - len(hyp)), L))
    bp = 1.0 if len(hyp) > closest else math.exp(1 - closest / len(hyp))
    return bp * math.exp(sum(logs) / n)


def diversity(texts: list[str], seed: int = 0, n_hyp: int = 300, n_ref: int = 100) -> dict:
    rng = random.Random(seed)
    texts = rng.sample(texts, min(N_DESCRIPTIONS, len(texts)))
    toks = [words(t) for t in texts]
    hyps = rng.sample(range(len(toks)), min(n_hyp, len(toks)))
    b2, b4 = [], []
    for i in hyps:          # self-BLEU: each description against a sample of the others
        refs = [toks[j] for j in rng.sample([j for j in range(len(toks)) if j != i], min(n_ref, len(toks) - 1))]
        b2.append(sentence_bleu(toks[i], refs, 2))
        b4.append(sentence_bleu(toks[i], refs, 4))
    bigrams = [tuple(t[i:i + 2]) for t in toks for i in range(len(t) - 1)]
    unigrams = Counter(w for t in toks for w in t)
    total = sum(unigrams.values())
    return {"n_descriptions": len(texts), "mean_words": statistics.fmean(len(t) for t in toks),
            "self_bleu_2": statistics.fmean(b2), "self_bleu_4": statistics.fmean(b4),
            "distinct_2": len(set(bigrams)) / max(1, len(bigrams)),
            "shannon_entropy_bits": -sum(c / total * math.log2(c / total) for c in unigrams.values())}


# --------------------------------------------------------------------------- dialogue samples
def sample_dialogues() -> None:
    rng = random.Random(0)
    ours = [json.loads(l) for l in open(ROOT / "data" / "corpus" / "sessions_annotated.jsonl")]
    rng.shuffle(ours)
    out = {"ours": ours[:N_DIALOGUES["ours"]]}
    for which, parse, key in (("extes", turns_from_extes, "description"), ("esconv", turns_from_esconv, "situation")):
        raw = json.loads((ROOT / "data" / "raw" / f"{which}.json").read_text())
        rng.shuffle(raw)
        rows = []
        for i, r in enumerate(raw):
            turns = parse(r)
            if len(turns) >= 6:
                rows.append({"session_id": f"{which}{i:05d}", "profile_id": f"{which}{i:05d}", "session_index": 1,
                             "turns": turns, "profile_snapshot": {"feeling": r.get(key, ""), "terminal_need": ""}})
            if len(rows) == N_DIALOGUES[which]:
                break
        out[which] = rows
    for which, rows in out.items():
        write_jsonl(RUN / "dialogues" / f"{which}.jsonl", rows)
        print(f"{which}: {len(rows)} dialogues -> {RUN / 'dialogues' / f'{which}.jsonl'}")


# --------------------------------------------------------------------------- tables
def judge_row(which: str) -> dict | None:
    from metrics import aggregate
    d = RUN / "scores" / which
    if not all((d / f"{s}.jsonl").exists() for s in ("basic", "crs", "rac")):
        return None
    agg = {s: aggregate(read_jsonl(d / f"{s}.jsonl"), s) for s in ("basic", "crs", "rac")}
    return {**{m: agg["basic"].get(f"{m}_100") for m in BASIC}, "basic_avg": agg["basic"].get("basic_avg_100"),
            "aff": agg["crs"].get("affective_improvement"), "neg": agg["crs"].get("negative_helper"),
            "sup": agg["rac"].get("supportiveness"), "man": agg["rac"].get("management"),
            "n": agg["basic"]["n"]}


def escrank_row(which: str) -> dict | None:
    f = RUN / "escrank" / f"{which}.jsonl"
    if not f.exists():
        return None
    rows = read_jsonl(f)
    out = {}
    for m in BASIC:
        vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
        out[m] = 25 * statistics.fmean(vals) if vals else None        # 0-4 per dialogue; x 25 as in Table 6
    return {**out, "n": len(rows)}


def fmt(x, nd=1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def table() -> None:
    div = {w: diversity(descriptions(w)) for w in SETS}
    judge = {w: judge_row(w) for w in SETS}
    esc = {w: escrank_row(w) for w in SETS}
    name = {"ours": "**ours (this corpus)**", "extes": "ExTES", "esconv": "ESConv"}
    lines = ["# Data-quality evaluation: our corpus next to ExTES and ESConv", "",
             "Shape of the baseline paper's Tables 2, 6 and 8 (COCOON, EMNLP 2025). Our rows and the ExTES / "
             "ESConv rows marked *ours* are measured here with the same code and scorer; *paper* rows are copied "
             "from the paper for reference (their judge is GPT-4o and some of their data is Chinese). Script: "
             "`extra/data_quality.py`.", ""]

    lines += ["## User-description diversity (paper Table 8)", "",
              f"{N_DESCRIPTIONS} descriptions per dataset (sampled, seed 0); self-BLEU of 300 descriptions against "
              "100 others each. BLEU lower = more diverse; D-2 and entropy higher = more diverse.", "",
              "| Dataset | source | BLEU-2 ↓ | BLEU-4 ↓ | D-2 ↑ | Shannon entropy ↑ | mean words |", "|---|---|---|---|---|---|---|"]
    for w in SETS:
        d = div[w]
        lines.append(f"| {name[w]} | ours | {d['self_bleu_2']:.3f} | {d['self_bleu_4']:.3f} | {d['distinct_2']:.3f} | "
                     f"{d['shannon_entropy_bits']:.2f} | {d['mean_words']:.0f} |")
    for w, v in PAPER["table8"].items():
        lines.append(f"| {w if w != 'cocoon' else 'COCOON'} | paper | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {v[3]:.2f} | - |")

    lines += ["", "## Dialogue quality, judge scales (paper Table 2)", "",
              "Six basic metrics on 0-100, CRS Aff / Neg and RAC Sup / Man on 1-7 (Neg lower is better). Our judge: "
              "Mistral-Nemo-12B (paper: GPT-4o).", "",
              "| Dataset | source | n | Flu | Div | Emp | Inf | Hum | Skil | Basic Avg | Aff | Neg ↓ | Sup | Man |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in SETS:
        r = judge[w]
        lines.append(f"| {name[w]} | ours | " + ("pending (lab, sop_queue2 step 0) |" + " - |" * 11 if r is None else
                     f"{r['n']} | " + " | ".join(fmt(r[m]) for m in BASIC) + f" | {fmt(r['basic_avg'])} | "
                     f"{fmt(r['aff'], 2)} | {fmt(r['neg'], 2)} | {fmt(r['sup'], 2)} | {fmt(r['man'], 2)} |"))
    for w, v in PAPER["table2"].items():
        lines.append(f"| {w if w != 'cocoon' else 'COCOON'} | paper | - | " + " | ".join(f"{x:.1f}" for x in v[:6])
                     + f" | {statistics.fmean(v[:6]):.1f} | " + " | ".join(f"{x:.2f}" for x in v[6:]) + " |")

    lines += ["", "## Dialogue quality, ESC-RANK (paper Table 6)", "",
              "ESC-RANK scores each dialogue 0-4 per dimension; shown x 25 as in the paper (its ceiling 75 = every "
              "dialogue at 3). Same scorer as the paper, so these rows would be the most directly comparable. Not run on 7 Oct: "
              "ESC-RANK's InternLM2 code needs an older transformers than the lab's Python 3.13 installs from wheels, "
              "and the 15 GB RAM server could not build it alongside the running jobs (`scripts/escrank_score.py`, "
              "ESCRANK_ENABLE=1).", "",
              "| Dataset | source | n | Flu | Div | Emp | Inf | Hum | Skil |", "|---|---|---|---|---|---|---|---|---|"]
    for w in SETS:
        r = esc[w]
        lines.append(f"| {name[w]} | ours | " + ("not run (see note) |" + " - |" * 6 if r is None else
                     f"{r['n']} | " + " | ".join(fmt(r[m]) for m in BASIC) + " |"))
    for w, v in PAPER["table6"].items():
        lines.append(f"| {w if w != 'cocoon' else 'COCOON'} | paper | - | " + " | ".join(f"{x:.1f}" for x in v) + " |")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "data_quality.json", {"table8": div, "table2": judge, "table6": esc, "paper": PAPER})
    print(f"wrote {OUT / 'data_quality.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Data-quality evaluation (paper Tables 2, 6, 8).")
    ap.add_argument("step", choices=("samples", "table"))
    args = ap.parse_args()
    sample_dialogues() if args.step == "samples" else table()


if __name__ == "__main__":
    main()
