"""Phase 1.1 - seed situations (PLAN Sec. 8.1).

EmpatheticDialogues supplies human-written narrated situations; ESConv supplies the problem-type taxonomy
used for stratification. Each candidate passes a sustainability screen: can this situation plausibly support
a three-step need chain? The screen is per item, because the SOP's own worry - that many ED situations are
too short - has to be tested rather than assumed.

Outputs data/seeds/seeds.jsonl, plus reports/seed_stats.md.

  python src/seeds.py --limit 400
  python src/seeds.py --route synthetic --limit 400     # the declared fallback route
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from common import (
    DATA,
    REPORTS,
    cfg,
    read_json,
    read_jsonl,
    rng_for,
    seq_id,
    write_jsonl,
)
from llm import LLM, pmap

PROBLEM_TYPES_PATH = DATA / "raw" / "problem_types.json"
SEEDS_PATH = DATA / "seeds" / "seeds.jsonl"

# ESConv's problem-type taxonomy. Stored as a file so the classifier and the stratifier cannot disagree.
DEFAULT_PROBLEM_TYPES = [
    "ongoing depression", "job crisis", "problems with friends", "academic pressure", "breakup with partner",
    "conflicts with parents", "sleep problems", "appearance anxiety", "school bullying", "issues with children",
    "procrastination", "alcohol abuse", "issues with parents", "other",
]

HELP_REQUEST_CUES = ("what should i do", "any advice", "please help", "how do i deal", "can anyone help")


def ensure_problem_types() -> list[str]:
    if PROBLEM_TYPES_PATH.exists():
        return read_json(PROBLEM_TYPES_PATH)
    from common import write_json

    write_json(PROBLEM_TYPES_PATH, DEFAULT_PROBLEM_TYPES)
    return DEFAULT_PROBLEM_TYPES


def load_empathetic_dialogues(limit: int | None = None) -> list[dict]:
    """Situations from EmpatheticDialogues. Falls back to a local JSONL export when datasets is absent.


    """
    local = DATA / "raw" / "empatheticdialogues.jsonl"
    rows: list[dict] = []
    if local.exists():
        for i, rec in enumerate(read_jsonl(local)):
            situation = (rec.get("situation") or rec.get("prompt") or "").strip()
            rows.append({"source_ref": rec.get("conv_id", f"local-{i}"),
                         "situation_text": situation.replace("_comma_", ","),
                         "emotion_label": rec.get("emotion") or rec.get("context") or "unknown"})
    else:  # pragma: no cover - network path
        from datasets import load_dataset

        ds = load_dataset("facebook/empathetic_dialogues", split="train")
        seen = set()
        for rec in ds:
            key = rec["conv_id"]
            if key in seen:
                continue
            seen.add(key)
            rows.append({"source_ref": key,
                         # `prompt` is the narrated situation; `context` is the 32-way emotion label.
                         "situation_text": rec["prompt"].strip().replace("_comma_", ","),
                         "emotion_label": rec["context"]})
    rows = [r for r in rows if r["situation_text"] and len(r["situation_text"].split()) >= 8]
    rows = [r for r in rows if not any(c in r["situation_text"].lower() for c in HELP_REQUEST_CUES)]
    return dedupe(rows)[: limit or len(rows)]


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = re.sub(r"\W+", " ", r["situation_text"].lower()).strip()[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def classify_problem_type(llm: LLM, situation: str, types: list[str]) -> str:
    prompt = ("Assign this situation exactly one label from the list. Reply with the label text only.\n\n"
              f"Labels: {', '.join(types)}\n\nSituation: {situation}")
    reply = llm.chat(prompt, temperature=0.0).strip().lower()
    for t in types:
        if t.lower() in reply:
            return t
    return "other"


def sustainability_screen(llm: LLM, situation: str) -> tuple[bool, str]:
    """Can a three-step need chain (surface feeling -> intermediate need -> terminal need) live here?"""
    prompt = (
        "A proactive emotional-support simulation needs situations deep enough to support a three-step need "
        "chain: the surface feeling, an intermediate unmet need, and a terminal need the person cannot name "
        "yet (belonging, autonomy, being seen, safety, competence).\n\n"
        f"Situation: {situation}\n\n"
        "Reply exactly: VERDICT: yes|no\nREASON: <one sentence>"
    )
    reply = llm.chat(prompt, temperature=0.0)
    # "yes|no" is the template echoed back, not an answer (v1: 3 of 702 replies, all counted as passes)
    verdict = bool(re.search(r"VERDICT:\s*yes\b(?!\s*\|)", reply, re.I))
    reason = (re.search(r"REASON:\s*(.+)", reply) or [None, "no reason given"])[1]
    return verdict, str(reason).strip()[:200]


def synthesize_situations(llm: LLM, n: int, types: list[str]) -> list[dict]:
    """Fallback route: fully LLM-sampled situations with a handful of seed examples in the prompt."""
    examples = (
        "- I moved cities for a job six months ago and I still eat dinner alone every night.\n"
        "- My brother got engaged and I said the right things and then felt hollow all week.\n"
        "- I got the promotion I asked for and now I dread opening my laptop.\n"
    )
    out = []
    per_type = max(1, n // max(1, len(types)))
    for t in types:
        prompt = (f"Write {per_type} short first-person situations, one per line, no numbering, in the style "
                  f"of these examples:\n{examples}\nEach must fit the problem type '{t}', be narrated rather "
                  f"than a request for advice, and hint at an unmet need without naming it.")
        for line in llm.chat(prompt).splitlines():
            line = line.strip("-* \t")
            if len(line.split()) >= 8:
                out.append({"source_ref": f"synth-{t}-{len(out)}", "situation_text": line,
                            "emotion_label": "unspecified", "problem_type": t})
    return out[:n]


def build(route: str = "seeded", limit: int | None = None, backend: str | None = None,
          out_path: Path = SEEDS_PATH) -> dict:
    types = ensure_problem_types()
    target = limit or int(cfg("corpus.n_profiles_target", default=1000))
    llm = LLM("judge", backend=backend)
    try:
        raw = (load_empathetic_dialogues(target * 2) if route == "seeded"
               else synthesize_situations(llm, target, types))

        def screen(item: tuple[int, dict]) -> dict:
            i, row = item
            ptype = row.get("problem_type") or classify_problem_type(llm, row["situation_text"], types)
            ok, reason = sustainability_screen(llm, row["situation_text"])
            return {
                "seed_id": seq_id("sd", i),
                "source": "empatheticdialogues" if route == "seeded" else "synthetic",
                "source_ref": row["source_ref"],
                "situation_text": row["situation_text"],
                "emotion_label": row.get("emotion_label", "unspecified"),
                "problem_type": ptype,
                "length_tokens": len(row["situation_text"].split()),
                "sustains_chain": ok,
                "screen_reason": reason,
            }

        records = list(pmap(screen, enumerate(raw), llm))
        kept = sum(1 for r in records if r["sustains_chain"])
    finally:
        llm.release()

    survivors = [r for r in records if r["sustains_chain"]]
    survivors = stratify(survivors, target)
    write_jsonl(out_path, survivors)
    stats = {
        "route": route,
        "candidates": len(records),
        "screen_pass_rate": (kept / len(records)) if records else 0.0,
        "released": len(survivors),
        "per_type": dict(Counter(r["problem_type"] for r in survivors)),
    }
    write_stats(stats)
    return stats


def stratify(rows: list[dict], target: int) -> list[dict]:
    """Balance across problem types, then across emotion labels inside a type."""
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["problem_type"], []).append(r)
    for group in by_type.values():
        group.sort(key=lambda r: (r["emotion_label"], r["seed_id"]))
    out: list[dict] = []
    rng = rng_for("seeds", "stratify")
    keys = sorted(by_type)
    while len(out) < target and any(by_type[k] for k in keys):
        for k in keys:
            if by_type[k] and len(out) < target:
                out.append(by_type[k].pop(rng.randrange(len(by_type[k]))))
    return out


def write_stats(stats: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Seed statistics", "",
             f"- route: `{stats['route']}`",
             f"- candidates screened: {stats['candidates']}",
             f"- sustainability screen pass rate: {stats['screen_pass_rate']:.2%}",
             f"- seeds released: {stats['released']}", "", "| problem type | seeds |", "|---|---|"]
    for k, v in sorted(stats["per_type"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    (REPORTS / "seed_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest and screen seed situations.")
    ap.add_argument("--route", choices=["seeded", "synthetic"], default="seeded")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None, help="override configs/models.yaml backend (e.g. echo)")
    ap.add_argument("--out", default=str(SEEDS_PATH))
    args = ap.parse_args()
    stats = build(route=args.route, limit=args.limit, backend=args.backend, out_path=Path(args.out))
    print(f"seeds: {stats['released']} released of {stats['candidates']} screened "
          f"({stats['screen_pass_rate']:.1%} passed) -> {args.out}")
    if not stats["released"]:
        raise SystemExit("no seed passed the sustainability screen - every later stage would run on nothing. "
                         "Check the judge's replies, or try --route synthetic.")


if __name__ == "__main__":
    main()
