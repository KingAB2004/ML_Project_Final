"""Phase 1.2-1.3 - need chains and hidden personas (PLAN Sec. 8.2, 8.3).

Three validators decide whether a profile is usable, and the middle one matters most: if the terminal need
is a paraphrase of the surface feeling, Success Rate becomes meaningless and the whole proactive premise is
vacuous. So triviality is checked mechanically AND by the judge, and failures are written out with a reason
rather than silently dropped.

  python src/profiles.py --limit 200
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from common import DATA, cfg, fill, read_prompt, read_jsonl, rng_for, to_int, write_jsonl
from llm import LLM, pmap
from memory import similarity

SEEDS_PATH = DATA / "seeds" / "seeds.jsonl"
PROFILES_PATH = DATA / "profiles" / "profiles.jsonl"
REJECTS_PATH = DATA / "profiles" / "profiles_rejected.jsonl"


def resistance_plan(n: int) -> list[str]:
    """Stratify resistance 1:2:1 so every arm sees all three levels and subgroups are analysable."""
    strata = cfg("corpus.resistance_strata", default={"low": 1, "medium": 2, "high": 1})
    order: list[str] = []
    for level, weight in strata.items():
        order += [level] * int(weight)
    return [order[i % len(order)] for i in range(n)]


def check_chain(chain: list[dict], llm: LLM | None = None) -> tuple[bool, str]:
    """Boundedness, non-triviality, inferability."""
    if len(chain) != int(cfg("corpus.need_chain_depth", default=3)):
        return False, f"chain has {len(chain)} nodes, expected 3"
    texts = [(c.get("text") or "").strip() for c in chain]
    if not all(texts):
        return False, "empty chain node text"
    sim = similarity(texts[0], texts[2])
    if sim > float(cfg("corpus.triviality_similarity_max", default=0.75)):
        return False, f"terminal need paraphrases the surface feeling (similarity {sim:.2f})"
    if llm is not None:
        prompt = ("Two statements from one person's need chain.\n"
                  f"Surface feeling: {texts[0]}\nTerminal need: {texts[2]}\n\n"
                  "Answer two questions exactly in this format:\n"
                  "RESTATEMENT: yes|no   (is the terminal need merely a restatement of the surface feeling?)\n"
                  "REACHABLE: yes|no     (could a skilled listener reach the terminal need from what this "
                  "person could plausibly say?)")
        reply = llm.chat(prompt, temperature=0.0)
        if re.search(r"RESTATEMENT:\s*yes\b(?!\s*\|)", reply, re.I):   # "yes|no" = the template echoed
            return False, "judge: terminal need is a restatement"
        if re.search(r"REACHABLE:\s*no", reply, re.I):
            return False, "judge: terminal need is not reachable from the situation"
    return True, "ok"


def build_profile(llm: LLM, seed: dict, resistance: str, ambiguous: bool) -> dict:
    out = llm.structured(
        fill(read_prompt("profile_seed.md"),
             situation_text=seed["situation_text"], emotion_label=seed.get("emotion_label", ""),
             problem_type=seed.get("problem_type", "other"), ambiguous=str(ambiguous).lower()),
        required=("need_chain", "feeling"),
        max_tokens=1024,  # a whole profile is ~400-600 tokens; the judge role's 256 cut it mid-JSON
    )
    # Model JSON is untrusted: a chain that is not a list, or nodes that are bare strings, must become a
    # profile check_chain rejects with a reason - not a crash that stops the stage.
    raw_chain = out.get("need_chain")
    chain = ([n if isinstance(n, dict) else {"text": str(n)} for n in raw_chain]
             if isinstance(raw_chain, list) else [])
    for i, node in enumerate(chain):
        node["text"] = str(node.get("text") or "").strip()
        node["node_id"] = f"nd{i:03d}"
        node["depth"] = to_int(node.get("depth"), i)
        node["parent_id"] = None if i == 0 else chain[i - 1]["node_id"]
    hidden = dict(out["persona_hidden"]) if isinstance(out.get("persona_hidden"), dict) else {}
    terminal = chain[-1]["text"] if chain else ""
    hidden.update({"terminal_need": terminal, "resistance_level": resistance})
    hidden.setdefault("disclosure_triggers", ["accurate reflection", "patience", "validation"])
    hidden.setdefault("disclosure_blockers", ["premature advice", "naming an undisclosed feeling",
                                              "repeated direct questioning"])
    return {
        "seed_id": seed["seed_id"],
        "route": "seeded" if seed.get("source") == "empatheticdialogues" else "synthetic",
        "emotion": out.get("emotion", seed.get("emotion_label", "")),
        "feeling": out.get("feeling", ""),
        "need_chain": chain,
        "terminal_need": terminal,
        "memory": out["memory"] if isinstance(out.get("memory"), list) else [],
        "persona_surface": out.get("persona_surface", {}) or {},
        "persona_hidden": hidden,
        "resistance_level": resistance,
        "problem_type": seed.get("problem_type", "other"),
        "ambiguous": ambiguous,
        "created_by": {"model": llm.spec["model"], "backend": llm.backend_name},
        "parse_failed": bool(out.get("parse_failed")),
    }


def build(limit: int | None = None, backend: str | None = None, seeds_path: Path = SEEDS_PATH,
          out_path: Path = PROFILES_PATH) -> dict:
    seeds = read_jsonl(seeds_path)
    if limit:
        seeds = seeds[:limit]
    levels = resistance_plan(len(seeds))
    # Seeds arrive round-robin over problem types, so assigning the 1:2:1 plan by position tied resistance
    # to problem type (v1: appearance anxiety 4 high vs 37 medium). Shuffle; the exact counts are kept.
    rng_for("profiles", "resistance").shuffle(levels)
    amb_fraction = float(cfg("corpus.ambiguous_profile_fraction", default=0.10))
    llm = LLM("judge", backend=backend)
    kept, rejected = [], []
    try:
        def make(item: tuple[int, dict]) -> tuple[dict, bool, str]:
            i, seed = item
            pid = f"p{i:06d}"
            ambiguous = rng_for(pid, "profile").random() < amb_fraction
            prof = build_profile(llm, seed, levels[i], ambiguous)
            prof["profile_id"] = pid
            ok, why = check_chain(prof["need_chain"], llm)
            return prof, ok, why

        for prof, ok, why in pmap(make, enumerate(seeds), llm):
            if ok and not prof["parse_failed"]:
                kept.append(prof)
            else:
                prof["reject_reason"] = why if ok is False else "unparseable generator output"
                rejected.append(prof)
    finally:
        llm.release()
    write_jsonl(out_path, kept)
    write_jsonl(REJECTS_PATH, rejected)
    return {
        "kept": len(kept), "rejected": len(rejected),
        "ambiguous": sum(1 for p in kept if p["ambiguous"]),
        "resistance": dict(Counter(p["resistance_level"] for p in kept)),
        "routes": dict(Counter(p["route"] for p in kept)),
    }


def split_profiles(profiles: list[dict], fractions: dict | None = None) -> dict[str, list[dict]]:
    """Split BY PROFILE so no profile appears in two splits, plus a disjoint calibration slice."""
    fr = fractions or cfg("train.split_fractions", default={"train": 0.8, "val": 0.1, "test": 0.1})
    rng = rng_for("split", "profiles")
    shuffled = list(profiles)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * float(fr.get("train", 0.8)))
    n_val = int(n * float(fr.get("val", 0.1)))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    n_cal = max(1, len(train) // 10) if train else 0
    return {"train": train[n_cal:], "val": val, "test": test, "calibration": train[:n_cal]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build hidden profiles with bounded need chains.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--seeds", default=str(SEEDS_PATH))
    ap.add_argument("--out", default=str(PROFILES_PATH))
    ap.add_argument("--split", action="store_true", help="also write train/val/test/calibration splits")
    args = ap.parse_args()
    stats = build(limit=args.limit, backend=args.backend, seeds_path=Path(args.seeds),
                  out_path=Path(args.out))
    print(f"profiles kept {stats['kept']}, rejected {stats['rejected']}, "
          f"resistance {stats['resistance']}")
    if args.split:
        splits = split_profiles(read_jsonl(args.out))
        for name, rows in splits.items():
            path = Path(args.out).with_name(f"profiles_{name}.jsonl")
            write_jsonl(path, rows)
            print(f"  {name}: {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
