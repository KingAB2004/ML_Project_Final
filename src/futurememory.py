"""Memory-aware fine-tuning data, Option B (chosen 7 Oct): the existing annotated corpus with its dialogues and
replies unchanged, plus the need-state brief on every supporter turn, and the thoughts of sessions 2+
re-annotated with that brief in view so the analysis can draw on what was remembered.

Three resumable steps; run them in order, or `all`:
  replay    each person's sessions in order through one need-state memory: the base Analyzer writes after every
            supporter turn, beliefs fade and prune over each real gap, and every supporter turn records the brief
            it stood at (sessions.replay_corpus)
  annotate  sessions 2+ re-annotated with the brief visible (annotate.annotate_session). Session 1 keeps its
            thoughts: its brief only restates the same conversation. A turn whose re-annotation fails to parse
            keeps its old thoughts.
  sft       build_sft for both arms; the brief replaces "[memory] none." in every input

    python src/futurememory.py all --splits train,val
    python src/train.py --arm with_thoughts --sft-dir data/sft_futuremem --out runs/futuremem_7B
    python src/train.py --arm wo_thoughts   --sft-dir data/sft_futuremem --out runs/futuremem_7B

Evaluate with the sftmem_* arms (configs/arms.yaml): brief given vs withheld shows whether the model uses it.
Regenerating the replies with the brief as well is Option C (sessions.py --memory).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from annotate import ANNOTATED_PATH, annotate_session, apply_annotations
from build_sft import build, split_assignment
from common import DATA, append_jsonl, read_jsonl
from llm import LLM, pmap
from sessions import replay_corpus

CORPUS_DIR = DATA / "corpus_futuremem"
SFT_DIR = DATA / "sft_futuremem"


def replay_step(src: Path, out_dir: Path, splits: set[str], backend: str | None = None,
                limit: int | None = None, llm: LLM | None = None) -> dict:
    ids = {s["profile_id"] for s in read_jsonl(src)}
    keep = {pid for pid, name in split_assignment(sorted(ids)).items() if name in splits}
    return replay_corpus(src, out_dir / "sessions_replay.jsonl", backend=backend, limit=limit,
                         profiles=keep, llm=llm)


def reannotate(llm: LLM, session: dict) -> dict:
    if session.get("session_index", 1) == 1:
        return session
    records = [r for r in annotate_session(llm, session) if not r["parse_failed"]]
    return apply_annotations(session, records)


def annotate_step(out_dir: Path, backend: str | None = None, limit: int | None = None,
                  llm: LLM | None = None) -> dict:
    src, out = out_dir / "sessions_replay.jsonl", out_dir / "sessions_annotated.jsonl"
    done = {s["session_id"] for s in read_jsonl(out)}
    todo = [s for s in read_jsonl(src) if s["session_id"] not in done][: limit or None]
    own = llm is None
    llm = llm or LLM("judge", backend=backend)          # the annotator of the original corpus (annotate.run)
    try:
        for session in pmap(lambda s: reannotate(llm, s), todo, llm):
            append_jsonl(out, session)
    finally:
        if own:
            llm.release()
    return {"sessions": len(todo), "reannotated": sum(s.get("session_index", 1) > 1 for s in todo)}


def sft_step(out_dir: Path, sft_dir: Path) -> dict:
    return {arm: build(arm, out_dir / "sessions_annotated.jsonl", sft_dir)["counts"]
            for arm in ("with_thoughts", "wo_thoughts")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Option B memory-aware SFT data from the existing corpus.")
    ap.add_argument("step", choices=["replay", "annotate", "sft", "all"])
    ap.add_argument("--source", default=str(ANNOTATED_PATH), help="the annotated corpus to add memory to")
    ap.add_argument("--splits", default="train,val", help="whose sessions to replay (training needs train,val)")
    ap.add_argument("--out", default=str(CORPUS_DIR))
    ap.add_argument("--sft-dir", default=str(SFT_DIR))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--limit", type=int, default=None, help="people for replay, sessions for annotate")
    args = ap.parse_args()
    out = Path(args.out)
    if args.step in ("replay", "all"):
        print("replay", replay_step(Path(args.source), out, set(args.splits.split(",")), args.backend, args.limit))
    if args.step in ("annotate", "all"):
        print("annotate", annotate_step(out, args.backend, args.limit))
    if args.step in ("sft", "all"):
        print("sft", sft_step(out, Path(args.sft_dir)))


if __name__ == "__main__":
    main()
