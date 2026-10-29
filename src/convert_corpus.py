
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_sft import R_CLOSE, R_OPEN, SUPPORTER_SYSTEM, render_input
from common import DATA, read_json, rng_for, write_jsonl

SFT_DIR = DATA / "sft"
SEEKER_KEYS = ("seeker", "usr", "user", "client", "来访者", "患者")
SUPPORTER_KEYS = ("supporter", "sys", "system", "counselor", "therapist", "assistant", "咨询师")


def normalize_role(raw: str) -> str | None:
    low = (raw or "").strip().lower()
    if any(k in low for k in SEEKER_KEYS):
        return "user"
    if low == "ai" or any(k in low for k in SUPPORTER_KEYS):
        return "supporter"
    return None


def turns_from_esconv(record: dict) -> list[dict]:
    out = []
    for i, utt in enumerate(record.get("dialog") or record.get("dialogue") or []):
        role = normalize_role(utt.get("speaker", ""))
        text = (utt.get("text") or utt.get("content") or "").strip()
        if role and text:
            out.append({"turn_index": len(out), "role": role, "text": text})
    return out


def turns_from_extes(record: dict) -> list[dict]:
    out = []
    for utt in record.get("content") or record.get("dialog") or record.get("conversation") or []:
        if not isinstance(utt, dict):
            continue
        for key, value in utt.items():
            role = normalize_role(key)
            if role and isinstance(value, str) and value.strip():
                out.append({"turn_index": len(out), "role": role, "text": value.strip()})
                break
    return out


PARSERS = {"extes": turns_from_extes}


def examples_from_turns(turns: list[dict], dialogue_id: str) -> list[dict]:
    """One example per supporter turn, in exactly the shape build_sft.py produces."""
    rows = []
    for turn in turns:
        if turn["role"] != "supporter" or turn["turn_index"] == 0:
            continue
        rows.append({
            "session_id": dialogue_id,
            "profile_id": dialogue_id,
            "turn_index": turn["turn_index"],
            "system": SUPPORTER_SYSTEM,
            "input": render_input(turns, turn["turn_index"]),
            "target": f"{R_OPEN}{turn['text']}{R_CLOSE}",
            "ladder_rung": None,
        })
    return rows


def convert(which: str, raw_path: Path, out_dir: Path = SFT_DIR,
            val_fraction: float = 0.1, max_examples: int | None = None) -> dict:
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found - fetch it first (scripts/fetch_data.py)")
    payload = read_json(raw_path)
    records = payload if isinstance(payload, list) else payload.get("data", [])
    parser = PARSERS[which]
    rows, kept, skipped = [], 0, 0
    for i, rec in enumerate(records):
        turns = parser(rec if isinstance(rec, dict) else {"conversation": rec})
        if len(turns) < 4:
            skipped += 1
            continue
        kept += 1
        rows.extend(examples_from_turns(turns, f"{which}-{i:06d}"))

    rng = rng_for(which, "convert_split")
    dialogue_ids = sorted({r["session_id"] for r in rows})
    rng.shuffle(dialogue_ids)
    if max_examples:
        # Size-matched comparison: whole dialogues, in the shuffled order, until max_examples is reached.
        per_dialogue: dict[str, int] = {}
        for r in rows:
            per_dialogue[r["session_id"]] = per_dialogue.get(r["session_id"], 0) + 1
        keep, total = [], 0
        for d in dialogue_ids:
            if total >= max_examples:
                break
            keep.append(d)
            total += per_dialogue[d]
        dialogue_ids, kept = keep, len(keep)
        rows = [r for r in rows if r["session_id"] in set(keep)]
    n_val = max(1, int(len(dialogue_ids) * val_fraction)) if dialogue_ids else 0
    val_ids = set(dialogue_ids[:n_val])
    train = [r for r in rows if r["session_id"] not in val_ids]
    val = [r for r in rows if r["session_id"] in val_ids]

    arm = f"corpus_{which}"
    write_jsonl(out_dir / f"{arm}_train.jsonl", train)
    write_jsonl(out_dir / f"{arm}_val.jsonl", val)
    write_jsonl(out_dir / f"{arm}_test.jsonl", [])
    return {"set": which, "dialogues_kept": kept, "dialogues_skipped": skipped,
            "train_examples": len(train), "val_examples": len(val), "arm": arm,
            "annotations": False}


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten an external corpus into our SFT format.")
    ap.add_argument("--set", required=True, choices=sorted(PARSERS))
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=str(SFT_DIR))
    ap.add_argument("--max-examples", type=int, default=None,
                    help="sample whole dialogues down to about this many examples (train + val)")
    args = ap.parse_args()
    stats = convert(args.set, Path(args.raw), Path(args.out), max_examples=args.max_examples)
    print(f"{stats['set']}: {stats['dialogues_kept']} dialogues kept, {stats['dialogues_skipped']} too "
          f"short; {stats['train_examples']} train / {stats['val_examples']} val examples "
          f"-> arm `{stats['arm']}` (response-only targets: this corpus carries no Analysis/Strategy)")


if __name__ == "__main__":
    main()
