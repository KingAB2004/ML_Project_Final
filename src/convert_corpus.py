"""Convert an external support corpus into our SFT format, so the same base model can be fine-tuned on
each one and the arms become comparable (the shape of the baseline paper's Tables 3 and 4).

Their comparison fine-tunes one base model on ESConv, ExTES, ESCoT, SoulChat and SMILE, then scores every
resulting supporter on the same user profiles. This module supplies the "training dataset" half of that:
each corpus is flattened into the same prompt/target files `train.py` already consumes.

Those corpora carry no Analysis/Strategy annotation of our kind, so their targets are response-only - the
equivalent of our `w/o thoughts` arm. That asymmetry is real and is printed in the report rather than
hidden: a corpus without annotations cannot be trained with them.

  python src/convert_corpus.py --set esconv --raw data/raw/esconv.json
  python src/convert_corpus.py --set extes  --raw data/raw/extes.json
  python src/convert_corpus.py --set smile  --raw data/raw/smile.json
"""
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


def turns_from_smile(record: dict) -> list[dict]:
    """SMILE-style releases are usually a flat list of 'role: text' lines."""
    out = []
    lines = record.get("dialogue") if isinstance(record.get("dialogue"), list) else \
        str(record.get("conversation", "")).splitlines()
    for line in lines:
        text = line if isinstance(line, str) else str(line)
        if ":" not in text and "：" not in text:
            continue
        head, _, body = text.replace("：", ":").partition(":")
        role = normalize_role(head)
        if role and body.strip():
            out.append({"turn_index": len(out), "role": role, "text": body.strip()})
    return out


PARSERS = {"esconv": turns_from_esconv, "extes": turns_from_extes, "smile": turns_from_smile}


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
            val_fraction: float = 0.1) -> dict:
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
    args = ap.parse_args()
    stats = convert(args.set, Path(args.raw), Path(args.out))
    print(f"{stats['set']}: {stats['dialogues_kept']} dialogues kept, {stats['dialogues_skipped']} too "
          f"short; {stats['train_examples']} train / {stats['val_examples']} val examples "
          f"-> arm `{stats['arm']}` (response-only targets: this corpus carries no Analysis/Strategy)")


if __name__ == "__main__":
    main()
