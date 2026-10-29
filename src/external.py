
from __future__ import annotations

import argparse
from pathlib import Path

from common import DATA, read_json, write_jsonl

OUT_DIR = DATA / "external"

RESISTANCE_CYCLE = ("low", "medium", "high")


def map_extes(raw: list | dict) -> list[dict]:
    """ExTES rows carry a scenario/description and a strategy-annotated dialogue, no hidden need."""
    rows = raw if isinstance(raw, list) else raw.get("data", [])
    out = []
    for i, rec in enumerate(rows):
        situation = (rec.get("scene") or rec.get("description") or rec.get("situation") or "").strip()
        if not situation:
            continue
        out.append({
            "profile_id": f"x{i:06d}",
            "source": "extes",
            "emotion": rec.get("emotion_type") or rec.get("emotion") or "unspecified",
            "feeling": situation[:240],
            "need_chain": [],
            "terminal_need": "",
            "memory": [{"text": situation, "when_relative": "recently", "salience": 0.7}],
            "persona_surface": {"style": "unspecified", "volunteers": "the situation itself"},
            "persona_hidden": {"terminal_need": "", "resistance_level": RESISTANCE_CYCLE[i % 3],
                               "disclosure_triggers": ["accurate reflection"],
                               "disclosure_blockers": ["premature advice"]},
            "resistance_level": RESISTANCE_CYCLE[i % 3],
            "problem_type": rec.get("problem_type", "other"),
            "ambiguous": False,
            "generated_fields": ["resistance_level", "disclosure_triggers", "disclosure_blockers"],
            "success_rate_applicable": False,
            "note": "ExTES has no annotated underlying need: Success Rate is not computed on this set.",
        })
    return out


def map_es_memeval(raw: list | dict) -> list[dict]:
    """ES-MemEval users are help-seekers across sessions; its memory probes are what transfers here."""
    rows = raw if isinstance(raw, list) else raw.get("data", [])
    out = []
    for i, rec in enumerate(rows):
        persona = rec.get("persona") or rec.get("profile") or {}
        history = rec.get("sessions") or rec.get("history") or []
        out.append({
            "profile_id": f"m{i:06d}",
            "source": "es_memeval",
            "emotion": persona.get("emotion", "unspecified"),
            "feeling": (persona.get("summary") or rec.get("summary") or "")[:240],
            "need_chain": [],
            "terminal_need": rec.get("target_need", "") or "",
            "memory": [{"text": str(s)[:200], "when_relative": f"session {j + 1}", "salience": 0.6}
                       for j, s in enumerate(history[:4])],
            "persona_surface": {"style": "help-seeking", "volunteers": "their situation readily"},
            "persona_hidden": {"terminal_need": rec.get("target_need", "") or "",
                               "resistance_level": "low",
                               "disclosure_triggers": ["being asked"], "disclosure_blockers": []},
            "resistance_level": "low",
            "problem_type": rec.get("problem_type", "other"),
            "ambiguous": False,
            "n_reference_sessions": len(history),
            "generated_fields": ["resistance_level", "disclosure_triggers"],
            "success_rate_applicable": bool(rec.get("target_need")),
            "note": ("ES-MemEval users seek help, so proactivity and reactance are not meaningful here; "
                     "only the memory and need-identification arms transfer."),
        })
    return out


MAPPERS = {"extes": map_extes, "es_memeval": map_es_memeval}


def run(which: str, raw_path: Path, out_dir: Path = OUT_DIR) -> dict:
    if not raw_path.exists():
        raise SystemExit(
            f"{raw_path} not found. Download the set first and record its version in "
            f"data/raw/PROVENANCE.md. If it cannot be obtained, use the fallback named in PLAN Sec. 12.4 "
            f"and write the substitution into reports/substitutions.md.")
    rows = MAPPERS[which](read_json(raw_path))
    out_path = out_dir / f"{which}_profiles.jsonl"
    write_jsonl(out_path, rows)
    return {"set": which, "profiles": len(rows), "path": str(out_path),
            "success_rate_applicable": sum(1 for r in rows if r["success_rate_applicable"])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapt an external evaluation set to our profile schema.")
    ap.add_argument("--set", required=True, choices=sorted(MAPPERS))
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    stats = run(args.set, Path(args.raw), Path(args.out))
    print(f"{stats['set']}: {stats['profiles']} profiles -> {stats['path']} "
          f"({stats['success_rate_applicable']} with an annotated need)")


if __name__ == "__main__":
    main()
