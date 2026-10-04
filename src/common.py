"""Shared plumbing: config, ids, seeds, file IO, spans, transcripts.

Every stage in this project reads files and writes files (PLAN.md R1). This module owns the
conventions those files follow so no stage has to invent its own.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import yaml

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

RUNGS = ("L0", "L1", "L2", "L3")
STATUSES = ("hypothesis", "confirmed", "disconfirmed", "resolved")


# --------------------------------------------------------------------------- config

_CONFIG_CACHE: dict[str, dict] = {}


def load_config(name: str = "default") -> dict:
    """Read configs/<name>.yaml once and memoize it."""
    if name not in _CONFIG_CACHE:
        with open(ROOT / "configs" / f"{name}.yaml", "r", encoding="utf-8") as fh:
            _CONFIG_CACHE[name] = yaml.safe_load(fh)
    return _CONFIG_CACHE[name]


def cfg(path: str, name: str = "default", default: Any = None) -> Any:
    """Dotted lookup: cfg("memory.decay_half_life_days")."""
    node: Any = load_config(name)
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def read_prompt(filename: str) -> str:
    with open(PROMPTS / filename, "r", encoding="utf-8") as fh:
        return fh.read()


def fill(template: str, **kwargs: Any) -> str:
    """str.format with a readable error when a placeholder is missing.

    Prompt files write literal JSON braces as {{ }}, so format() is safe on them.
    """
    try:
        return template.format(**kwargs)
    except KeyError as exc:  # pragma: no cover - surfaced in tests as a clear message
        raise KeyError(f"prompt placeholder {exc} was not supplied") from exc


# --------------------------------------------------------------------------- ids and seeds


def utc_stamp(fmt: str = "%Y%m%dT%H%M%SZ") -> str:
    return datetime.now(timezone.utc).strftime(fmt)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def run_id(arm: str = "run") -> str:
    return f"{utc_stamp()}-{git_sha()}-{arm}"


def new_run_dir(arm: str = "run", config_name: str = "default") -> Path:
    """Create runs/<run_id>/ and snapshot the config + resolved environment into it."""
    path = RUNS / run_id(arm)
    for sub in ("dialogues", "scores", "memory", "conformal", "cache"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    write_json(path / "config.snapshot.json", {
        "arm": arm,
        "config": load_config(config_name),
        "models": load_config("models"),
        "git_sha": git_sha(),
        "created_at": utc_stamp(),
    })
    return path


def stable_hash(*parts: Any) -> str:
    blob = "\x1f".join(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def derive_seed(record_id: str, stage: str, root_seed: int | None = None) -> int:
    """Per-record seed so a record regenerates identically wherever it sits in a batch (PLAN 4.2)."""
    root = cfg("seed", default=0) if root_seed is None else root_seed
    digest = stable_hash(root, record_id, stage)
    return int(digest[:8], 16)


def rng_for(record_id: str, stage: str) -> random.Random:
    return random.Random(derive_seed(record_id, stage))


def seq_id(prefix: str, n: int, width: int = 6) -> str:
    return f"{prefix}{n:0{width}d}"


# --------------------------------------------------------------------------- file io


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, so a concurrent reader of the LLM cache never sees a half-written file.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def existing_ids(path: str | Path, key: str) -> set[str]:
    """Ids already written, so a stage can skip them and be resumable (PLAN R5)."""
    return {rec[key] for rec in read_jsonl(path) if key in rec}


def require_fields(record: dict, fields: Sequence[str], where: str) -> None:
    missing = [f for f in fields if f not in record]
    if missing:
        raise ValueError(f"{where}: missing required field(s) {missing}")


# --------------------------------------------------------------------------- spans


@dataclass
class Span:
    """A verbatim quote from a user turn. The unit of grounding (PLAN R4)."""

    session_id: str
    turn_index: int
    quote: str
    char_start: int = -1
    char_end: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Span":
        return Span(
            session_id=d.get("session_id", ""),
            turn_index=int(d.get("turn_index", -1)),
            quote=d.get("quote", ""),
            char_start=int(d.get("char_start", -1)),
            char_end=int(d.get("char_end", -1)),
        )


def locate_span(haystack: str, quote: str) -> tuple[int, int] | None:
    idx = haystack.find(quote)
    if idx < 0:
        return None
    return idx, idx + len(quote)


# --------------------------------------------------------------------------- transcripts


def render_transcript(turns: Sequence[dict], numbered: bool = False) -> str:
    """Flat transcript for judges and prompts. Roles are always spelled out."""
    lines = []
    for turn in turns:
        role = "supporter" if turn.get("role") == "supporter" else "seeker"
        prefix = f"[{turn.get('turn_index')}] " if numbered else ""
        lines.append(f"{prefix}{role}: {turn.get('text', '').strip()}")
    return "\n".join(lines)


def user_turns(turns: Sequence[dict]) -> list[dict]:
    return [t for t in turns if t.get("role") == "user"]


def supporter_turns(turns: Sequence[dict]) -> list[dict]:
    return [t for t in turns if t.get("role") == "supporter"]


def mirrored_history(turns: Sequence[dict], speaker: str) -> list[dict]:
    """Chat history from one side's point of view: the other side is 'user', this side 'assistant'.

    Kept from the upstream two-history design; it is the one part of COCOON's chat loop worth reusing.
    """
    out = []
    for turn in turns:
        role = "assistant" if turn.get("role") == speaker else "user"
        out.append({"role": role, "content": turn.get("text", "")})
    return out


def to_int(value: Any, default: int = 0, lo: int | None = None, hi: int | None = None) -> int:
    """An integer out of model JSON: 2, "2", "2 - intermediate need" all give 2; anything else the default.
    One malformed field must not crash a stage that has hours of work in flight."""
    m = re.search(r"-?\d+", str(value)) if value is not None and not isinstance(value, bool) else None
    n = int(m.group()) if m else default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+(?:\.\d+)?", str(value)) if value is not None else None
        return float(m.group()) if m else default


def _norm_line(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def fresh_turn(chat: Callable[..., str], prompt: str, turns: Sequence[dict], label: str, **kw: Any) -> str:
    """One generated turn that does not copy an earlier line of the session.

    A 7B model sometimes loses track of which speaker it is, or falls back on a stock line it already
    used, and repeats an earlier turn verbatim. Resample once with a nudge (bypassing the cache, which
    would return the same copy); if the retry still repeats, keep it - the filter or the judge decides.
    A leading "<label>:" the model adds is stripped.
    """
    def clean(text: str) -> str:
        return re.sub(rf"^\s*{label}\s*:\s*", "", (text or "").strip(), flags=re.I)

    seen = {_norm_line(t.get("text", "")) for t in turns}
    text = clean(chat(prompt, **kw))
    if _norm_line(text) in seen:
        kw = {**kw, "use_cache": False}
        text = clean(chat(f"{prompt}\n\nYour previous attempt repeated a line that was already said. "
                          f"Write something new.", **kw))
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


# --------------------------------------------------------------------------- misc


def retry(fn: Callable[[], Any], attempts: int = 3, delay: float = 1.0, label: str = "call") -> Any:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network/model faults
            last = exc
            if i + 1 < attempts:
                time.sleep(delay * (i + 1))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


def chunks(items: Sequence[Any], size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def normalize_scale(mean_score: float, lo: int = 1, hi: int = 7) -> float:
    """Map a 1-7 instrument mean onto [0,1]."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (mean_score - lo) / (hi - lo)))


def token_estimate(text: str) -> int:
    """Cheap proxy used only for prompt-budget guards, never for billing."""
    return max(1, len(text) // 4)


def truncate_tokens(text: str, budget: int) -> str:
    limit = budget * 4
    return text if len(text) <= limit else text[: limit - 3] + "..."
