"""Scale rendering, score parsing, aggregation, and the judge-separation guard.

Every psychological instrument in this project is scored through here, so the redaction rules and the
"judge is a different model" rule (PLAN Sec. 11.4 controls 1 and 2) live in exactly one place.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence

from common import cfg, fill, normalize_scale, read_prompt, render_transcript
from llm import LLM, role_spec

SCALE_PROMPTS = {
    "aels": "scale_aels.md",
    "crs": "scale_crs.md",
    "rac": "scale_rac.md",
    "success": "scale_success.md",
    "basic": "scale_basic.md",
    "ip": "metric_ip.md",
    "pri": "metric_pri.md",
}

# What each scale may see about the hidden profile. Anything not listed is redacted (PLAN 6.10).
ALLOWED_INFO = {
    "aels": ("feeling", "terminal_need"),
    "crs": ("feeling", "terminal_need"),
    "rac": ("feeling", "terminal_need"),
    "basic": ("feeling",),
    "success": ("terminal_need",),   # ground truth is the point of this scale
    "ip": ("feeling",),              # never the need, never the resistance level
    "pri": ("feeling",),             # never the resistance level: the metric would become circular
}
NEVER_SHOWN = ("resistance_level", "persona_hidden", "disclosure_triggers", "disclosure_blockers")


class JudgeSeparationError(RuntimeError):
    """The judge must not be the supporter or the simulator (PLAN Sec. 11.4 control 2)."""


def assert_judge_separate(judge_role: str = "judge",
                          other_roles: Sequence[str] = ("supporter", "simulator", "generator")) -> None:
    judge = role_spec(judge_role)
    for other in other_roles:
        try:
            spec = role_spec(other)
        except KeyError:
            continue
        if spec["model"] == judge["model"]:
            raise JudgeSeparationError(
                f"judge model '{judge['model']}' is also configured for role '{other}'. PRI reads the "
                f"simulator's own turn, so a shared model would rate its own output."
            )


def build_info(profile: dict, scale: str) -> tuple[str, list[str]]:
    """Render the 'info' block for a scale, and report what was withheld."""
    allowed = ALLOWED_INFO.get(scale, ())
    lines, redacted = [], []
    if "feeling" in allowed and profile.get("feeling"):
        lines.append(f"feeling: {profile['feeling']}")
    else:
        redacted.append("feeling")
    if "terminal_need" in allowed and profile.get("terminal_need"):
        lines.append(f"need: {profile['terminal_need']}")
    else:
        redacted.append("terminal_need")
    redacted.extend(k for k in NEVER_SHOWN if k in profile)
    return ("\n".join(lines) or "(withheld)"), sorted(set(redacted))


def parse_scores(text: str, expected: Sequence[int], lo: int = 1, hi: int = 7) -> dict[int, int] | None:
    """Parse 'N: score' lines. Returns None if any expected item is missing or out of range.

    Regex shape follows the upstream scorers (the one transferable idea there), but a partial parse is a
    failure here rather than a silently averaged subset.
    """
    found: dict[int, int] = {}
    # Optional "Question"/"Item" before the number and "Score" after the separator: judges write
    # "Question 1: Score - 4" as often as "1: 4".
    label = r"(?:(?:question|item|q)\s*)?\(?(\d{1,2})\)?\s*[.:：-]+\s*(?:score\s*[.:：=-]*\s*)?([1-7])\b"
    for num, score in re.findall(r"(?mi)^\s*" + label, text):
        found[int(num)] = int(score)
    # Items not on their own line (e.g. "(1). 4  (2): 5") fill only the gaps the line-anchored pass left.
    for num, score in re.findall(r"(?i)(?:^|\s)" + label, text):
        found.setdefault(int(num), int(score))
    out = {}
    for item in expected:
        if item not in found or not (lo <= found[item] <= hi):
            return None
        out[item] = found[item]
    return out


def expected_items(scale: str) -> list[int]:
    prompt = read_prompt(SCALE_PROMPTS[scale])
    block = prompt.split("[Rating Scale]")[0]
    return sorted({int(n) for n in re.findall(r"(?m)^\s*(\d{1,2})\.\s", block)})


@dataclass
class ScaleResult:
    target: str
    scale: str
    items: dict[int, float] = field(default_factory=dict)
    mean: float = 0.0
    normalized: float = 0.0
    n_samples: int = 0
    judge_model: str = ""
    context_redactions: list[str] = field(default_factory=list)
    parse_failed: bool = False

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["items"] = {str(k): v for k, v in self.items.items()}
        return d


class Judge:
    """Wraps one resident judge model. Load it once, sweep a whole directory of dialogues, release it."""

    def __init__(self, llm: LLM | None = None, role: str = "judge", **llm_kwargs: Any):
        assert_judge_separate(role)
        self.llm = llm or LLM(role, **llm_kwargs)
        self.samples = int(cfg("judge.samples_per_item", default=3))
        self.lo = int(cfg("judge.scale_min", default=1))
        self.hi = int(cfg("judge.scale_max", default=7))

    def score(self, scale: str, target: str, dialogue: str, profile: dict,
              target_text: str = "") -> ScaleResult:
        items_expected = expected_items(scale)
        info, redacted = build_info(profile, scale)
        template = read_prompt(SCALE_PROMPTS[scale])
        kwargs = {"info": info, "diag": dialogue}
        if "{target}" in template:
            kwargs["target"] = target_text
        prompt = fill(template, **kwargs)

        samples: list[dict[int, int]] = []
        temperature = float(cfg("judge.temperature", default=0.0))
        for k in range(self.samples):
            text = self.llm.chat(prompt, temperature=temperature, use_cache=(k == 0))
            parsed = parse_scores(text, items_expected, self.lo, self.hi)
            if not parsed:
                # Judges often score only the items they find relevant. Ask once more for every item; a
                # partial answer is still rejected, never filled in.
                repair = (f"{prompt}\n\nYour previous answer was:\n{text}\n\nIt did not score every question. "
                          f"Score ALL questions {items_expected[0]}-{items_expected[-1]}, one per line as "
                          f"'number: score' with a score from {self.lo} to {self.hi}, and nothing else.")
                text = self.llm.chat(repair, temperature=temperature, use_cache=(k == 0))
                parsed = parse_scores(text, items_expected, self.lo, self.hi)
            if parsed:
                samples.append(parsed)
        result = ScaleResult(target=target, scale=scale, judge_model=self.llm.spec["model"],
                             context_redactions=redacted, n_samples=len(samples))
        if not samples:
            result.parse_failed = True
            return result
        result.items = {i: statistics.fmean(s[i] for s in samples) for i in items_expected}
        result.mean = statistics.fmean(result.items.values())
        result.normalized = normalize_scale(result.mean, self.lo, self.hi)
        return result

    # -- convenience wrappers used by the stages ---------------------------
    def score_session(self, scale: str, session: dict, profile: dict) -> ScaleResult:
        return self.score(scale, session["session_id"], render_transcript(session["turns"]), profile)

    def score_turn(self, scale: str, session: dict, turn_index: int, profile: dict) -> ScaleResult:
        turns = session["turns"]
        prefix = [t for t in turns if t["turn_index"] <= turn_index]
        target_text = next((t["text"] for t in turns if t["turn_index"] == turn_index), "")
        return self.score(scale, f"{session['session_id']}#{turn_index}",
                          render_transcript(prefix), profile, target_text=target_text)
