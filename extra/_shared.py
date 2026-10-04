"""Plumbing shared by the extra analyses: paths, loaders, judge-style rating, a few distributions numpy
does not ship, cluster bootstrap, plots.

The extra analyses read what the main pipeline already wrote under runs/<id>/ (dialogues, scores, memory)
and write only under runs/<id>/extra/<analysis>/. They never modify a main-pipeline file.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

EXTRA = Path(__file__).resolve().parent
ROOT = EXTRA.parent
sys.path.insert(0, str(ROOT / "src"))

from common import fill, read_jsonl, rng_for, write_json, write_jsonl  # noqa: E402
from judge import parse_scores                                         # noqa: E402
from report import fmt, markdown_table                                 # noqa: E402

PROMPTS = EXTRA / "prompts"
RUNG_ORDER = ("L0", "L1", "L2", "L3")

# Scaling study: every size fine-tuned with the same recipe on the same 1000-profile SFT files.
# params = non-embedding parameters (Kaplan et al. 2020 fit scaling laws on these), from the Qwen2.5 report.
MODEL_SIZES = {
    "0.5B": {"base": "Qwen/Qwen2.5-0.5B-Instruct", "dir": "runs/scale_qwen2.5_0.5B", "params": 0.36e9},
    "3B": {"base": "Qwen/Qwen2.5-3B-Instruct", "dir": "runs/scale_qwen2.5_3B", "params": 2.77e9},
    "7B": {"base": "Qwen/Qwen2.5-7B-Instruct", "dir": "runs/full_1000", "params": 6.53e9},
}
SFT_ARMS = ("with_thoughts", "wo_thoughts")
STUDY = "runs/scaling"          # where the cross-size analyses write: runs/scaling/extra/<analysis>/

__all__ = ["EXTRA", "ROOT", "PROMPTS", "RUNG_ORDER", "MODEL_SIZES", "SFT_ARMS", "STUDY", "adapter_dir",
           "sizes_with_adapters", "fill", "read_jsonl", "rng_for", "write_json",
           "write_jsonl", "fmt", "markdown_table", "read_extra_prompt", "run_path", "out_dir",
           "load_arm_dialogues", "rate_items", "chi2_sf", "norm_sf", "cluster_bootstrap", "pyplot"]


def read_extra_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def run_path(run: str | Path) -> Path:
    """A run directory given relative to the repository root or to the current directory."""
    p = Path(run)
    return p if p.is_absolute() or p.exists() else ROOT / p


def adapter_dir(size: str, arm: str) -> Path:
    return ROOT / MODEL_SIZES[size]["dir"] / f"adapter_{arm}"


def sizes_with_adapters(sizes: Sequence[str] | None = None) -> list[str]:
    """The sizes (smallest first) whose two adapters (with and wo thoughts) are both trained."""
    return [s for s in (sizes or MODEL_SIZES)
            if all((adapter_dir(s, a) / "adapter_config.json").exists() for a in SFT_ARMS)]


def out_dir(run_dir: Path, name: str) -> Path:
    d = run_dir / "extra" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_arm_dialogues(run_dir: Path, arms: Sequence[str] | None = None,
                       include_calibration: bool = False) -> dict[str, list[dict]]:
    """{arm: sessions} from runs/<id>/dialogues/. The calibration arm is left out unless asked for: its
    profiles are not the test profiles, so pooling it with the test arms would mix populations."""
    out: dict[str, list[dict]] = {}
    for f in sorted((run_dir / "dialogues").glob("*.jsonl")):
        if arms and f.stem not in arms:
            continue
        if not arms and not include_calibration and f.stem.startswith("calib"):
            continue
        out[f.stem] = read_jsonl(f)
    return out


def rate_items(llm: Any, prompt: str, n_items: int, lo: int = 1, hi: int = 7) -> dict[int, int] | None:
    """Judge-style 'number: score' ratings with the same single repair turn judge.Judge.score uses. A
    partial answer is a failure (None), never filled in."""
    expected = list(range(1, n_items + 1))
    text = llm.chat(prompt, temperature=0.0)
    parsed = parse_scores(text, expected, lo, hi)
    if parsed is None:
        repair = (f"{prompt}\n\nYour previous answer was:\n{text}\n\nIt did not score every question. "
                  f"Score ALL questions 1-{n_items}, one per line as 'number: score' with a score from "
                  f"{lo} to {hi}, and nothing else.")
        parsed = parse_scores(llm.chat(repair, temperature=0.0), expected, lo, hi)
    return parsed


# --------------------------------------------------------------------------- distributions


def _gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x): series below a+1, Lentz continued fraction above."""
    if x <= 0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_front))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(log_front) * h)


def chi2_sf(x: float, df: float) -> float:
    """P(X > x) for X ~ chi-square(df)."""
    return 1.0 if df <= 0 else _gamma_q(df / 2.0, x / 2.0)


def norm_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def cluster_bootstrap(clusters: Sequence[Any], stat: Callable[[list], float | None], reps: int = 2000,
                      tag: str = "extra") -> tuple[float | None, float | None]:
    """Percentile 95 % CI, resampling whole clusters (profiles): turns and sessions of one profile are not
    independent, the same resampling unit report.py uses."""
    rng = rng_for(tag, "cluster_bootstrap")
    n = len(clusters)
    vals = []
    for _ in range(reps if n else 0):
        v = stat([clusters[rng.randrange(n)] for _ in range(n)])
        if v is not None and math.isfinite(v):
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def pyplot():
    """matplotlib.pyplot on a file-only backend, or None: plots are optional, the numbers are not."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None
