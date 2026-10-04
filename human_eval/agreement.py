"""Judge-human agreement (PLAN Sec. 11.4 control 3).

Weighted Cohen's kappa between the two human raters, Krippendorff's alpha across all three raters (judge
included, ordinal difference function), and Spearman's rho between the judge mean and the human mean.

Every headline IP/PRI number in the report carries this figure, so this module writes it where report.py
looks for it: runs/<run_id>/scores/agreement.json.

  python human_eval/agreement.py --run runs/<run_id> --metric ip
"""
from __future__ import annotations

import argparse
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common import read_json, read_jsonl, write_json  # noqa: E402


def mean_score(scores: dict) -> float | None:
    vals = [float(v) for v in scores.values() if v is not None]
    return statistics.fmean(vals) if vals else None


def weighted_kappa(a: list[float], b: list[float], lo: int = 1, hi: int = 7) -> float | None:
    """Quadratic-weighted kappa on rounded ordinal ratings."""
    if not a or len(a) != len(b):
        return None
    cats = list(range(lo, hi + 1))
    n = len(a)
    ar = [min(hi, max(lo, round(x))) for x in a]
    br = [min(hi, max(lo, round(x))) for x in b]
    obs = {(i, j): 0 for i in cats for j in cats}
    for x, y in zip(ar, br):
        obs[(x, y)] += 1
    ra = {c: sum(1 for x in ar if x == c) / n for c in cats}
    rb = {c: sum(1 for y in br if y == c) / n for c in cats}
    denom_max = (hi - lo) ** 2
    num = den = 0.0
    for i in cats:
        for j in cats:
            w = ((i - j) ** 2) / denom_max
            num += w * obs[(i, j)] / n
            den += w * ra[i] * rb[j]
    return 1.0 - num / den if den else None


def krippendorff_alpha(matrix: list[list[float | None]]) -> float | None:
    """Ordinal-ish alpha with a squared difference function. Rows are items, columns raters."""
    pairs = []
    for row in matrix:
        vals = [v for v in row if v is not None]
        for x, y in combinations(vals, 2):
            pairs.append((float(x), float(y)))
    if not pairs:
        return None
    do = statistics.fmean((x - y) ** 2 for x, y in pairs)
    allv = [float(v) for row in matrix for v in row if v is not None]
    de = statistics.fmean((x - y) ** 2 for x, y in combinations(allv, 2)) if len(allv) > 1 else 0.0
    return 1.0 - do / de if de else None


def spearman_rho(a: list[float], b: list[float]) -> float | None:
    if not a or len(a) != len(b) or len(a) < 2:
        return None

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def run(run_dir: Path, metric: str = "ip", forms_dir: Path | None = None) -> dict:
    forms_dir = forms_dir or Path(__file__).resolve().parent / "forms" / metric
    sampled = {r["item_id"]: r for r in read_jsonl(forms_dir / "sampled.jsonl")}
    rater_files = sorted(p for p in forms_dir.glob("rater_*.json"))
    if not sampled or not rater_files:
        return {"error": f"no sampled items or rater sheets in {forms_dir}", "metric": metric}

    human: dict[str, list[float]] = {}
    for path in rater_files:
        for row in read_json(path):
            m = mean_score(row.get("scores", {}))
            if m is not None:
                human.setdefault(row["item_id"], []).append(m)

    item_ids = [i for i in sampled if len(human.get(i, [])) >= 2]
    if not item_ids:
        return {"error": "raters have not filled in at least two ratings per item yet", "metric": metric,
                "items_pending": len(sampled)}

    a = [human[i][0] for i in item_ids]
    b = [human[i][1] for i in item_ids]
    judge_raw = [sampled[i]["judge_score"] * 6 + 1 for i in item_ids]   # back to the 1-7 scale
    human_mean = [statistics.fmean(human[i]) for i in item_ids]

    out = {
        "metric": metric,
        "n_items": len(item_ids),
        "weighted_kappa": weighted_kappa(a, b),
        "krippendorff_alpha": krippendorff_alpha([[human[i][0], human[i][1], judge_raw[k]]
                                                  for k, i in enumerate(item_ids)]),
        "spearman_rho": spearman_rho(judge_raw, human_mean),
        "judge_mean": statistics.fmean(judge_raw),
        "human_mean": statistics.fmean(human_mean),
        "note": ("Both raters helped write the rubric, so a high kappa is partly shared training; raters "
                 "were blind to arm and to each other, and this limitation is stated in the report."),
    }
    write_json(run_dir / "scores" / "agreement.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge-human agreement on the stratified subsample.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default="ip", choices=["ip", "pri"])
    ap.add_argument("--forms", default=None)
    args = ap.parse_args()
    out = run(Path(args.run), args.metric, Path(args.forms) if args.forms else None)
    if "error" in out:
        print(f"agreement not computed: {out['error']}")
        return
    print(f"n={out['n_items']} kappa={out['weighted_kappa']} alpha={out['krippendorff_alpha']} "
          f"rho={out['spearman_rho']}")


if __name__ == "__main__":
    main()
