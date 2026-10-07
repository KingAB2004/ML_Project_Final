"""Reply loss and PVI of the fine-tuned models split by session number (1-4), from extra/pvi.py's per-example
files on the validation profiles. Teacher-forced on gold replies: how well each model handles follow-up sessions,
not a live multi-session rollout. Writes pvi_by_session.{png,json} next to the inputs."""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZES = ("0.5B", "3B", "7B")
COLORS = {"0.5B": "#2a78d6", "3B": "#eb6834", "7B": "#1baf7a"}     # dataviz reference slots 1-3 (validated)


def by_session(path: Path) -> dict[int, dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        if not r.get("skipped"):
            groups[int(r["session_id"].rsplit("-s", 1)[1])].append(r)
    out = {}
    for k, rows in sorted(groups.items()):
        tokens = sum(r["n_tokens"] for r in rows)
        out[k] = {"turns": len(rows), "profiles": len({r["profile_id"] for r in rows}),
                  "nll_with": -sum(r["logp_with"] for r in rows) / tokens,
                  "nll_wo": -sum(r["logp_wo"] for r in rows) / tokens,
                  "pvi_bits": st.mean(r["pvi_bits"] for r in rows)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/v3_full1000/analyses/pvi")
    d = Path(ap.parse_args().dir)
    data = {s: by_session(d / f"per_example_{s}.jsonl") for s in SIZES if (d / f"per_example_{s}.jsonl").exists()}
    (d / "pvi_by_session.json").write_text(json.dumps(data, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, title in ((axes[0], "nll_with", "Reply loss, fine-tuned with thoughts (lower is better)"),
                           (axes[1], "pvi_bits", "Bits the thoughts add per turn")):
        for size, rows in data.items():
            xs = list(rows)
            ys = [rows[k][key] for k in xs]
            ax.plot(xs, ys, color=COLORS[size], lw=2, marker="o", ms=8, label=f"Qwen2.5-{size}")
            ax.annotate(size, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=10, color="#52514e")
        ax.set_title(title, fontsize=11, color="#0b0b0b")
        ax.set_xlabel("session with this person", color="#52514e")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.7, 4.5)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(d / "pvi_by_session.png", dpi=160, facecolor="#ffffff")
    for size, rows in data.items():
        print(size, {k: (round(v["nll_with"], 3), round(v["pvi_bits"], 2)) for k, v in rows.items()})


if __name__ == "__main__":
    main()
