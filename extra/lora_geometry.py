"""Where in the network does the thoughts supervision land? The geometry of the LoRA updates.

For every adapted matrix the update is dW = s * B A (B: out x r, A: r x in, s = lora_alpha / r). Its
singular values come from the r x r core, never forming dW: B = Q_B R_B, A^T = Q_A R_A, then
dW = Q_B (s R_B R_A^T) Q_A^T and the SVD of the core gives U = Q_B U_c, sigma, V = Q_A V_c.

  ||dW||_F, sigma_1            size of the update
  stable rank   ||dW||_F^2 / sigma_1^2
  effective rank  exp(H(p)), p_i = sigma_i / sum sigma (Roy & Vetterli 2007): how many directions are used
  energy@k      sum_{i<=k} sigma_i^2 / ||dW||_F^2
  subspace similarity  phi(k) = ||U_with[:, :k]^T U_wo[:, :k]||_F^2 / k  (Hu et al. 2021, Sec. 7): 1 = the
                two adapters move the same top-k output directions; two random k-dim subspaces of R^out give
                about k / out
CPU, seconds per adapter (only the adapter files are read).

--truncate (GPU): replace every dW by its best rank-k approximation (Eckart-Young) for k in --ks, and
measure the validation loss each time; k = 0 is the base model. How much of rank 32 does the task need?

  python extra/lora_geometry.py                  # every size with both adapters, CPU part only
  python extra/lora_geometry.py --truncate       # + the rank-k loss curve (GPU)

Writes runs/scaling/extra/lora_geometry/: summary.json, per_matrix.jsonl, report.md and plots.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics

import numpy as np

from _shared import (MODEL_SIZES, ROOT, SFT_ARMS, STUDY, adapter_dir, fmt, markdown_table, out_dir, pyplot,
                     read_jsonl, run_path, sizes_with_adapters, write_json, write_jsonl)

KEY = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_([AB])\.weight$")
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SIM_KS = (1, 4, 8)
DEFAULT_KS = (0, 1, 2, 4, 8, 16, 32)


def load_factors(path) -> tuple[dict[tuple[int, str], tuple[np.ndarray, np.ndarray]], float]:
    """{(layer, module): (A, B)} and the scale s = alpha / r, from a saved PEFT adapter."""
    from safetensors.numpy import load_file

    cfg = json.loads((path / "adapter_config.json").read_text())
    tensors = load_file(str(path / "adapter_model.safetensors"))
    parts: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for name, t in tensors.items():
        m = KEY.search(name)
        if m:
            parts.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = t.astype(np.float64)
    return {k: (v["A"], v["B"]) for k, v in parts.items()}, cfg["lora_alpha"] / cfg["r"]


def svd_lowrank(A: np.ndarray, B: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """U (out x r), sigma (r), V (in x r) of s * B @ A without forming the out x in matrix."""
    qb, rb = np.linalg.qr(B)
    qa, ra = np.linalg.qr(A.T)
    uc, sig, vct = np.linalg.svd(s * rb @ ra.T)
    return qb @ uc, sig, qa @ vct.T


def spectrum_stats(sig: np.ndarray) -> dict:
    sig = np.clip(sig, 0, None)
    fro2 = float((sig ** 2).sum())
    p = sig / sig.sum() if sig.sum() > 0 else sig
    h = -float(sum(x * math.log(x) for x in p if x > 0))
    return {"fro": math.sqrt(fro2), "sigma1": float(sig[0]), "stable_rank": fro2 / float(sig[0]) ** 2 if sig[0] else 0.0,
            "effective_rank": math.exp(h), "energy_at_4": float((sig[:4] ** 2).sum()) / fro2 if fro2 else 0.0}


def subspace_similarity(u1: np.ndarray, u2: np.ndarray, k: int) -> float:
    return float(np.linalg.norm(u1[:, :k].T @ u2[:, :k]) ** 2 / k)


def analyse_size(size: str) -> list[dict]:
    decomposed = {}
    for arm in SFT_ARMS:
        factors, s = load_factors(adapter_dir(size, arm))
        decomposed[arm] = {key: svd_lowrank(A, B, s) for key, (A, B) in factors.items()}
    n_layers = 1 + max(layer for layer, _ in decomposed["with_thoughts"])
    rows = []
    for key in sorted(decomposed["with_thoughts"]):
        layer, module = key
        row = {"size": size, "layer": layer, "depth": layer / max(1, n_layers - 1), "module": module}
        for arm in SFT_ARMS:
            row[arm] = spectrum_stats(decomposed[arm][key][1])
        uw, uo = decomposed["with_thoughts"][key][0], decomposed["wo_thoughts"][key][0]
        row["similarity"] = {f"k{k}": subspace_similarity(uw, uo, k) for k in SIM_KS}
        row["similarity_random"] = {f"k{k}": k / uw.shape[0] for k in SIM_KS}
        rows.append(row)
    return rows


def size_summary(rows: list[dict]) -> dict:
    out = {}
    for arm in SFT_ARMS:
        out[arm] = {stat: statistics.fmean(r[arm][stat] for r in rows)
                    for stat in ("fro", "stable_rank", "effective_rank", "energy_at_4")}
        out[arm]["by_module_effective_rank"] = {m: statistics.fmean(r[arm]["effective_rank"] for r in rows if r["module"] == m)
                                                for m in MODULES if any(r["module"] == m for r in rows)}
        # where the update mass sits: share of the total squared norm in each third of the depth
        tot = sum(r[arm]["fro"] ** 2 for r in rows)
        out[arm]["norm_share_by_depth_third"] = [sum(r[arm]["fro"] ** 2 for r in rows if lo <= r["depth"] < hi) / tot
                                                 for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))]
    out["similarity"] = {k: statistics.fmean(r["similarity"][k] for r in rows) for k in rows[0]["similarity"]}
    out["similarity_random"] = {k: statistics.fmean(r["similarity_random"][k] for r in rows) for k in rows[0]["similarity"]}
    out["thoughts_vs_plain_norm_ratio"] = out["with_thoughts"]["fro"] / out["wo_thoughts"]["fro"]
    return out


# --------------------------------------------------------------------------- rank-k truncation (GPU)


def truncation_curve(size: str, arm: str, rows: list[dict], ks) -> dict[int, float]:
    import torch

    import _lm

    model, tok = _lm.load(MODEL_SIZES[size]["base"], str(adapter_dir(size, arm)))
    loras = [m for _, m in model.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})]
    saved = [(m.lora_A["default"].weight.detach().clone(), m.lora_B["default"].weight.detach().clone()) for m in loras]
    svds = []
    for m, (A, B) in zip(loras, saved):
        s = m.scaling["default"]
        qb, rb = torch.linalg.qr(B.float())
        qa, ra = torch.linalg.qr(A.float().T)
        uc, sig, vct = torch.linalg.svd(s * rb @ ra.T)
        svds.append((qb @ uc, sig, qa @ vct.T, s))
    curve = {}
    for k in ks:
        if k == 0:
            with model.disable_adapter():
                curve[0] = _lm.mean_nll(model, tok, rows)
            continue
        for m, (A, B), (u, sig, v, s) in zip(loras, saved, svds):
            r = A.shape[0]
            kk = min(k, r)
            newB = torch.zeros_like(B, dtype=torch.float32)
            newA = torch.zeros_like(A, dtype=torch.float32)
            newB[:, :kk] = u[:, :kk] * (sig[:kk] / s)      # s * newB @ newA = U_k diag(sigma_k) V_k^T
            newA[:kk, :] = v[:, :kk].T
            m.lora_A["default"].weight.data.copy_(newA.to(A.dtype))
            m.lora_B["default"].weight.data.copy_(newB.to(B.dtype))
        curve[k] = _lm.mean_nll(model, tok, rows)
        print(f"  {size} {arm} rank {k}: val loss {curve[k]:.4f}", flush=True)
    _lm.release(model)
    return curve


def rank_needed(curve: dict[int, float], share: float = 0.95) -> int | None:
    """Smallest k that recovers `share` of the loss drop the full adapter achieves over the base."""
    full = curve[max(curve)]
    base = curve.get(0)
    if base is None or base <= full:
        return None
    for k in sorted(curve):
        if k and (base - curve[k]) >= share * (base - full):
            return k
    return None


# --------------------------------------------------------------------------- report


def plots(per_size: dict[str, list[dict]], curves: dict, dest) -> list[str]:
    plt = pyplot()
    if plt is None:
        return []
    made = []
    # 1. effective rank, layer x module, per size and arm
    fig, axes = plt.subplots(len(per_size), 2, figsize=(7, 2.6 * len(per_size)), squeeze=False)
    for i, (size, rows) in enumerate(per_size.items()):
        n_layers = 1 + max(r["layer"] for r in rows)
        for j, arm in enumerate(SFT_ARMS):
            grid = np.full((len(MODULES), n_layers), np.nan)
            for r in rows:
                grid[MODULES.index(r["module"]), r["layer"]] = r[arm]["effective_rank"]
            im = axes[i][j].imshow(grid, aspect="auto", cmap="viridis", vmin=1, vmax=32)
            axes[i][j].set_yticks(range(len(MODULES)), MODULES, fontsize=6)
            axes[i][j].set_xlabel("layer", fontsize=7)
            axes[i][j].set_title(f"{size} {arm}: effective rank", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.savefig(dest / "effective_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    made.append("effective_rank.png")

    # 2. update norm and with/wo similarity against relative depth
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for size, rows in per_size.items():
        depths = sorted({r["depth"] for r in rows})
        for arm, style in zip(SFT_ARMS, ("-", "--")):
            axes[0].plot(depths, [math.sqrt(sum(r[arm]["fro"] ** 2 for r in rows if r["depth"] == d)) for d in depths],
                         style, label=f"{size} {arm}")
        axes[1].plot(depths, [statistics.fmean(r["similarity"]["k4"] for r in rows if r["depth"] == d) for d in depths],
                     label=size)
    axes[0].set_xlabel("relative depth")
    axes[0].set_ylabel("||dW||_F per layer")
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("relative depth")
    axes[1].set_ylabel("phi(k=4): with vs wo thoughts")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "norm_and_similarity_by_depth.png", dpi=150)
    plt.close(fig)
    made.append("norm_and_similarity_by_depth.png")

    if curves:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for (size, arm), c in sorted(curves.items()):
            ks = sorted(k for k in c if k > 0)
            ax.plot(ks, [c[k] for k in ks], "-o" if arm == "with_thoughts" else "--s", ms=3, label=f"{size} {arm}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rank k kept")
        ax.set_ylabel("validation loss")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(dest / "rank_truncation.png", dpi=150)
        plt.close(fig)
        made.append("rank_truncation.png")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="SVD geometry of the LoRA updates, per size and arm.")
    ap.add_argument("--sizes", nargs="*", default=None)
    ap.add_argument("--truncate", action="store_true", help="also the rank-k validation-loss curve (GPU)")
    ap.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    ap.add_argument("--limit", type=int, default=300, help="validation examples per truncation point")
    args = ap.parse_args()

    sizes = sizes_with_adapters(args.sizes)
    if not sizes:
        raise SystemExit("no size has both adapters yet")
    dest = out_dir(run_path(STUDY), "lora_geometry")
    per_size = {s: analyse_size(s) for s in sizes}
    write_jsonl(dest / "per_matrix.jsonl", [r for rows in per_size.values() for r in rows])
    summary = {s: size_summary(rows) for s, rows in per_size.items()}

    curves_path = dest / "truncation.json"
    curves = {tuple(k.split("|")): {int(kk): v for kk, v in c.items()}
              for k, c in (json.loads(curves_path.read_text()) if curves_path.exists() else {}).items()}
    if args.truncate:
        val = [r for r in read_jsonl(ROOT / "data" / "sft" / "with_thoughts_val.jsonl")][: args.limit]
        val_wo = [r for r in read_jsonl(ROOT / "data" / "sft" / "wo_thoughts_val.jsonl")][: args.limit]
        for size in sizes:
            for arm, rows in (("with_thoughts", val), ("wo_thoughts", val_wo)):
                if (size, arm) not in curves or set(args.ks) - set(curves[(size, arm)]):
                    curves[(size, arm)] = truncation_curve(size, arm, rows, args.ks)
                    curves_path.write_text(json.dumps({f"{a}|{b}": c for (a, b), c in curves.items()}, indent=1))
    figs = plots(per_size, curves, dest)
    write_json(dest / "summary.json", {"sizes": summary, "truncation": {f"{a}|{b}": c for (a, b), c in curves.items()},
                                       "rank_needed_95": {f"{a}|{b}": rank_needed(c) for (a, b), c in curves.items()}})

    table = []
    for s, r in summary.items():
        for arm in SFT_ARMS:
            a = r[arm]
            table.append({"size": s, "arm": arm, "mean ||dW||_F": fmt(a["fro"], 4), "stable rank": fmt(a["stable_rank"], 2),
                          "effective rank (/32)": fmt(a["effective_rank"], 2), "energy in top 4": f"{a['energy_at_4']:.1%}",
                          "norm share early/mid/late": " / ".join(f"{x:.0%}" for x in a["norm_share_by_depth_third"])})
    sim = [{"size": s, **{f"phi({k})": f"{r['similarity'][k]:.3f} (random {r['similarity_random'][k]:.4f})"
                          for k in r["similarity"]}, "||dW|| with / wo": fmt(r["thoughts_vs_plain_norm_ratio"])}
           for s, r in summary.items()]
    lines = ["# Geometry of the LoRA updates", "",
             "Per adapted matrix dW = s B A: singular values from the r x r core. Effective rank = exp(entropy of "
             "the normalized singular values), out of r = 32. Norm share = fraction of the summed ||dW||^2 in the "
             "first, middle and last third of the layers.", "", markdown_table(table, list(table[0])), "",
             "## Do the two arms move the same directions?", "",
             "phi(k) = ||U_with[:, :k]^T U_wo[:, :k]||_F^2 / k over the top-k left singular vectors; 1 = same "
             "subspace, random subspaces give about k / d.", "", markdown_table(sim, list(sim[0])), ""]
    if curves:
        ks = sorted({k for c in curves.values() for k in c})
        tr = [{"size": a, "arm": b, **{f"k={k}": fmt(c.get(k), 4) for k in ks}, "k for 95 %": rank_needed(c) or "-"}
              for (a, b), c in sorted(curves.items())]
        lines += ["## Rank-k truncation (validation loss)", "",
                  f"Every dW replaced by its best rank-k approximation (Eckart-Young); k = 0 is the base model. "
                  f"First {args.limit} validation examples. 'k for 95 %' = smallest k recovering 95 % of the full "
                  f"adapter's loss drop.", "", markdown_table(tr, list(tr[0])), ""]
    lines += [f"![{f}]({f})" for f in figs]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {dest / 'report.md'}")


if __name__ == "__main__":
    main()
