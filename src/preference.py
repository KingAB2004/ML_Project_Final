"""Preference alignment (DPO): the third comparison approach to our pipeline.

Our approach keeps the fine-tuned model as it is and puts a gate around it; steering pushes its activations at run
time. DPO changes the weights instead: the fine-tuned listener is trained to prefer its own less intrusive replies
over its more intrusive ones, so no gate or extra call is needed when it runs.

  sample  1200 training-split contexts (data/sft/with_thoughts_train.jsonl, sessions 1-4, training people only); the
          SFT model (runs/full_1000/adapter_with_thoughts) writes 4 replies to each at temperature 0.9.
  rank    every reply is scored on the IP scale (prompts/metric_ip.md) by the base Qwen 7B in the critic role, through
          Ollama - NOT by the Mistral judge that scores the evaluation, so the model cannot be trained to please the
          judge it is measured with. Pair = least vs most intrusive reply of a context, kept when they differ by at
          least one scale point on average (1/6 normalised) and their replies are within 1.5x in length (so "shorter"
          is not what gets learned).
  train   DPO (Rafailov et al. 2023) on the SFT adapter: reference = the SFT adapter's log-probs, computed once before
          training; loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))), beta 0.1, 1 epoch, LoRA lr 2e-5.
          5 % of pairs held out for preference accuracy.
  final   (scripts/preference.sh) SFT and DPO adapters on the 100 test people, multi-session, same session seeds.
  report  DPO vs SFT (paired), vs steering, cells A-D and step 12.

    python src/preference.py sample --out runs/preference
    python src/preference.py rank   --out runs/preference --backend ollama
    python src/preference.py train  --out runs/preference
    python src/preference.py report --out runs/preference
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from build_sft import parse_target
from common import DATA, ROOT, append_jsonl, existing_ids, read_jsonl, write_json

SFT = "runs/full_1000/adapter_with_thoughts"
K = 4


def contexts(n: int, seed: int) -> list[dict]:
    rows = read_jsonl(DATA / "sft" / "with_thoughts_train.jsonl")
    for i, r in enumerate(rows):
        r["id"] = f"{r['session_id']}#{r['turn_index']}"
    return random.Random(seed).sample(rows, n)


def sample(out: Path, n: int, seed: int, adapter: str) -> None:
    import torch

    from common import load_config
    from llm import LLM

    path = out / "candidates.jsonl"
    done = existing_ids(path, "id")
    todo = [r for r in contexts(n, seed) if r["id"] not in done]
    print(f"{len(todo)} of {n} contexts to sample")
    if not todo:
        return
    load_config("models")["roles"]["supporter"]["adapter"] = adapter
    llm = LLM("supporter", backend="transformers")
    model, tok = llm.backend.model, llm.backend.tokenizer
    try:
        for i, r in enumerate(todo):
            msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["input"]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(text, return_tensors="pt").to(model.device)
            torch.manual_seed(seed + i)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=384,
                                     num_return_sequences=K, pad_token_id=tok.eos_token_id)
            raws = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
            append_jsonl(path, {"id": r["id"], "system": r["system"], "input": r["input"], "candidates": raws,
                                "responses": [parse_target(x)["response"] for x in raws]})
            if i % 50 == 0:
                print(f"{i}/{len(todo)}")
    finally:
        model = None
        llm.release()


def transcript(context: str, reply: str) -> str:
    convo = context.split("Conversation so far:\n", 1)[-1].split("\n\nWrite the supporter's next turn.")[0]
    return f"{convo}\nsupporter: {reply}"


def rank(out: Path, backend: str | None) -> None:
    from judge import Judge
    from llm import LLM, pmap

    rows = read_jsonl(out / "candidates.jsonl")
    llm = LLM("critic", backend=backend)
    # The judge's scoring routine (its prompt, parser, and the re-ask when items are missing - the 7B critic often
    # answers item 1 only), run on the critic model rather than the evaluation judge.
    ranker = Judge(llm=llm)

    def score(row: dict) -> dict:
        ips = []
        for i, reply in enumerate(row["responses"]):
            res = ranker.score("ip", f"{row['id']}~{i}", transcript(row["input"], reply), {}, target_text=reply) \
                if reply else None
            ips.append(res.normalized if res and not res.parse_failed else None)
        return {**row, "ip": ips}

    try:
        scored = list(pmap(score, rows, llm))
    finally:
        llm.release()
    pairs, skipped = [], {"unparsed": 0, "small gap": 0, "length": 0}
    for r in scored:
        ok = [(ip, i) for i, ip in enumerate(r["ip"]) if ip is not None and r["responses"][i]]
        if len(ok) < 2:
            skipped["unparsed"] += 1
            continue
        (lo, c), (hi, j) = min(ok), max(ok)
        if hi - lo < 1 / 6:
            skipped["small gap"] += 1
            continue
        lc, lr = len(r["responses"][c].split()), len(r["responses"][j].split())
        if not (1 / 1.5 <= lc / max(1, lr) <= 1.5):
            skipped["length"] += 1
            continue
        pairs.append({"id": r["id"], "system": r["system"], "input": r["input"], "chosen": r["candidates"][c],
                      "rejected": r["candidates"][j], "ip_chosen": lo, "ip_rejected": hi})
    with open(out / "pairs.jsonl", "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats = {"contexts": len(scored), "pairs": len(pairs), "skipped": skipped,
             "mean_ip_chosen": sum(p["ip_chosen"] for p in pairs) / max(1, len(pairs)),
             "mean_ip_rejected": sum(p["ip_rejected"] for p in pairs) / max(1, len(pairs)),
             "ranker": llm.spec.get("ollama_tag") or llm.spec["model"]}
    write_json(out / "pairs_stats.json", stats)
    print(stats)


def train(out: Path, adapter: str, beta: float, lr: float, seed: int) -> None:
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from common import load_config
    from train import encode_example

    dest = out / "adapter_dpo"
    if (dest / "adapter_model.safetensors").exists():
        print(f"{dest} exists, kept")
        return
    pairs = read_jsonl(out / "pairs.jsonl")
    if len(pairs) < 20:
        raise SystemExit(f"only {len(pairs)} preference pairs in {out / 'pairs.jsonl'}: check pairs_stats.json")
    random.Random(seed).shuffle(pairs)
    n_val = max(1, len(pairs) // 20)
    val, tr = pairs[:n_val], pairs[n_val:]
    model_id = load_config("models")["roles"]["supporter"]["model"]
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto",
                                                 torch_dtype=dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    for m in model.modules():   # no dropout: the policy and the reference must score identically at step 0
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.train()               # gradient checkpointing only runs in train mode

    enc = {}
    for p in pairs:
        sides = [encode_example(tok, {"system": p["system"], "input": p["input"], "target": p[k]}, 2048)
                 for k in ("chosen", "rejected")]
        if all(sides):
            enc[p["id"]] = dict(zip(("chosen", "rejected"), sides))
    val, tr = [p for p in val if p["id"] in enc], [p for p in tr if p["id"] in enc]

    def logp(p: dict, side: str) -> torch.Tensor:
        ex = enc[p["id"]][side]
        ids = torch.tensor([ex["input_ids"]], device=model.device)
        tgt = torch.tensor(ex["labels"][1:], device=model.device)
        mask = tgt != -100
        logits = model(input_ids=ids).logits[0, :-1][mask].float()   # response positions only
        return torch.gather(F.log_softmax(logits, -1), 1, tgt[mask].unsqueeze(-1)).sum()

    with torch.no_grad():   # reference = the SFT adapter before any update
        ref = {p["id"]: (logp(p, "chosen").item(), logp(p, "rejected").item()) for p in val + tr}

    def margin(p: dict) -> torch.Tensor:
        rc, rr = ref[p["id"]]
        return beta * ((logp(p, "chosen") - rc) - (logp(p, "rejected") - rr))

    def evaluate() -> tuple[float, float]:
        with torch.no_grad():
            m = [margin(p).item() for p in val]
        return sum(x > 0 for x in m) / len(m), sum(m) / len(m)

    acc0, m0 = evaluate()
    log = [{"step": 0, "val_pref_acc": acc0, "val_margin": m0}]
    print(log[-1])
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr)
    accum, run_loss = 8, 0.0
    for i, p in enumerate(tr, 1):
        loss = -F.logsigmoid(margin(p)) / accum
        loss.backward()
        run_loss += loss.item()
        if i % accum == 0 or i == len(tr):
            opt.step()
            opt.zero_grad()
        if i % 160 == 0 or i == len(tr):
            acc, m = evaluate()
            log.append({"step": i, "train_loss": run_loss / (i / accum), "val_pref_acc": acc, "val_margin": m})
            print(log[-1])
    model.save_pretrained(str(dest))
    write_json(out / "dpo_log.json", {"beta": beta, "lr": lr, "pairs_train": len(tr), "pairs_val": len(val),
                                      "reference": adapter, "log": log})


def report(out: Path, dest: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from recovery_cost import diff_ci, mean_by_person
    from steering import arm_stats

    rows = [(out, "pref_mono_sft"), (out, "pref_mono_dpo")]
    steer = ROOT / "runs/steering"
    rows += [(steer, p.stem) for p in sorted((steer / "dialogues").glob("steer_mono_L*_c*.jsonl"))]
    rows += [(ROOT / "runs/baselines_1000", c) for c in
             ("cellA_mono_ungated", "cellB_mono_gated", "cellC_dec_ungated", "cellD_dec_gated")]
    rows.append((ROOT / "runs/sft_agents", "sft_dec_gated_mem"))
    rows = [(r, t) for r, t in rows if (r / "dialogues" / f"{t}.jsonl").exists()]
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    lines = [__doc__.split("\n\n")[0], ""]
    if (out / "pairs_stats.json").exists():
        lines.append(f"pairs: {(out / 'pairs_stats.json').read_text().strip()}")
    if (out / "dpo_log.json").exists():
        lines.append(f"DPO held-out preference accuracy by step: "
                     f"{[(x['step'], round(x['val_pref_acc'], 3)) for x in json.loads((out / 'dpo_log.json').read_text())['log']]}")
    lines += ["", "arm  sessions  IP  PRI  Success  counterfactual PRI [95% CI]  fallback  words"]
    for r, t in rows:
        s = arm_stats(r, t)
        cf = (f"{s['cf_pri']:+.4f} [{s['cf_ci'][0]:+.4f}, {s['cf_ci'][1]:+.4f}]" if s.get("cf_pri") is not None
              else "n/a")
        lines.append(f"{t}  {s['sessions']}  {f(s['ip'])}  {f(s['pri'])}  {f(s['success'])}  {cf}  "
                     f"{s['fallback']:.3f}  {f(s['words'])}")
    lines += ["", "differences, by-person bootstrap 95% CI (b - a)"]
    base = (out, "pref_mono_dpo")
    for a in [(out, "pref_mono_sft")] + [x for x in rows if x[0] != out]:
        for scale in ("ip", "pri", "success"):
            if (a[0] / "scores" / a[1] / f"{scale}.jsonl").exists() and \
                    (base[0] / "scores" / base[1] / f"{scale}.jsonl").exists():
                d, lo, hi = diff_ci(mean_by_person(a[0], a[1], scale), mean_by_person(base[0], base[1], scale))
                lines.append(f"   pref_mono_dpo - {a[1]}  {scale:8s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines += ["", "SFT and DPO share people and session seeds (paired). Steering ran on the same backend but other "
              "seeds; cells A-D ran on Ollama, step 12 with four agents - those comparisons are unpaired."]
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("pairs_stats.json", "dpo_log.json"):
        if (out / name).exists():
            (dest / name).write_text((out / name).read_text())
    (dest / "dpo_comparison.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Preference alignment (DPO) of the fine-tuned listener.")
    ap.add_argument("cmd", choices=["sample", "rank", "train", "report"])
    ap.add_argument("--out", default="runs/preference")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", default=SFT)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--report-dir", default=str(ROOT / "temp_results/18_preference_alignment_dpo"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "sample":
        sample(out, args.n, args.seed, args.adapter)
    elif args.cmd == "rank":
        rank(out, args.backend)
    elif args.cmd == "train":
        train(out, args.adapter, args.beta, args.lr, args.seed)
    else:
        report(out, Path(args.report_dir))


if __name__ == "__main__":
    main()
