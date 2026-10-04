"""QLoRA supervised fine-tuning of the supporter (PLAN Sec. 9.2).

4-bit base, low-rank adapters, gradient checkpointing, loss on target tokens only. Sized for a 12 GB card:
micro-batch 1 with accumulation, sequence length 2048, paged optimizer.

Two adapters come out of this stage - `with_thoughts` and `wo_thoughts` - and that pair is the annotation
ablation the SOP promises as a reproduction check.

Nothing is imported from torch/transformers/peft until main() runs, so the rest of the project stays
importable on a machine with no GPU.

  python src/train.py --arm with_thoughts
  python src/train.py --arm wo_thoughts
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from common import DATA, cfg, read_jsonl

SFT_DIR = DATA / "sft"


def load_rows(arm: str, split: str, sft_dir: Path = SFT_DIR) -> list[dict]:
    """Works for our two arms and for any `corpus_<name>` produced by src/convert_corpus.py."""
    return [{"system": r.get("system", ""), "input": r["input"], "target": r["target"]}
            for r in read_jsonl(sft_dir / f"{arm}_{split}.jsonl")]


def encode_example(tokenizer, row: dict, seq_len: int) -> dict | None:
    """Prompt through the model's own chat template - exactly what llm.TransformersBackend sends at
    inference - so the adapter learns the format it will be served in. Loss on target tokens only.
    Returns None when the example does not fit: cutting it would drop the target (an all-masked example
    gives a NaN loss) or the chat header (a malformed prompt)."""
    msgs = ([{"role": "system", "content": row["system"]}] if row.get("system") else []) + \
        [{"role": "user", "content": row["input"]}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t = tokenizer(row["target"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > seq_len:
        return None
    return {"input_ids": p + t, "labels": [-100] * len(p) + t, "attention_mask": [1] * (len(p) + len(t))}


def build_trainer(arm: str, model_id: str, out_dir: Path, resume_adapter: str | None,
                  train_rows: list[dict], val_rows: list[dict]):  # pragma: no cover - needs GPU
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)

    tcfg = cfg("train")
    seq_len = int(tcfg["seq_len"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # bf16 needs an Ampere-or-newer GPU; older cards (V100, T4) train in fp16 instead of crashing.
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto",
                                                 torch_dtype=dtype)
    # Not prepare_model_for_kbit_training: it upcasts every non-quantized weight to fp32, and Qwen2.5-7B's
    # 152k-vocab embedding + lm_head alone then take ~4.4 GB more - out of memory on a 12 GB card at step 2.
    # Kept in bf16 (fp16); only checkpointing and input grads (what checkpointing needs) are switched on.
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if resume_adapter:
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=int(tcfg["lora_r"]), lora_alpha=int(tcfg["lora_alpha"]),
            lora_dropout=float(tcfg["lora_dropout"]), bias="none", task_type="CAUSAL_LM",
            target_modules=list(tcfg["target_modules"])))
    model.print_trainable_parameters()

    def encode_all(rows: list[dict], split: str) -> list[dict]:
        out = [e for e in (encode_example(tokenizer, r, seq_len) for r in rows) if e is not None]
        if len(out) < len(rows):
            print(f"[{split}] dropped {len(rows) - len(out)} of {len(rows)} examples longer than {seq_len} tokens")
        return out

    train_ds = Dataset.from_list(encode_all(train_rows, "train"))
    val_enc = encode_all(val_rows, "val") if val_rows else []
    val_ds = Dataset.from_list(val_enc) if val_enc else None

    # transformers 5 dropped warmup_ratio; its warmup_steps takes a float < 1 as the ratio instead.
    warmup = float(tcfg["warmup_ratio"])
    warmup_kw = ({"warmup_ratio": warmup} if "warmup_ratio" in inspect.signature(TrainingArguments).parameters
                 else {"warmup_steps": warmup})
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg["micro_batch_size"]),
        # the default eval batch is 8: 8 sequences of 152k-vocab fp32 logits OOMed the 12 GB card at the
        # end-of-epoch evaluation, after training itself had fit. Training's batch size is known to fit.
        per_device_eval_batch_size=int(tcfg["micro_batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]),
        num_train_epochs=float(tcfg["epochs"]),
        learning_rate=float(tcfg["lr"]),
        **warmup_kw,
        lr_scheduler_type="cosine",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        load_best_model_at_end=val_ds is not None,
        report_to=[],
    )
    return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                   data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))


def free_gpu_from_ollama() -> None:
    """Ollama keeps a model resident for minutes after its last request (the judge: ~7.5 GB plus KV cache);
    training needs that VRAM. Best effort: with no Ollama running there is nothing to free."""
    import urllib.request

    from llm import ollama_host

    try:
        with urllib.request.urlopen(f"{ollama_host()}/api/ps", timeout=5) as resp:
            names = [m["name"] for m in json.load(resp).get("models", [])]
        for name in names:
            req = urllib.request.Request(f"{ollama_host()}/api/generate", method="POST",
                                         data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
            print(f"unloaded {name} from Ollama to free the GPU")
    except Exception:
        pass


def main() -> None:  # pragma: no cover - needs GPU
    ap = argparse.ArgumentParser(description="QLoRA SFT for the supporter.")
    ap.add_argument("--arm", default="with_thoughts",
                    help="with_thoughts | wo_thoughts | corpus_<name> from src/convert_corpus.py")
    ap.add_argument("--model", default=None, help="defaults to configs/models.yaml roles.supporter.model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume-adapter", default=None,
                    help="continue an existing adapter instead of starting a new one")
    ap.add_argument("--sft-dir", default=str(SFT_DIR))
    args = ap.parse_args()

    from common import load_config, new_run_dir, write_json

    model_id = args.model or load_config("models")["roles"]["supporter"]["model"]
    run_dir = Path(args.out) if args.out else new_run_dir(f"train_{args.arm}")
    out_dir = run_dir / f"adapter_{args.arm}"
    train_rows = load_rows(args.arm, "train", Path(args.sft_dir))
    val_rows = load_rows(args.arm, "val", Path(args.sft_dir))
    if not train_rows:
        raise SystemExit(f"no training rows for arm '{args.arm}' - run build_sft.py first")

    free_gpu_from_ollama()
    trainer = build_trainer(args.arm, model_id, out_dir, args.resume_adapter, train_rows, val_rows)
    trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    write_json(run_dir / "train_summary.json", {
        "arm": args.arm, "model": model_id, "adapter": str(out_dir),
        "n_train": len(train_rows), "n_val": len(val_rows),
        "resumed_from": args.resume_adapter, "config": cfg("train"),
        # the loss curve (train every logging step, eval per epoch) and which epoch the saved adapter is
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
    })
    print(f"adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
