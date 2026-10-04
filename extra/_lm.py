"""Teacher-forced log-probabilities from a 4-bit base + LoRA adapter (GPU). Used by pvi.py and lora_geometry.py.

The model is loaded exactly as src/train.py trained it (nf4, double quantization, bf16 compute) and every
example is laid out exactly as train.encode_example laid it out: the chat-templated prompt, then the target
and EOS. torch / transformers / peft are imported on first use, so the CPU parts of the analyses run without
them.
"""
from __future__ import annotations

import math

import _shared  # noqa: F401  (puts src/ on sys.path)


def load(base: str, adapter: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                               bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto",
                                                 dtype=dtype)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def release(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def prompt_text(tok, row: dict) -> str:
    msgs = ([{"role": "system", "content": row["system"]}] if row.get("system") else []) + \
        [{"role": "user", "content": row["input"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def target_logprobs(model, tok, row: dict, scored_from: int = 0, seq_len: int = 2048) -> list[float] | None:
    """Log-probability (nats) of each target token whose text starts at or after character `scored_from` of
    row['target'] (EOS included, as in training). Everything before it is context. None when the example is
    longer than seq_len (training dropped it too) or a token straddles the boundary (the scored spans of two
    layouts would then not be the same token sequence)."""
    import torch

    p = tok(prompt_text(tok, row), add_special_tokens=False)["input_ids"]
    enc = tok(row["target"] + tok.eos_token, add_special_tokens=False, return_offsets_mapping=True)
    t, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(p) + len(t) > seq_len:
        return None
    if any(a < scored_from < b for a, b in offsets):
        return None
    first = next(i for i, (a, _) in enumerate(offsets) if a >= scored_from)
    ids = torch.tensor([p + t], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    pos = torch.arange(len(p) + first, len(p) + len(t), device=model.device)
    lp = torch.log_softmax(logits[pos - 1].float(), dim=-1)          # position i-1 predicts token i
    return lp.gather(1, ids[0, pos].unsqueeze(1)).squeeze(1).tolist()


def mean_nll(model, tok, rows: list[dict]) -> float:
    """Mean over examples of the per-token target NLL: what the Trainer reports as eval_loss at batch 1."""
    vals = []
    for r in rows:
        lp = target_logprobs(model, tok, r)
        if lp:
            vals.append(-sum(lp) / len(lp))
    return sum(vals) / len(vals) if vals else math.nan
