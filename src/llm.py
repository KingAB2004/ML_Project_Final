
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from common import (
    ROOT,
    cfg,
    load_config,
    read_json,
    stable_hash,
    write_json,
)

Messages = list[dict]

_RESIDENT: "LLM | None" = None
_LOG_LOCK = threading.Lock()

PARALLEL_BACKENDS = ("ollama", "echo")


def pmap(fn: Callable[[Any], Any], items: Iterable[Any], llm: Any) -> Iterator[Any]:
    """map(fn, items), run `parallel.workers` at a time when `llm`'s backend allows it. Results come back
    in input order, so callers write output exactly as the serial loop did. Records are independent and
    seeded by id (common.rng_for), so the order calls complete in does not change what they produce.

    For Ollama the server must also accept that many requests: OLLAMA_NUM_PARALLEL >= parallel.workers.
    """
    items = list(items)
    workers = int(cfg("parallel.workers", default=1))
    if workers <= 1 or len(items) <= 1 or getattr(llm, "backend_name", None) not in PARALLEL_BACKENDS:
        yield from map(fn, items)
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield from pool.map(fn, items)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class ResidencyError(RuntimeError):
    """Raised when a second distinct model is asked for while one is still resident."""


def role_spec(role: str) -> dict:
    models = load_config("models")
    if role not in models["roles"]:
        raise KeyError(f"unknown model role '{role}' - add it to configs/models.yaml")
    spec = dict(models["roles"][role])
    spec.setdefault("adapter", None)
    spec.setdefault("temperature", 0.7)
    spec.setdefault("top_p", 1.0)
    spec.setdefault("max_tokens", 512)
    spec.setdefault("quantization", None)
    spec["role"] = role
    return spec


def residency_key(spec: dict) -> str:
    return (f"{spec.get('ollama_tag') or spec['model']}|{spec.get('quantization')}|"
            f"{spec.get('adapter')}")


# --------------------------------------------------------------------------- backends


class EchoBackend:
    """Deterministic stub. Returns shape-correct output so logic can be exercised without a GPU.

    It is not a model: it recognises three prompt shapes (a rating scale, a JSON schema, free text) and
    answers in the matching format. Any pipeline that works against echo is wired correctly; whether it
    works *well* is a question only the real backend can answer.
    """

    name = "echo"

    def __init__(self, spec: dict):
        self.spec = spec

    def generate(self, system: str | None, messages: Messages, use_adapter: bool = True,
                 **_: Any) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        if "[Question List]" in prompt:
            return self._scale_reply(prompt)
        if "Return JSON only" in prompt or "Return JSON" in prompt:
            return json.dumps(echo_stub(self.spec.get("role", ""), prompt))
        return echo_text(self.spec.get("role", ""), prompt)

    @staticmethod
    def _scale_reply(prompt: str) -> str:
        items = re.findall(r"^\s*(\d+)\.\s", prompt, flags=re.M)
        numbers = sorted({int(i) for i in items}) or [1]
        mid = 5 if "Strongly Disagree" in prompt else 2
        return "\n".join(f"{n}: {mid}" for n in numbers)


ECHO_SEEKER_LINES = (
    "I don't know. It's fine, mostly. Work has just been a lot lately.",
    "Not much to tell, honestly. Same week as every other week.",
    "I keep meaning to slow down and then I don't. Anyway.",
    "It's not a big deal. Other people have it worse than me.",
    "I haven't really told anyone how bad the last few days were.",
    "Maybe. I'm not sure what I'd even ask for if I did.",
)
ECHO_SUPPORTER_LINES = (
    "That sounds heavy to carry on your own.",
    "Thanks for saying that much. I'm not going anywhere.",
    "A week like that would wear anyone down.",
    "I'm glad you told me that part.",
    "That makes sense to me, hearing how the week went.",
    "I'll stay with that for a moment rather than rushing past it.",
)


def echo_text(role: str, prompt: str) -> str:
    """Deterministic but varied, so a stubbed dialogue does not look like one repeated line.

    Varying matters: filter.py rejects verbatim repetition, so a constant stub would make every
    end-to-end dry run fail for a reason that has nothing to do with the pipeline.
    """
    if "VERDICT: yes|no" in prompt:  # seeds.sustainability_screen; anything else fails every candidate
        return "VERDICT: yes\nREASON: echo stub accepts every candidate."
    pool = ECHO_SEEKER_LINES if role == "simulator" else ECHO_SUPPORTER_LINES
    if role not in ("simulator", "generator", "supporter", "analyzer", "strategist", "critic"):
        return "Acknowledged."
    index = int(stable_hash(prompt)[:8], 16) % len(pool)
    for k in range(len(pool)):
        line = pool[(index + k) % len(pool)]
        if line not in prompt:
            return line
    return f"{pool[index]} ({prompt.count(pool[index]) + 1})"


def echo_stub(role: str, prompt: str) -> dict:
    """Shape-correct stub per JSON-emitting role."""
    stubs: dict[str, dict] = {
        "analyzer": {
            "emotional_state": {"label": "flat, worn down", "gloss": "low mood with fatigue",
                                "evidence_spans": [], "confidence": 0.5},
            "implicit_needs": [{"text": "to be seen without having to explain", "depth": 1,
                                "evidence_spans": [], "confidence": 0.4, "status_hint": "hypothesis"}],
            "resistance_estimate": {"level": "medium", "evidence_spans": [], "confidence": 0.5},
            "open_questions": ["who, if anyone, knows how bad this week was"],
        },
        "strategist": {
            "plan": "reflect the tiredness and leave an opening, no interpretation yet",
            "justification_chain": ["seeker says work is a lot", "no need disclosed", "stay shallow"],
            "target_node_id": None,
            "requested_rung": "L0",
        },
        "critic": {
            "item_scores": {str(i): 2 for i in range(1, 8)},
            "violations": [],
            "draft_rung": "L0",
            "feedback": "",
        },
    }
    if role in stubs:
        return stubs[role]
    if "ladder_rung" in prompt and "analysis" in prompt:
        return {"analysis": "tired and unwilling to name it", "strategy": "reflect and wait",
                "ladder_rung": "L0", "user_disclosure_depth": 1}
    if '"rung"' in prompt:
        return {"rung": "L1", "reason": "stub"}
    if "need_chain" in prompt:
        return {
            "emotion": "low, worn down",
            "feeling": "I just feel like I'm running on empty",
            "need_chain": [
                {"depth": 0, "text": "running on empty"},
                {"depth": 1, "text": "wants rest without guilt"},
                {"depth": 2, "text": "wants to matter to someone beyond what they produce"},
            ],
            "memory": [{"text": "worked two weekends in a row", "when_relative": "last two weeks",
                        "salience": 0.8}],
            "persona_surface": {"style": "understated", "verbosity": "brief", "tone": "dry",
                                "volunteers": "work logistics"},
            "persona_hidden": {"disclosure_triggers": ["accurate reflection"],
                               "disclosure_blockers": ["quick advice"]},
            "resistance_level": "medium",
            "applied": "intensified",
            "notes": "stub",
        }
    return {}


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def ollama_host() -> str:
    import os

    return (os.environ.get("OLLAMA_HOST")
            or load_config("models").get("ollama_host")
            or DEFAULT_OLLAMA_HOST).rstrip("/")


def ollama_tag(spec: dict) -> str:
    """The Ollama model tag for a role, e.g. `qwen2.5:7b-instruct`.

    Falls back to the Hugging Face id's last path segment lowercased, which is usually wrong - so every
    role that will run on Ollama should carry an explicit `ollama_tag` in configs/models.yaml.
    """
    return spec.get("ollama_tag") or spec["model"].split("/")[-1].lower()


def ollama_payload(spec: dict, system: str | None, messages: Messages, over: dict | None = None) -> dict:
    """Build the /api/chat request body. Pure function, so it can be checked without a server."""
    over = over or {}
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages
    ]
    options = {
        "temperature": float(over.get("temperature", spec.get("temperature", 0.7))),
        "top_p": float(over.get("top_p", spec.get("top_p", 1.0))),
        "num_predict": int(over.get("max_tokens", spec.get("max_tokens", 512))),
    }
    if spec.get("max_model_len"):
        options["num_ctx"] = int(spec["max_model_len"])
    return {"model": ollama_tag(spec), "messages": msgs, "stream": False, "options": options,
            "keep_alive": spec.get("keep_alive", "5m")}


class OllamaBackend:
    """Talks to a local Ollama server. Nothing proprietary, nothing to pip install.

    Residency is the server's business, but the project's one-model rule still applies: `release()` sends
    keep_alive=0 so the weights are dropped before the next role is loaded, which is what keeps a 12 GB
    card from holding the generator and the judge at once.
    """

    name = "ollama"

    def __init__(self, spec: dict):
        self.spec = spec
        self.host = ollama_host()

    def _post(self, path: str, body: dict, timeout: int = 600) -> dict:
        import json as _json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            f"{self.host}{path}", data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach Ollama at {self.host} ({exc}). Start it with `ollama serve`, or pull the "
                f"model with `ollama pull {ollama_tag(self.spec)}`.") from exc

    def generate(self, system: str | None, messages: Messages, use_adapter: bool = True,
                 **over: Any) -> str:
        data = self._post("/api/chat", ollama_payload(self.spec, system, messages, over))
        return (data.get("message", {}) or {}).get("content", "").strip()

    def unload(self) -> None:
        try:
            self._post("/api/chat", {"model": ollama_tag(self.spec), "messages": [], "keep_alive": 0},
                       timeout=30)
        except Exception:  # pragma: no cover - unloading is best effort
            pass


class TransformersBackend:  # pragma: no cover - requires GPU
    name = "transformers"

    def __init__(self, spec: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.spec = spec
        quant = None
        if spec.get("quantization") in ("nf4", "bnb", "4bit"):
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model"])
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["model"], quantization_config=quant, device_map="auto", torch_dtype="auto"
        )
        # What serves the adapter (use_adapter=True). Normally the resident model itself.
        self.tuned, self.tuned_tokenizer = self.model, self.tokenizer
        if spec.get("adapter"):
            from peft import PeftModel

            base = spec.get("adapter_base")
            if base and base != spec["model"]:
                # Scaling study: the adapter belongs to a smaller base. It is loaded beside the resident model,
                # which keeps serving every other role (the simulator above all), so every size meets the
                # same seeker. Two 4-bit models, 7B + 3B, fit a 12 GB card.
                small = AutoModelForCausalLM.from_pretrained(
                    base, quantization_config=quant, device_map="auto", torch_dtype="auto")
                self.tuned = PeftModel.from_pretrained(small, spec["adapter"]).eval()
                self.tuned_tokenizer = AutoTokenizer.from_pretrained(base)
            else:
                self.model = PeftModel.from_pretrained(self.model, spec["adapter"])
                self.tuned = self.model
        self.model.eval()

    def generate(self, system: str | None, messages: Messages, use_adapter: bool = True,
                 **over: Any) -> str:
        import contextlib

        import torch

        model, tok = (self.tuned, self.tuned_tokenizer) if use_adapter else (self.model, self.tokenizer)
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to(model.device)
        temp = float(over.get("temperature", self.spec["temperature"]))
        # An adapter that is loaded but disabled gives the base model back without a second load.
        adapter_ctx = (model.disable_adapter() if (not use_adapter and hasattr(model, "disable_adapter"))
                       else contextlib.nullcontext())
        with torch.no_grad(), adapter_ctx:
            out = model.generate(
                **enc,
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                top_p=float(over.get("top_p", self.spec["top_p"])),
                max_new_tokens=int(over.get("max_tokens", self.spec["max_tokens"])),
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class VLLMBackend:  # pragma: no cover - requires GPU
    name = "vllm"

    def __init__(self, spec: dict):
        from vllm import LLM as VLLMEngine

        self.spec = spec
        self.engine = VLLMEngine(
            model=spec["model"],
            quantization=spec.get("quantization"),
            max_model_len=spec.get("max_model_len", 4096),
            gpu_memory_utilization=spec.get("gpu_memory_utilization", 0.90),
            enable_lora=bool(spec.get("adapter")),
        )
        self.tokenizer = self.engine.get_tokenizer()

    def _render(self, system: str | None, messages: Messages) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def generate(self, system: str | None, messages: Messages, **over: Any) -> str:
        return self.generate_batch([(system, messages)], **over)[0]

    def _lora(self, use_adapter: bool):
        if not (use_adapter and self.spec.get("adapter")):
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("supporter", 1, self.spec["adapter"])

    def generate_batch(self, pairs: Sequence[tuple[str | None, Messages]], use_adapter: bool = True,
                       **over: Any) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=float(over.get("temperature", self.spec["temperature"])),
            top_p=float(over.get("top_p", self.spec["top_p"])),
            max_tokens=int(over.get("max_tokens", self.spec["max_tokens"])),
        )
        prompts = [self._render(s, m) for s, m in pairs]
        outs = self.engine.generate(prompts, params, lora_request=self._lora(use_adapter))
        return [o.outputs[0].text.strip() for o in outs]


BACKENDS = {"echo": EchoBackend, "ollama": OllamaBackend, "transformers": TransformersBackend,
            "vllm": VLLMBackend}


# --------------------------------------------------------------------------- the LLM handle


@dataclass
class CallLog:
    path: Path | None = None

    def record(self, entry: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


class LLM:
    """One resident model, addressed by role name."""

    def __init__(self, role: str, backend: str | None = None, cache_dir: str | Path | None = None,
                 call_log: str | Path | None = None):
        global _RESIDENT
        self.spec = role_spec(role)
        self.role = role
        self.key = residency_key(self.spec)
        if _RESIDENT is not None and _RESIDENT.key != self.key:
            raise ResidencyError(
                f"'{_RESIDENT.spec['model']}' is still resident; call .release() before loading "
                f"'{self.spec['model']}' (PLAN R2: one model at a time)"
            )
        self.backend_name = backend or load_config("models").get("backend", "echo")
        self.backend = BACKENDS[self.backend_name](self.spec)
        cache_root = cache_dir or load_config("models").get("cache_dir", "runs/_cache")
        self.cache_dir = Path(cache_root) if Path(cache_root).is_absolute() else ROOT / cache_root
        self.log = CallLog(Path(call_log) if call_log else None)
        _RESIDENT = self

    # -- lifecycle ---------------------------------------------------------
    def release(self) -> None:
        global _RESIDENT
        if hasattr(self.backend, "unload"):
            self.backend.unload()
        try:
            import torch

            for attr in ("model", "tuned"):          # tuned: a second, smaller base (adapter_base)
                if getattr(self.backend, attr, None) is not None:
                    delattr(self.backend, attr)
            torch.cuda.empty_cache()
        except Exception:
            pass
        if _RESIDENT is self:
            _RESIDENT = None

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()

    # -- calls -------------------------------------------------------------
    def _cache_path(self, system: str | None, messages: Messages, over: dict) -> Path:
        key = stable_hash(self.role, self.spec["model"], self.spec.get("adapter"), system, messages, over)
        return self.cache_dir / f"{key}.json"

    def chat(self, prompt: str | Messages, system: str | None = None, use_cache: bool = True,
             **over: Any) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        path = self._cache_path(system, messages, over)
        if use_cache and path.exists():
            return read_json(path)["text"]
        text = self.backend.generate(system, messages, **over)
        if use_cache:
            write_json(path, {"text": text})
        self.log.record({
            "role": self.role, "model": self.spec["model"], "backend": self.backend_name,
            "prompt_hash": stable_hash(system, messages)[:16],
            "prompt_chars": sum(len(m.get("content", "")) for m in messages),
            "output_chars": len(text), "overrides": over,
        })
        return text

    def view(self, role: str) -> "RoleView":
        return RoleView(self, role)

    def chat_batch(self, prompts: Sequence[str], system: str | None = None, **over: Any) -> list[str]:
        """Batched where the backend supports it, sequential otherwise. Cache is honoured either way."""
        pending: list[int] = []
        results: list[str | None] = [None] * len(prompts)
        for i, p in enumerate(prompts):
            msgs = [{"role": "user", "content": p}]
            path = self._cache_path(system, msgs, over)
            if path.exists():
                results[i] = read_json(path)["text"]
            else:
                pending.append(i)
        if pending and hasattr(self.backend, "generate_batch"):
            pairs = [(system, [{"role": "user", "content": prompts[i]}]) for i in pending]
            for i, text in zip(pending, self.backend.generate_batch(pairs, **over)):
                results[i] = text
                write_json(self._cache_path(system, [{"role": "user", "content": prompts[i]}], over),
                           {"text": text})
        else:
            for i in pending:
                results[i] = self.chat(prompts[i], system=system, **over)
        return [r or "" for r in results]

    def structured(self, prompt: str, required: Sequence[str] = (), system: str | None = None,
                   attempts: int = 3, **over: Any) -> dict:
        """Sample, parse, repair. Returns {'parse_failed': True, 'raw': ...} rather than raising.

        Failures are recorded, never dropped silently - the failure rate is a reported number (PLAN Sec. 5).
        """
        last_error = ""
        for attempt in range(attempts):
            text = self.chat(prompt if attempt == 0 else
                             f"{prompt}\n\nYour previous reply could not be parsed ({last_error}). "
                             f"Reply with valid JSON only, no prose, no code fences.",
                             system=system, use_cache=(attempt == 0), **over)
            parsed = extract_json(text)
            if parsed is None:
                last_error = "no JSON object found"
                continue
            missing = [k for k in required if k not in parsed]
            if missing:
                last_error = f"missing keys {missing}"
                continue
            parsed["parse_failed"] = False
            return parsed
        return {"parse_failed": True, "error": last_error, "raw": text}


class RoleView:
    """A second role served by an already-resident model.

    The four agent roles, the corpus generator and the user simulator all map to one base model, so they
    must not each load weights. A view reuses the parent LLM's backend and cache, swaps in that role's
    sampling parameters, and turns the supporter adapter off for roles that are not the supporter.
    """

    def __init__(self, parent: "LLM", role: str):
        self.parent = parent
        self.role = role
        self.spec = dict(role_spec(role))
        self.spec["model"] = parent.spec["model"]          # the resident weights, whatever config wished for
        self.backend_name = parent.backend_name
        self.use_adapter = bool(parent.spec.get("adapter")) and role == parent.role

    def _defaults(self, over: dict) -> dict:
        out = {"temperature": self.spec.get("temperature"), "top_p": self.spec.get("top_p"),
               "max_tokens": self.spec.get("max_tokens"), "use_adapter": self.use_adapter}
        out.update({k: v for k, v in over.items() if v is not None})
        return {k: v for k, v in out.items() if v is not None}

    def chat(self, prompt, system: str | None = None, use_cache: bool = True, **over: Any) -> str:
        return self.parent.chat(prompt, system=system, use_cache=use_cache, **self._defaults(over))

    def structured(self, prompt: str, required: Sequence[str] = (), system: str | None = None,
                   attempts: int = 3, **over: Any) -> dict:
        return self.parent.structured(prompt, required=required, system=system, attempts=attempts,
                                      **self._defaults(over))

    def release(self) -> None:
        """A view owns nothing; releasing it is a no-op so callers can treat it like an LLM."""
        return None


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply, tolerating fences and trailing prose."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def resident() -> "LLM | None":
    return _RESIDENT


def release_all() -> None:
    if _RESIDENT is not None:
        _RESIDENT.release()
