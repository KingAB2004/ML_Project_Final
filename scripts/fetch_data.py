"""Download every external input and put it where the pipeline expects it.

  python scripts/fetch_data.py --all                 # datasets only
  python scripts/fetch_data.py --all --map           # datasets, then map the external profile sets
  python scripts/fetch_data.py --models              # model weights too (large; Mistral is gated)
  python scripts/fetch_data.py --ed --force          # re-fetch one item
  python scripts/fetch_data.py --self-check          # parsing checks, no network

What lands where:

  data/raw/empatheticdialogues.jsonl   one object per conversation: conv_id, situation, emotion
  data/raw/problem_types.json          ESConv problem-type taxonomy (from the data when present)
  data/raw/esconv.json                 ESConv, as downloaded
  data/raw/extes.json                  ExTES.json out of the repo's ExTES.zip
  data/raw/es_memeval.json             ES-MemEval's EvoEmo dataset (data/evo_emo.json)
  data/raw/_manifest.json              url, sha256, bytes, date for everything fetched
  data/raw/PROVENANCE.md               the same, as a table, with a licence column to fill in

Downloads use urllib from the standard library. `datasets` and `huggingface_hub` are used when installed and
skipped otherwise - every item has a plain-URL path, so this runs on a machine with only the standard
library plus pyyaml.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "_manifest.json"

USER_AGENT = "COCCON_NEW-fetch/1.0 (course project; contact via repository)"

SOURCES = {
    "ed": {
        "name": "EmpatheticDialogues",
        "hf": "facebook/empathetic_dialogues",
        "url": "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
        "page": "https://huggingface.co/datasets/facebook/empathetic_dialogues",
        "licence": "CC BY-NC 4.0",
        "target": RAW / "empatheticdialogues.jsonl",
    },
    "esconv": {
        "name": "ESConv",
        "hf": "thu-coai/esconv",
        "url": "https://huggingface.co/datasets/thu-coai/esconv/resolve/main/ESConv.json",
        "page": "https://huggingface.co/datasets/thu-coai/esconv",
        "licence": "see repository",
        "target": RAW / "esconv.json",
    },
    "extes": {
        "name": "ExTES",
        "url": "https://github.com/pandazzh2020/ExTES/raw/main/ExTES.zip",
        "page": "https://github.com/pandazzh2020/ExTES",
        "licence": "see repository",
        "target": RAW / "extes.json",
    },
    "memeval": {
        "name": "ES-MemEval / EvoEmo",
        "url": "https://raw.githubusercontent.com/slptongji/ES-MemEval/main/data/evo_emo.json",
        "url_alt": "https://raw.githubusercontent.com/slptongji/ES-MemEval/master/data/evo_emo.json",
        "page": "https://github.com/slptongji/ES-MemEval",
        "licence": "see repository",
        "target": RAW / "es_memeval.json",
    },
}

MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", "generator / simulator / agents (AWQ 4-bit)"),
    ("Qwen/Qwen2.5-7B-Instruct", "QLoRA base for the supporter"),
    ("mistralai/Mistral-Nemo-Instruct-2407", "judge - GATED: accept the licence and `hf auth login` first"),
    ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "judge fallback"),
    ("BAAI/bge-small-en-v1.5", "embeddings for the dense-retrieval memory baseline (optional)"),
]

# Fallback taxonomy, identical to the one embedded in src/seeds.py. Used only when the download yields no
# problem-type field of its own.
FALLBACK_PROBLEM_TYPES = [
    "ongoing depression", "job crisis", "problems with friends", "academic pressure", "breakup with partner",
    "conflicts with parents", "sleep problems", "appearance anxiety", "school bullying", "issues with children",
    "procrastination", "alcohol abuse", "issues with parents", "other",
]


# --------------------------------------------------------------------------- small helpers


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_bytes(url: str, alt: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        if alt:
            log(f"    {url} failed ({exc}); trying {alt}")
            return fetch_bytes(alt)
        raise


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def record(key: str, url: str, blob_hash: str, size: int, extra: dict | None = None) -> None:
    man = load_manifest()
    man[key] = {"name": SOURCES.get(key, {}).get("name", key), "url": url, "sha256": blob_hash,
                "bytes": size, "downloaded": date.today().isoformat(), **(extra or {})}
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")


def skip(target: Path, force: bool) -> bool:
    if target.exists() and not force:
        log(f"  {target.relative_to(ROOT)} already present - skipping (use --force to refetch)")
        return True
    return False


def unescape(text: str) -> str:
    """EmpatheticDialogues encodes commas as _comma_ in its CSV release."""
    return (text or "").replace("_comma_", ",").strip()


# --------------------------------------------------------------------------- per-source parsing


def ed_rows_from_csv(text: str) -> list[dict]:
    """One row per conversation: the situation is the `prompt` column, the emotion is `context`."""
    reader = csv.DictReader(io.StringIO(text))
    seen: set[str] = set()
    out: list[dict] = []
    for rec in reader:
        conv = (rec.get("conv_id") or "").strip()
        if not conv or conv in seen:
            continue
        situation = unescape(rec.get("prompt", ""))
        if not situation:
            continue
        seen.add(conv)
        out.append({"conv_id": conv, "situation": situation,
                    "emotion": (rec.get("context") or "unknown").strip()})
    return out


def ed_rows_from_hf(dataset) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for rec in dataset:
        conv = rec.get("conv_id")
        if not conv or conv in seen:
            continue
        seen.add(conv)
        out.append({"conv_id": conv, "situation": unescape(rec.get("prompt", "")),
                    "emotion": rec.get("context", "unknown")})
    return [r for r in out if r["situation"]]


def problem_types_from_esconv(payload) -> list[str]:
    """Read the taxonomy out of the data when the release carries it, else fall back to the fixed list."""
    rows = payload if isinstance(payload, list) else payload.get("data", payload.get("train", []))
    found: set[str] = set()
    for rec in rows if isinstance(rows, list) else []:
        for key in ("problem_type", "problem", "topic"):
            value = rec.get(key) if isinstance(rec, dict) else None
            if isinstance(value, str) and value.strip():
                found.add(value.strip().lower())
    types = sorted(found)
    if "other" not in types:
        types.append("other")
    return types if len(types) > 3 else list(FALLBACK_PROBLEM_TYPES)


def extes_json_from_zip(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".json") and not n.startswith("__MACOSX")]
        if not names:
            raise SystemExit(f"no .json inside the ExTES archive; members: {zf.namelist()[:10]}")
        pick = next((n for n in names if "extes" in n.lower()), max(names, key=lambda n: zf.getinfo(n).file_size))
        log(f"    using {pick} from the archive")
        return zf.read(pick)


# --------------------------------------------------------------------------- fetchers


def get_ed(force: bool) -> None:
    log("EmpatheticDialogues (profile seeds)")
    target = SOURCES["ed"]["target"]
    if skip(target, force):
        return
    rows: list[dict] = []
    url = SOURCES["ed"]["hf"]
    try:
        from datasets import load_dataset

        log("    loading through `datasets` (facebook/empathetic_dialogues, split=train)")
        rows = ed_rows_from_hf(load_dataset(SOURCES["ed"]["hf"], split="train"))
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the plain tarball
        log(f"    `datasets` route unavailable ({type(exc).__name__}); downloading the tarball")
        url = SOURCES["ed"]["url"]
        blob = fetch_bytes(url)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith("train.csv")), None)
            if member is None:
                raise SystemExit("train.csv not found inside the EmpatheticDialogues tarball") from exc
            handle = tf.extractfile(member)
            rows = ed_rows_from_csv(handle.read().decode("utf-8", errors="replace"))
    if not rows:
        raise SystemExit("EmpatheticDialogues yielded no rows - check the source and try again")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    blob = target.read_bytes()
    record("ed", url, sha256(blob), len(blob), {"conversations": len(rows)})
    log(f"    {len(rows)} conversations -> {target.relative_to(ROOT)}")


def get_esconv(force: bool) -> None:
    log("ESConv (problem-type taxonomy, reactive baseline)")
    target = SOURCES["esconv"]["target"]
    types_path = RAW / "problem_types.json"
    if skip(target, force) and types_path.exists():
        return
    url = SOURCES["esconv"]["url"]
    payload = None
    try:
        blob = fetch_bytes(url)
        payload = json.loads(blob.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log(f"    direct download failed ({type(exc).__name__}); trying `datasets`")
        try:
            from datasets import load_dataset

            ds = load_dataset(SOURCES["esconv"]["hf"], split="train")
            payload = [json.loads(r["text"]) if isinstance(r.get("text"), str) else dict(r) for r in ds]
            blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            url = SOURCES["esconv"]["hf"]
        except Exception as exc2:  # noqa: BLE001
            log(f"    ESConv unavailable ({type(exc2).__name__}). Writing the fallback taxonomy only.")
            types_path.write_text(json.dumps(FALLBACK_PROBLEM_TYPES, indent=2) + "\n", encoding="utf-8")
            return
    target.write_bytes(blob)
    types = problem_types_from_esconv(payload)
    types_path.write_text(json.dumps(types, indent=2) + "\n", encoding="utf-8")
    record("esconv", url, sha256(blob), len(blob), {"problem_types": len(types)})
    log(f"    {target.relative_to(ROOT)}; {len(types)} problem types -> {types_path.relative_to(ROOT)}")


def get_extes(force: bool) -> None:
    log("ExTES (second evaluation profile set)")
    target = SOURCES["extes"]["target"]
    if skip(target, force):
        return
    url = SOURCES["extes"]["url"]
    blob = fetch_bytes(url)
    data = extes_json_from_zip(blob) if blob[:2] == b"PK" else blob
    target.write_bytes(data)
    record("extes", url, sha256(data), len(data))
    log(f"    {len(data) // 1024} KiB -> {target.relative_to(ROOT)}")


def get_memeval(force: bool) -> None:
    log("ES-MemEval / EvoEmo (multi-session memory evaluation)")
    target = SOURCES["memeval"]["target"]
    if skip(target, force):
        return
    url = SOURCES["memeval"]["url"]
    blob = fetch_bytes(url, alt=SOURCES["memeval"].get("url_alt"))
    target.write_bytes(blob)
    record("memeval", url, sha256(blob), len(blob))
    log(f"    {len(blob) // 1024} KiB -> {target.relative_to(ROOT)}")


def get_ollama(force: bool) -> None:
    """Pull the tags named in configs/models.yaml through a local Ollama server."""
    import shutil as _shutil
    import subprocess

    log("Ollama models (local GGUF serving)")
    if _shutil.which("ollama") is None:
        log("    ollama is not installed. Linux/macOS: curl -fsSL https://ollama.com/install.sh | sh")
        log("    Windows / other: https://ollama.com/download . Then rerun with --ollama.")
        return
    sys.path.insert(0, str(ROOT / "src"))
    import yaml  # noqa: PLC0415

    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    tags = sorted({spec.get("ollama_tag") for spec in models["roles"].values() if spec.get("ollama_tag")})
    for tag in tags:
        log(f"    ollama pull {tag}")
        result = subprocess.run(["ollama", "pull", tag], capture_output=True, text=True)
        if result.returncode == 0:
            record(f"ollama:{tag}", f"https://ollama.com/library/{tag.split(':')[0]}", "-", 0,
                   {"name": f"ollama {tag}"})
        else:
            log(f"      failed: {(result.stderr or '').strip()[:200]}")
    log("    the fine-tuned supporter cannot be served this way until the adapter is merged and "
        "converted to GGUF; run supporter arms on backend: transformers until then")


def get_models(force: bool) -> None:
    log("Model weights (tens of GB; one at a time is fine, nothing is loaded here)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("    huggingface_hub is not installed: `pip install huggingface_hub`, then rerun with --models")
        return
    for repo, why in MODELS:
        log(f"    {repo}  [{why}]")
        try:
            path = snapshot_download(repo_id=repo, resume_download=True)
            record(f"model:{repo}", f"https://huggingface.co/{repo}", "-", 0, {"local_path": path})
            log(f"      -> {path}")
        except Exception as exc:  # noqa: BLE001 - a gated or missing repo must not stop the rest
            log(f"      skipped: {type(exc).__name__}: {exc}")
            if "gated" in str(exc).lower() or "401" in str(exc):
                log("      this repo is gated: accept the licence on the model page, then `hf auth login`")


def write_provenance() -> None:
    man = load_manifest()
    lines = ["# Dataset and model provenance", "",
             "Generated by `scripts/fetch_data.py`. Fill in the licence column from each source page before",
             "releasing anything built from these files.", "",
             "| Item | Source | Licence | sha256 (12) | Bytes | Downloaded |", "|---|---|---|---|---|---|"]
    for key, meta in sorted(man.items()):
        src = SOURCES.get(key, {})
        licence = src.get("licence", "see source")
        page = src.get("page", meta.get("url", ""))
        lines.append(f"| {meta.get('name', key)} | {page} | {licence} | "
                     f"`{str(meta.get('sha256', '-'))[:12]}` | {meta.get('bytes', 0)} | "
                     f"{meta.get('downloaded', '')} |")
    lines += ["", "## Expected local shapes", "",
              "- `empatheticdialogues.jsonl` - `{conv_id, situation, emotion}` per conversation. On Hugging",
              "  Face the situation is the `prompt` column and the emotion label is `context`; `_comma_`",
              "  escapes are already unescaped here.",
              "- `esconv.json` - as downloaded. `problem_types.json` is derived from it when it carries a",
              "  problem-type field, otherwise it is the fallback list embedded in `src/seeds.py`.",
              "- `extes.json` - `ExTES.json`, extracted from the repository's `ExTES.zip`.",
              "- `es_memeval.json` - the EvoEmo dataset (`data/evo_emo.json` in the ES-MemEval repository).",
              "", "## Models", "",
              "| Role | Repo | Note |", "|---|---|---|"]
    for repo, why in MODELS:
        lines.append(f"| {why} | https://huggingface.co/{repo} | {'gated' if 'mistralai' in repo else ''} |")
    (RAW / "PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"provenance written to {(RAW / 'PROVENANCE.md').relative_to(ROOT)}")


def map_external() -> None:
    """Run the schema mappers so data/external/*.jsonl exists for the evaluation arms."""
    sys.path.insert(0, str(ROOT / "src"))
    import external  # noqa: PLC0415 - imported late so this script works before src is on the path

    for which, raw in (("extes", SOURCES["extes"]["target"]), ("es_memeval", SOURCES["memeval"]["target"])):
        if not raw.exists():
            log(f"  {which}: {raw.name} missing, skipping the mapping")
            continue
        try:
            stats = external.run(which, raw)
            log(f"  {which}: {stats['profiles']} profiles -> {stats['path']} "
                f"({stats['success_rate_applicable']} with an annotated need)")
        except SystemExit as exc:
            log(f"  {which}: {exc}")


# --------------------------------------------------------------------------- self-check


def self_check() -> int:
    """Parsing checks on inline fixtures. No network, no downloads."""
    problems: list[str] = []

    csv_text = ("conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags\n"
                "hit:1_conv:1,1,lonely,I moved cities_comma_ and I eat alone,1,hello,5|5|5,\n"
                "hit:1_conv:1,2,lonely,I moved cities_comma_ and I eat alone,2,hi there,5|5|5,\n"
                "hit:2_conv:2,1,proud,My daughter graduated,3,great news,5|5|5,\n"
                "hit:3_conv:3,1,sad,,4,nothing,5|5|5,\n")
    rows = ed_rows_from_csv(csv_text)
    if len(rows) != 2:
        problems.append(f"ED csv: expected 2 conversations (one deduped, one empty dropped), got {len(rows)}")
    if rows and rows[0]["situation"] != "I moved cities, and I eat alone":
        problems.append(f"ED csv: _comma_ not unescaped: {rows[0]['situation']!r}")
    if rows and rows[0]["emotion"] != "lonely":
        problems.append("ED csv: emotion must come from the `context` column")

    hf_like = [{"conv_id": "c1", "prompt": "a_comma_ b", "context": "lonely"},
               {"conv_id": "c1", "prompt": "a_comma_ b", "context": "lonely"},
               {"conv_id": "c2", "prompt": "", "context": "sad"}]
    hf_rows = ed_rows_from_hf(hf_like)
    if len(hf_rows) != 1 or hf_rows[0]["situation"] != "a, b":
        problems.append(f"ED hf: expected one cleaned row, got {hf_rows}")

    typed = problem_types_from_esconv([{"problem_type": "Job Crisis"}, {"problem_type": "sleep problems"},
                                       {"problem_type": "academic pressure"}, {"problem_type": "job crisis"}])
    if typed != ["academic pressure", "job crisis", "sleep problems", "other"]:
        problems.append(f"ESConv taxonomy: unexpected extraction {typed}")
    if problem_types_from_esconv([{"x": 1}]) != FALLBACK_PROBLEM_TYPES:
        problems.append("ESConv taxonomy: a release without a problem-type field must use the fallback")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("junk/readme.txt", "hello")
        zf.writestr("ExTES/ExTES.json", '[{"scene": "x"}]')
    picked = extes_json_from_zip(buf.getvalue())
    if json.loads(picked)[0]["scene"] != "x":
        problems.append("ExTES zip: wrong member extracted")

    if unescape(" a_comma_b ") != "a,b":
        problems.append("unescape/strip is wrong")

    for problem in problems:
        log(f"FAIL {problem}")
    log(f"self-check: {len(problems)} problem(s) found" if problems
        else "self-check: all parsing checks pass (ED csv, ED hf rows, ESConv taxonomy, ExTES zip, unescape)")
    return 1 if problems else 0


# --------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and place every external input.")
    ap.add_argument("--all", action="store_true", help="all four datasets")
    ap.add_argument("--ed", action="store_true")
    ap.add_argument("--esconv", action="store_true")
    ap.add_argument("--extes", action="store_true")
    ap.add_argument("--memeval", action="store_true")
    ap.add_argument("--models", action="store_true", help="also download HF model weights (large)")
    ap.add_argument("--ollama", action="store_true",
                    help="pull the Ollama tags named in configs/models.yaml (for backend: ollama)")
    ap.add_argument("--map", action="store_true", help="run src/external.py on the downloaded sets")
    ap.add_argument("--force", action="store_true", help="refetch even if the target exists")
    ap.add_argument("--self-check", action="store_true", help="parsing checks only, no network")
    args = ap.parse_args()

    if args.self_check:
        return self_check()

    jobs = []
    if args.all or args.ed:
        jobs.append(get_ed)
    if args.all or args.esconv:
        jobs.append(get_esconv)
    if args.all or args.extes:
        jobs.append(get_extes)
    if args.all or args.memeval:
        jobs.append(get_memeval)
    if args.models:
        jobs.append(get_models)
    if args.ollama:
        jobs.append(get_ollama)
    if not jobs:
        ap.error("nothing selected: pass --all, --models, --ollama, or individual dataset flags "
                 "(--ed --esconv --extes --memeval)")

    RAW.mkdir(parents=True, exist_ok=True)
    failures = []
    for job in jobs:
        try:
            job(args.force)
        except SystemExit as exc:
            failures.append(f"{job.__name__}: {exc}")
            log(f"  FAILED {job.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - one dead source must not block the others
            failures.append(f"{job.__name__}: {type(exc).__name__}: {exc}")
            log(f"  FAILED {job.__name__}: {type(exc).__name__}: {exc}")

    write_provenance()
    if args.map:
        log("mapping external profile sets")
        map_external()

    if failures:
        log("\nsome sources failed:")
        for f in failures:
            log(f"  - {f}")
        log("Fetch the file by hand from its page, drop it at the path named in PROVENANCE.md, and rerun.")
        return 1
    log("\ndone. Next: python src/seeds.py --limit 100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
