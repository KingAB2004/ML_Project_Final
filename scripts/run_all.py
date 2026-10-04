"""Run the whole project end to end and leave a presentable result set behind.

Every stage runs as its own SUBPROCESS. That is not ceremony: only one model may be resident at a time
(PLAN R2), and the cheapest way to guarantee the generator's weights are gone before the judge loads is to
let the process exit. It also makes the pipeline resumable - a crashed stage is one command to repeat.

  python scripts/run_all.py --scale smoke              # minutes, echo backend, proves the wiring
  python scripts/run_all.py --scale pilot  --backend ollama
  python scripts/run_all.py --scale full   --backend ollama --train
  python scripts/run_all.py --scale full --only eval,score,report --run-dir runs/<id>
  python scripts/run_all.py --scale pilot --dry-run    # print the plan, run nothing

Stages, in order:

  check    the component checks (tests/test_all.py)
  corpus   seeds -> profiles(+splits) -> first sessions -> linked sessions -> integrity -> filter -> annotate
  sft      build the with_thoughts and wo_thoughts training files
  train    QLoRA fine-tuning of both arms                      (only with --train; needs a GPU)
  calib    ungated dialogues on held-out profiles -> conformal calibration, one per alpha in the arms
  eval     roll out every arm in configs/arms.yaml
  score    per arm: the instruments, IP, and the counterfactual PRI
  human    draw the blinded rating subsample (people fill it in, then rerun `report`)
  report   tables, 2x2 effects, paired bootstrap CIs, guards

What lands where:

  runs/<run_id>/dialogues/<arm>.jsonl     the dialogues of each arm
  runs/<run_id>/scores/<arm>/*.jsonl      every instrument, per arm
  runs/<run_id>/conformal/*.json          calibration, risk curve, alpha sweep
  runs/<run_id>/pipeline.log              one line per stage: command, exit code, wall clock
  runs/<run_id>/pipeline_state.json       which stages are done (this is what --resume reads)
  reports/results.md                      the presentable output
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PY = sys.executable

# scale -> (profiles built, test profiles per arm, calibration profiles, calibration turns)
# profiles=None builds corpus.n_profiles_target; arm_dialogues=None evaluates every test profile.
# 30 calibration profiles give ~90 sessions, enough for the 300 calibration turns.
SCALES = {
    "smoke": {"profiles": 12, "arm_dialogues": 2, "calib_profiles": 2, "calib_turns": 8, "backend": "echo"},
    "pilot": {"profiles": 100, "arm_dialogues": 20, "calib_profiles": 10, "calib_turns": 60, "backend": None},
    "full": {"profiles": None, "arm_dialogues": None, "calib_profiles": 30, "calib_turns": None, "backend": None},
}

STAGE_ORDER = ["check", "corpus", "sft", "train", "calib", "eval", "score", "human", "report"]
# Decomposed + ungated (cell C's distribution) on the CALIBRATION profiles: the gate is never fitted on the
# test profiles it is scored on.
CALIBRATION_ARM = "calib_dec_ungated"
# Default arm set: the baselines and the 2x2 (on the base model; Ollama cannot load an adapter), plus the
# two fine-tuned supporters when --train runs. `--arms all` runs every arm in configs/arms.yaml.
CORE_ARMS = ["base_instruct", "reactive_baseline", "cellA_mono_ungated", "cellB_mono_gated",
             "cellC_dec_ungated", "cellD_dec_gated"]
SFT_ARMS = {"sft_with_thoughts": "with_thoughts", "sft_wo_thoughts": "wo_thoughts"}
TRAINVAL = "data/profiles/profiles_trainval.jsonl"


@lru_cache(maxsize=1)
def load_arms() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "configs" / "arms.yaml").read_text(encoding="utf-8"))


def limit_args(n: int | None) -> list[str]:
    return ["--limit", str(n)] if n else []


def backend_args(backend: str | None) -> list[str]:
    return ["--backend", backend] if backend else []


def build_plan(scale: str, backend: str | None, run_dir: Path, arms: list[str], train: bool,
               only: list[str] | None) -> list[tuple[str, list[str], Path | None]]:
    """(stage, command, the artifact that proves it ran). Pure function, so it can be checked."""
    s = SCALES[scale]
    backend = backend or s["backend"]
    b = backend_args(backend)
    n_prof = s["profiles"]
    n_arm = s["arm_dialogues"]
    data = ROOT / "data"
    plan: list[tuple[str, list[str], Path | None]] = []

    plan.append(("check", [PY, str(ROOT / "tests" / "test_all.py")], None))

    plan += [
        # The original EmpatheticDialogues release; `datasets` 4.x no longer loads the Hub's script-based copy.
        ("corpus", [PY, str(ROOT / "scripts" / "fetch_data.py"), "--ed"],
         data / "raw" / "empatheticdialogues.jsonl"),
        ("corpus", [PY, str(SRC / "seeds.py"), *limit_args(n_prof), *b], data / "seeds" / "seeds.jsonl"),
        ("corpus", [PY, str(SRC / "profiles.py"), *limit_args(n_prof), "--split", *b],
         data / "profiles" / "profiles.jsonl"),
        # The corpus is generated for train + val profiles only; test and calibration never train.
        ("corpus", [PY, "-c", "import pathlib as p; d=p.Path('data/profiles'); "
                    f"p.Path('{TRAINVAL}').write_text(''.join((d / f'profiles_{{s}}.jsonl').read_text() "
                    "for s in ('train', 'val')))"], ROOT / TRAINVAL),
        ("corpus", [PY, str(SRC / "dialogue.py"), "--profiles", TRAINVAL, *b],
         data / "corpus" / "sessions.jsonl"),
        ("corpus", [PY, str(SRC / "sessions.py"), "--profiles", TRAINVAL, *b], None),
        ("corpus", [PY, str(SRC / "sessions.py"), "--check"], None),
        ("corpus", [PY, str(SRC / "filter.py"), *b], data / "corpus" / "sessions_filtered.jsonl"),
        ("corpus", [PY, str(SRC / "annotate.py"), *b], data / "corpus" / "sessions_annotated.jsonl"),
    ]

    plan += [
        ("sft", [PY, str(SRC / "build_sft.py"), "--arm", "with_thoughts"],
         data / "sft" / "with_thoughts_train.jsonl"),
        ("sft", [PY, str(SRC / "build_sft.py"), "--arm", "wo_thoughts"],
         data / "sft" / "wo_thoughts_train.jsonl"),
    ]

    if train:
        for arm in ("with_thoughts", "wo_thoughts"):
            plan.append(("train", [PY, str(SRC / "train.py"), "--arm", arm,
                                   "--out", str(run_dir)], run_dir / f"adapter_{arm}"))
        # The training-corpus comparison: flatten each external corpus, then fine-tune on it.
        for which, raw in (("esconv", data / "raw" / "esconv.json"),
                           ("extes", data / "raw" / "extes.json"),
                           ("smile", data / "raw" / "smile.json")):
            if not raw.exists():
                continue
            plan.append(("sft", [PY, str(SRC / "convert_corpus.py"), "--set", which, "--raw", str(raw)],
                         data / "sft" / f"corpus_{which}_train.jsonl"))
            plan.append(("train", [PY, str(SRC / "train.py"), "--arm", f"corpus_{which}",
                                   "--out", str(run_dir)], run_dir / f"adapter_corpus_{which}"))

    def eval_and_score(arm: str) -> None:
        cal = run_dir / "conformal" / calibration_name(arm_alpha(arm))
        extra = b
        if arm in SFT_ARMS:
            # The adapter this run trained. Ollama cannot load an adapter, so these arms run on
            # transformers (echo stays echo, so the smoke run exercises the same wiring).
            extra = ["--adapter", str(run_dir / f"adapter_{SFT_ARMS[arm]}"),
                     "--backend", "echo" if backend == "echo" else "transformers"]
        dial = run_dir / "dialogues" / f"{arm}.jsonl"
        plan.append(("eval", [PY, str(SRC / "evaluate.py"), "--arm", arm, "--run-dir", str(run_dir),
                              "--calibration", str(cal), *limit_args(n_arm), *extra], dial))
        plan.append(("score", [PY, str(SRC / "metrics.py"), "--dialogues", str(dial), *b],
                     run_dir / "scores" / arm / "success.jsonl"))
        plan.append(("score", [PY, str(SRC / "counterfactual.py"), "--dialogues", str(dial), *b],
                     run_dir / "scores" / arm / "pri_summary.json"))

    # Order: dataset -> fine-tuning -> test the fine-tuned supporters -> baselines. The fine-tuned arms are
    # ungated, so they need no calibration; the gate is fitted only before the baseline arms that use it.
    for arm in [a for a in arms if a in SFT_ARMS]:
        eval_and_score(arm)

    # Calibration: ungated dialogues first, then one calibration per alpha the arms ask for.
    plan.append(("calib", [PY, str(SRC / "evaluate.py"), "--arm", CALIBRATION_ARM,
                           "--run-dir", str(run_dir), *limit_args(s["calib_profiles"]), *b],
                 run_dir / "dialogues" / f"{CALIBRATION_ARM}.jsonl"))
    for alpha in alphas_in_use(arms):
        out = run_dir / "conformal" / calibration_name(alpha)
        cmd = [PY, str(SRC / "calibrate.py"), "--dialogues",
               str(run_dir / "dialogues" / f"{CALIBRATION_ARM}.jsonl"), "--out", str(out), *b]
        if alpha is not None:
            cmd += ["--alpha", str(alpha)]
        if s["calib_turns"]:
            cmd += ["--limit", str(s["calib_turns"])]
        plan.append(("calib", cmd, out))

    for arm in [a for a in arms if a not in SFT_ARMS]:
        eval_and_score(arm)

    plan.append(("human", [PY, str(ROOT / "human_eval" / "sample.py"), "--run", str(run_dir),
                           "--metric", "ip"], ROOT / "human_eval" / "forms" / "ip" / "sampled.jsonl"))
    plan.append(("report", [PY, str(SRC / "report.py"), "--run", str(run_dir)], None))

    if only:
        plan = [step for step in plan if step[0] in only]
    return plan


def arm_alpha(arm: str) -> float | None:
    spec = load_arms()["arms"].get(arm, {})
    return spec.get("gate_alpha")


def alphas_in_use(arms: list[str]) -> list[float | None]:
    found = {arm_alpha(a) for a in arms}
    found.add(None)                       # the config default, used by every arm that names no alpha
    return sorted(found, key=lambda x: (x is not None, x))


def calibration_name(alpha: float | None) -> str:
    return "calibration.json" if alpha is None else f"calibration_alpha_{alpha}.json"


def default_arms(train: bool = False) -> list[str]:
    return CORE_ARMS + (list(SFT_ARMS) if train else [])


# --------------------------------------------------------------------------- execution


def load_state(run_dir: Path) -> dict:
    path = run_dir / "pipeline_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(run_dir: Path, state: dict) -> None:
    (run_dir / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def log_line(run_dir: Path, text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "pipeline.log", "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} {text}\n")
    print(text, flush=True)


def step_key(cmd: list[str]) -> str:
    return " ".join(Path(c).name if c.endswith(".py") else c for c in cmd[1:])


def run_step(stage: str, cmd: list[str], artifact: Path | None, run_dir: Path, state: dict,
             force: bool, keep_going: bool) -> bool:
    key = step_key(cmd)
    if not force and state.get(key, {}).get("ok") and (artifact is None or artifact.exists()):
        log_line(run_dir, f"SKIP [{stage}] {key}")
        return True
    log_line(run_dir, f"RUN  [{stage}] {key}")
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    took = time.time() - started
    tail = (result.stdout or "").strip().splitlines()[-3:]
    for line in tail:
        log_line(run_dir, f"       {line}")
    state[key] = {"stage": stage, "ok": result.returncode == 0, "seconds": round(took, 1),
                  "finished": datetime.now(timezone.utc).isoformat()}
    save_state(run_dir, state)
    if result.returncode != 0:
        err = (result.stderr or "").strip().splitlines()[-8:]
        log_line(run_dir, f"FAIL [{stage}] exit {result.returncode} after {took:.1f}s")
        for line in err:
            log_line(run_dir, f"       {line}")
        return keep_going
    log_line(run_dir, f"OK   [{stage}] {took:.1f}s")
    return True


def summarize(run_dir: Path, state: dict) -> None:
    by_stage: dict[str, list[float]] = {}
    failed = []
    for key, meta in state.items():
        by_stage.setdefault(meta["stage"], []).append(meta.get("seconds", 0.0))
        if not meta.get("ok"):
            failed.append(key)
    log_line(run_dir, "")
    log_line(run_dir, "stage timings (seconds)")
    for stage in STAGE_ORDER:
        if stage in by_stage:
            log_line(run_dir, f"  {stage:8s} {sum(by_stage[stage]):8.1f}  ({len(by_stage[stage])} steps)")
    if failed:
        log_line(run_dir, f"\n{len(failed)} step(s) failed:")
        for key in failed:
            log_line(run_dir, f"  - {key}")
    log_line(run_dir, f"\nresults: {(ROOT / 'reports' / 'results.md')}")
    log_line(run_dir, f"run dir: {run_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the whole pipeline and produce the result set.")
    ap.add_argument("--scale", choices=sorted(SCALES), default="pilot")
    ap.add_argument("--backend", default=None, help="echo | ollama | transformers | vllm")
    ap.add_argument("--run-dir", default=None, help="reuse an existing run directory (resumes it)")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm names, or 'all' (default: baselines + 2x2, + SFT arms with --train)")
    ap.add_argument("--only", default=None,
                    help=f"comma-separated stages to run: {','.join(STAGE_ORDER)}")
    ap.add_argument("--train", action="store_true", help="include QLoRA fine-tuning (needs a GPU)")
    ap.add_argument("--force", action="store_true", help="rerun steps already marked done")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing step")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    arms = (list(load_arms()["arms"]) if args.arms == "all" else
            args.arms.split(",") if args.arms else default_arms(args.train))
    only = args.only.split(",") if args.only else None
    if only:
        unknown = [s for s in only if s not in STAGE_ORDER]
        if unknown:
            ap.error(f"unknown stage(s) {unknown}; choose from {STAGE_ORDER}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        sys.path.insert(0, str(SRC))
        from common import new_run_dir

        run_dir = new_run_dir(f"pipeline_{args.scale}")

    plan = build_plan(args.scale, args.backend, run_dir, arms, args.train, only)

    if args.dry_run:
        print(f"run dir: {run_dir}\n{len(plan)} steps, {len(arms)} arms, scale={args.scale}\n")
        for stage, cmd, artifact in plan:
            print(f"[{stage:7s}] {step_key(cmd)}")
        return 0

    state = load_state(run_dir)
    log_line(run_dir, f"=== pipeline start: scale={args.scale} backend={args.backend or 'config'} "
                      f"arms={len(arms)} steps={len(plan)}")
    for stage, cmd, artifact in plan:
        if not run_step(stage, cmd, artifact, run_dir, state, args.force, args.keep_going):
            log_line(run_dir, "stopping (pass --keep-going to push past a failing step)")
            summarize(run_dir, state)
            return 1
    summarize(run_dir, state)
    return 0 if all(m.get("ok") for m in state.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
