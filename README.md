# COCCON_NEW

Implementation of `SOP_12340340_12340370.pdf` / `PLAN.md`: a proactive emotional-support system built on
open-weight models, with five enhancements over the COCOON framework.

| Enhancement | What it is | Where it lives |
|---|---|---|
| E1 | English proactive corpus with multi-session temporal structure | `src/seeds.py`, `src/profiles.py`, `src/dialogue.py`, `src/sessions.py`, `src/filter.py`, `src/annotate.py` |
| E2 | Multi-agent pipeline, typed grounding contract, conformal critic gate | `src/agents.py`, `src/grounding.py`, `src/conformal.py`, `src/calibrate.py` |
| E3 | Turn-level Intrusiveness Penalty and Psychological Reactance Index | `src/metrics.py`, `src/counterfactual.py`, `human_eval/` |
| E4 | Need-state trajectory memory (belief, not facts) | `src/memory.py`, `src/baselines/memory_*.py` |
| Fine-tuning | QLoRA SFT with Analysis/Strategy in the target, plus the `w/o thoughts` arm | `src/build_sft.py`, `src/train.py` |
| Depth ladder | L0-L3 cap used by the Strategist and the `overreaching_depth` check (part of E2, not a pacing policy) | `src/pacing.py` |
| Evaluation | Arm runner, instruments, statistics, guarded reporting | `src/evaluate.py`, `src/judge.py`, `src/report.py` |


## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install pyyaml numpy                 # enough for the checks and all file-level logic
pip install -r requirements.txt          # add this when you want real inference or training
```

`configs/models.yaml` ships with `backend: echo` - deterministic stubs, no GPU. Switch to `ollama`,
`transformers` or `vllm` when you have the hardware; nothing else changes.

**Ollama** (no proprietary API anywhere in this project): install with `curl -fsSL https://ollama.com/install.sh | sh`
(or https://ollama.com/download), then `python scripts/fetch_data.py --ollama` to pull the tags listed in
`configs/models.yaml` - `qwen2.5:7b-instruct` for the generator/simulator/agents and
`mistral-nemo:12b-instruct-2407-q4_K_M` for the judge. Set `backend: ollama` and the project talks to
`http://localhost:11434` over plain HTTP (no Python client needed); `release()` sends `keep_alive: 0`, so the
one-model-at-a-time rule still holds on a 12 GB card. Ollama serves GGUF and cannot train: `src/train.py`
always runs on the transformers backend, and serving the fine-tuned supporter through Ollama means merging
the adapter, converting to GGUF with llama.cpp, and `ollama create`. Until then run supporter arms on
`backend: transformers`.

## One command for the whole thing

On a fresh GPU machine, `scripts/full_run.sh` does the setup (venv, Ollama + models, the fine-tuning base
weights, configs) and then runs every phase below through `run_all.py`:

```
./scripts/full_run.sh smoke                                # ~10 min wiring check in a throwaway copy; run it first
N_PROFILES=1000 WORKERS=8 ./scripts/full_run.sh dataset    # stop once the dataset is built
N_PROFILES=1000 WORKERS=8 ./scripts/full_run.sh            # everything; rerun the same command to resume
```

Order: dataset (seeds, profiles, sessions, filter, annotate, SFT files) -> fine-tuning (with/wo thoughts) ->
test the fine-tuned supporters -> gate calibration + baselines -> report. The dataset lands in `data/corpus/`
(annotated multi-session dialogues) and `data/sft/` (training/validation files).

By default the runner evaluates the baselines and the 2x2 (`base_instruct`, `reactive_baseline`,
`cellA`-`cellD`, on the base model through Ollama) and, with `--train`, the two fine-tuned supporters on
the adapters it just trained (transformers backend). `--arms all` runs every arm in `configs/arms.yaml`.
The gate is calibrated on the calibration profiles (`calib_dec_ungated`), never on the test profiles.
The `scripts/phase*50.sh` / `run_v3.sh` files are the lab-server (RTX 3060) 50-profile run, not for new machines.

```
python scripts/run_all.py --scale smoke                       # minutes, echo backend, proves the wiring
python scripts/run_all.py --scale pilot --backend ollama      # 100 profiles, 20 dialogues per arm
python scripts/run_all.py --scale full  --backend ollama --train
python scripts/run_all.py --scale full --dry-run              # print the 60-odd steps, run nothing
python scripts/run_all.py --only eval,score,report --run-dir runs/<id>
```

Each stage runs as its own subprocess, so the generator's weights are gone before the judge loads, and a
crashed stage is one command to repeat. `runs/<id>/pipeline_state.json` records what finished, so rerunning
resumes; `--force` overrides, `--keep-going` pushes past a failure. `runs/<id>/pipeline.log` holds per-stage
wall clock, and the run ends with a timing summary.

The ablations are arms, not bespoke scripts, so the runner covers them by iterating `configs/arms.yaml`:
the annotation ablation (`sft_with_thoughts` vs `sft_wo_thoughts`), the 2x2 of decomposition x gate
(`cellA`-`cellD`), the alpha sweep (a separate calibration is fitted per alpha), and the five memory arms
on the internal and both external sets. `src/report.py` then writes `reports/results.md` with the per-arm
table, the 2x2 main effects and interaction, the risk-coverage frontier, and paired-bootstrap CIs.




