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

## Design rules that the code enforces, not just documents

1. **Every stage reads files and writes files.** A stage is one script; rerunning it costs only that stage.
2. **One model resident at a time.** `llm.LLM` raises `ResidencyError` on a second distinct model. Several
   conversational roles share one resident model through `LLM.view(role)`.
3. **Structured output augments, never replaces, raw history.** Every agent and judge prompt contains the
   verbatim turns.
4. **Nothing enters belief state ungrounded.** `grounding.validate_span` gates every memory write.
5. **Resumable.** Stages skip ids already present in their output file.
6. **No proprietary API anywhere.** There is no code path that can call one.

> Scope: `modified_SOP_12340340_12340370.pdf`, which has **four** enhancements. Enhancement 5 (adaptive
> disclosure pacing) and its readiness value, privileged teacher and distillation stage have been removed;
> see `CHANGES.md`. The disclosure ladder itself remains, because Enhancement 2 caps depth with it.

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

## Order of work

```
python tests/test_all.py                                     # component checks, echo backend
python scripts/fetch_data.py --self-check                    # download-parsing checks, no network
python scripts/fetch_data.py --all --map                     # datasets into data/raw + data/external
python scripts/fetch_data.py --models                        # weights (large; Mistral-Nemo is gated)

python src/seeds.py     --limit 100                          # pilot first: gate G1 decides the route
python src/profiles.py   --limit 100 --split
python src/dialogue.py  --limit 100
python src/sessions.py  --limit 100
python src/sessions.py  --check                              # temporal integrity
python src/filter.py
python src/annotate.py

python src/build_sft.py --arm with_thoughts
python src/build_sft.py --arm wo_thoughts
python src/train.py     --arm with_thoughts                  # needs a GPU

python src/evaluate.py  --arm cellC_dec_ungated --limit 30    # ungated dialogues for calibration
python src/calibrate.py --dialogues runs/<id>/dialogues/cellC_dec_ungated.jsonl
python src/evaluate.py  --arm cellD_dec_gated --run-dir runs/<id>

python src/metrics.py        --dialogues runs/<id>/dialogues/cellD_dec_gated.jsonl
python src/counterfactual.py --dialogues runs/<id>/dialogues/cellD_dec_gated.jsonl

python human_eval/sample.py    --run runs/<id> --metric ip    # then two people fill the sheets
python human_eval/agreement.py --run runs/<id> --metric ip
python src/report.py --run runs/<id>
```

Scoring is always a separate pass from generation: the judge is a different model and must not be co-resident
with the generator.

## What the report refuses to print

`src/report.py` fails rather than footnoting when a cell has no sample size, when the judge model equals the
supporter's or the simulator's, when an out-of-distribution result is described as a guarantee, and it marks
any IP/PRI headline that has no judge-human agreement figure yet.

## Comparing against other corpora (the baseline paper's Tables 3 and 4)

Their comparison fine-tunes one base model on each corpus and scores every resulting supporter on the same
profiles. That is reproduced here as arms, not as a bespoke script:

```
python src/convert_corpus.py --set esconv --raw data/raw/esconv.json   # -> data/sft/corpus_esconv_*.jsonl
python src/train.py --arm corpus_esconv --out runs/<id>                # one adapter per corpus
# put the adapter paths into configs/arms.yaml -> adapters:
python src/evaluate.py --arm corpus_esconv --arm corpus_ours --run-dir runs/<id>
```

`--train` in `scripts/run_all.py` does the convert-and-train steps for every corpus whose raw file is
present. The report then prints a **Training-corpus comparison** table with the same columns as the paper:
SR, Basic Avg (the six basic metrics rescaled to 0-100), and the four scale dimensions Aff / Neg / Sup / Man,
for our profiles and for the ExTES profiles.

Two honest asymmetries are printed with it: corpora other than ours carry no Analysis/Strategy annotation,
so their adapters train response-only (the same condition as our `w/o thoughts` arm); and the item groupings
behind Aff/Neg/Sup/Man are ours, listed in `src/metrics.py`, because the published instruments define items,
not groupings.

## Corpus size

COCOON reports **3.5k dialogues at 18.82 average turns** (their Table 1, Chinese, single-session). Nothing in
this code caps corpus size - the only limits are config values, and every stage resumes, so raising them and
rerunning costs only the new records:

| Knob | Default | Effect |
|---|---|---|
| `corpus.n_profiles_target` | 1000 | profiles attempted; ~1000 x 3 sessions x ~0.65 filter survival ~= 2k sessions (sized to one 12 GB GPU; 2000 gives ~3.9k) |
| `corpus.sessions_per_profile` | [2, 4] | sessions per linked sequence |
| `corpus.turns_per_session` | [8, 12] | supporter turns, so 16-24 total turns per session |
| `eval.n_dialogues_per_arm` | 150 | evaluation only; does not touch corpus size |

Seed supply is not the constraint either: EmpatheticDialogues has ~19.5k unique situations, and `seeds.py`
pulls `2 x target` candidates before screening. The real constraint is generation time - roughly 3-5k output
tokens per session, so the default ~3k generated sessions is ~10-15M output tokens plus judging.

## Reused from the upstream COCOON repository

`reference_cocoon/` holds the five upstream files that are worth keeping as reference: the Chinese proactive
speaker prompt (the behavioural spec for our English simulator), the English reactive speaker prompt (our
reactive baseline), and the AELS / Comforting-Responses / RAC scale prompts, which are cross-checked against
our own implementations in `prompts/`. No upstream Python is used: it assumes GPT-4o, hardcodes internal
cluster URLs and author-absolute paths, and is Chinese-first.
