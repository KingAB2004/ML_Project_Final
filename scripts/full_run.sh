#!/bin/bash
# Full COCCON_NEW run on a fresh GPU machine: setup, then every phase in order (scripts/run_all.py):
#   1. dataset   : check -> seeds -> profiles+splits -> sessions -> filter -> annotate -> SFT files
#   2. fine-tune : QLoRA with_thoughts, then wo_thoughts
#   3. test      : roll out + score the two fine-tuned supporters on the test profiles
#   4. baselines : gate calibration -> roll out + score base_instruct, reactive_baseline, cellA-cellD
#   5. report    : human-rating sample + reports/results.md
#
#   ./scripts/full_run.sh smoke                              # ~10 min wiring check in a throwaway copy (stub
#                                                            #   model, real QLoRA on a few examples) - first
#   N_PROFILES=1000 WORKERS=8 ./scripts/full_run.sh dataset  # stop after step 1 (the dataset)
#   N_PROFILES=1000 WORKERS=8 ./scripts/full_run.sh          # everything; resumes, so after `dataset` it
#                                                            #   skips straight to fine-tuning
#
# Needs: Linux, an NVIDIA GPU with drivers, python3 (3.10+) with venv, internet; sudo only to install Ollama
# and set its parallelism. Results: runs/full_<N>/results via reports/results.md, log in runs/full_<N>/pipeline.log
set -euo pipefail
cd "$(dirname "$0")/.."
N=${N_PROFILES:-1000}
WORKERS=${WORKERS:-8}        # concurrent Ollama requests; 8 suits a 24 GB+ card, 4 a 12 GB one
RUN=${RUN_DIR:-runs/full_$N}

# --- 1. Python environment (vllm and autoawq are optional and unused: skipped) -----------------------------
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
grep -vE '^(vllm|autoawq)' requirements.txt > .venv/requirements_fullrun.txt
pip install -q -r .venv/requirements_fullrun.txt

# --- 1b. GPU check, before any long step: a torch or bitsandbytes build without kernels for this GPU would
#         otherwise surface only at fine-tuning, hours into the run. RTX 50xx (Blackwell, sm_120) needs
#         torch >= 2.7 built for CUDA >= 12.8 and an NVIDIA driver >= 570.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
gpu_ok() {
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch sees no GPU (driver too old for this torch build?)"
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()                                   # fails with "no kernel image" on an unsupported build
import bitsandbytes as bnb
lin = bnb.nn.Linear4bit(512, 512, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
lin(x).sum().item()                                    # the 4-bit kernel QLoRA training uses
cap = torch.cuda.get_device_capability()
print(f"GPU ok: {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}, torch {torch.__version__} "
      f"(CUDA {torch.version.cuda}), bitsandbytes {bnb.__version__}")
PY
}
if ! gpu_ok; then
  # The default PyPI wheel can target a newer CUDA than this driver supports: try the CUDA 12.8 build.
  echo "GPU check failed with the default torch wheel - retrying with the CUDA 12.8 build"
  pip install -q --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
  gpu_ok || { echo "GPU check failed: update the NVIDIA driver (>= 570 for RTX 50xx) and rerun"; exit 1; }
fi

if [ "${1:-}" = "smoke" ]; then
  # A copy, so the stub data never lands in this checkout's data/ (the real run refuses to start on it).
  S=$(mktemp -d /tmp/coccon_smoke.XXXX)
  tar --exclude=./.venv --exclude=./runs --exclude=./results --exclude='*/__pycache__' -cf - . | tar -xf - -C "$S"
  cd "$S"
  "${OLDPWD}/.venv/bin/python" scripts/run_all.py --scale smoke --train --run-dir runs/smoke
  echo "smoke OK - copy at $S (delete it when done)"
  exit 0
fi

# --- 2. Ollama: serves the supporter/simulator/agents and the judge ---------------------------------------
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
if command -v systemctl >/dev/null && [ -d /etc/systemd/system ]; then
  # sudo only when the override is missing or differs, so a rerun (or a non-interactive session) needs no password
  CONF=$(printf '[Service]\nEnvironment="OLLAMA_NUM_PARALLEL=%s"\nEnvironment="OLLAMA_FLASH_ATTENTION=1"\nEnvironment="OLLAMA_KV_CACHE_TYPE=q8_0"\n' "$WORKERS")
  if [ "$(cat /etc/systemd/system/ollama.service.d/coccon.conf 2>/dev/null)" != "$CONF" ]; then
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    echo "$CONF" | sudo tee /etc/systemd/system/ollama.service.d/coccon.conf >/dev/null
    sudo systemctl daemon-reload && sudo systemctl restart ollama
  fi
else
  # no systemd (e.g. a container): run the server ourselves
  pgrep -x ollama >/dev/null || { OLLAMA_NUM_PARALLEL=$WORKERS OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
                                  nohup ollama serve > ollama.log 2>&1 & }
fi
until curl -sf http://localhost:11434/api/tags >/dev/null; do sleep 2; done
ollama pull qwen2.5:7b-instruct
ollama pull mistral-nemo:12b-instruct-2407-q4_K_M

# --- 3. Base model weights for fine-tuning (the Hub connection can be flaky: retry) ------------------------
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
# `hf` is the CLI since huggingface_hub 0.34; `huggingface-cli` no longer works in 1.x (only an older hub has it)
HF=$(command -v hf || command -v huggingface-cli)
for i in $(seq 1 30); do "$HF" download Qwen/Qwen2.5-7B-Instruct >/dev/null && break; echo "download retry $i"; sleep 30; done
python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"

# --- 4. Configuration for this run (idempotent edits) -----------------------------------------------------
sed -i 's|^backend: echo |backend: ollama|' configs/models.yaml
sed -i "s|^  workers: [0-9]*|  workers: $WORKERS|; s|^  n_profiles_target: [0-9]*|  n_profiles_target: $N|" configs/default.yaml
# QLoRA micro-batch by VRAM, effective batch 16 either way: 4 x 4 needs ~16 GB, 1 x 16 fits a 12 GB card.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -ge 20000 ]; then MB=4; GA=4; else MB=1; GA=16; fi
sed -i "s|^  micro_batch_size: [0-9]*|  micro_batch_size: $MB|; s|^  grad_accum: [0-9]*|  grad_accum: $GA|" configs/default.yaml
grep -q '^backend: ollama' configs/models.yaml || { echo "configs/models.yaml backend is not ollama"; exit 1; }

# --- 5. Fresh data only: stages resume by id, so another run's corpus would be silently reused ------------
if [ ! -f "$RUN/pipeline_state.json" ] && { [ -s data/corpus/sessions.jsonl ] || [ -s data/profiles/profiles.jsonl ]; }; then
  echo "data/corpus or data/profiles holds another run's data. Move data/{seeds,profiles,corpus,sft} and"
  echo "runs/_cache aside (or start from a clean copy) before a new full run."
  exit 1
fi

# --- 6. Every phase, one at a time, resumable -------------------------------------------------------------
if [ "${1:-}" = "dataset" ]; then
  python scripts/run_all.py --scale full --backend ollama --train --run-dir "$RUN" --only check,corpus,sft
  echo "dataset ready: data/sft/{with,wo}_thoughts_{train,val}.jsonl and data/corpus/sessions_annotated.jsonl"
else
  python scripts/run_all.py --scale full --backend ollama --train --run-dir "$RUN"
fi
