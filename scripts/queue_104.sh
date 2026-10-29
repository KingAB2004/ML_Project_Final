#!/bin/bash
# 10.10.10.104 queue after 10.10.2.41 was lost (Oct 16): transformers stack for the adapter runs, then (after the
# ESC-RANK re-run) the ExTES corpus comparison and cellD_extes (our full pipeline on ExTES people, Ollama).
#   (setsid nohup ./scripts/queue_104.sh > runs/queue_104.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }

echo "=== transformers stack $(date)"
for i in 1 2 3 4 5; do
    python -c "import torch, transformers, peft, bitsandbytes, datasets; assert torch.cuda.is_available()" 2>/dev/null && break
    pip install -q torch --index-url https://download.pytorch.org/whl/cu128
    pip install -q "transformers>=4.44,<5" "huggingface_hub>=0.34,<1.0" "datasets>=3,<4.3" accelerate peft bitsandbytes sentencepiece
done
python -c "import torch, transformers, peft, datasets, huggingface_hub; print('env', torch.__version__, transformers.__version__, peft.__version__, datasets.__version__, huggingface_hub.__version__, torch.cuda.is_available())" \
    || { echo "!!! environment not usable $(date)"; exit 1; }

echo "waiting for the ESC-RANK re-run $(date)"
until grep -q "escrank fix done" runs/escrank_fix.log 2>/dev/null; do sleep 120; done

echo "=== ExTES comparison $(date)"
NO_WAIT=1 ./scripts/extes.sh

echo "=== cellD_extes $(date)"
R=runs/baselines_1000
run python src/evaluate.py --arm cellD_extes --run-dir $R --calibration $R/conformal/calibration.json --limit 100 --backend ollama
run python src/metrics.py --dialogues $R/dialogues/cellD_extes.jsonl --backend ollama --scale success --scale ip --scale pri --scale aels
echo "=== queue_104 complete $(date)"
