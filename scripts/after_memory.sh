#!/bin/bash
# 10.10.10.104, after scripts/memory_arms.sh: memory metrics on all four arms, then ESC-RANK (paper Table 6)
# on the data-quality samples and the data-quality table again with it.
#   (setsid nohup ./scripts/after_memory.sh > runs/after_memory.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
M=runs/memory_v2
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "queued $(date); waiting for the memory arms"
until grep -q "memory comparison complete" runs/memory_v2.log 2>/dev/null; do sleep 300; done

echo "=== memory metrics, all arms $(date)"
run python extra/memory_eval.py --run $M --judge --backend ollama
run python extra/memory_calibration.py --run $M --backend ollama

echo "=== ESC-RANK $(date)"
until grep -q "escrank setup done" runs/escrank_setup.log 2>/dev/null; do sleep 60; done
free_gpu
for set in ours extes esconv; do
    ESCRANK_ENABLE=1 run ~/escrank_venv/bin/python scripts/escrank_score.py --set $set
done
run python extra/data_quality.py table

echo "=== after-memory queue complete $(date)"
