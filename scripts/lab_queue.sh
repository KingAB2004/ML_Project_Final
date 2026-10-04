#!/bin/bash
# Lab server (RTX 3060): the work queued after the 7B fine-tune, one GPU job at a time.
#   1. the 7B's PVI and rank-truncation analyses (forward passes, ~1.5 h): the laptop has 8 GB, the lab 12 GB
#   2. scripts/scaling_eval.sh: test-profile evaluation of every fine-tuned size
# Waits for the training queue, never stops anything.
#   (setsid nohup ./scripts/lab_queue.sh > lab_queue.log 2>&1 &)
cd "$(dirname "$0")/.."
source ~/COCCON_NEW/coccon/bin/activate
TRAIN_QUEUE_PID=${TRAIN_QUEUE_PID:-1165336}
echo "queued $(date); waiting for the 7B training queue (pid $TRAIN_QUEUE_PID)"
while kill -0 "$TRAIN_QUEUE_PID" 2>/dev/null; do sleep 300; done
echo "=== 7B analyses start $(date)"
python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"
python extra/pvi.py --sizes 7B || echo "pvi 7B FAILED"
python extra/lora_geometry.py --sizes 7B --truncate || echo "lora_geometry 7B FAILED"
echo "=== 7B analyses done $(date)"
exec ./scripts/scaling_eval.sh
