#!/bin/bash
# Memory comparison (10.10.10.104, RTX 3060, Ollama): need-state memory vs no memory, a rewritten summary and
# bag-of-words retrieval. Same untuned Qwen2.5-7B, four agents, gate off, 100 test people, 2-4 sessions each,
# profiles that advance between sessions and a corrective seeker; only the memory differs.
# Every step resumes where it stopped.   (setsid nohup ./scripts/memory_arms.sh > runs/memory_v2.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
M=runs/memory_v2
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }

for arm in mem_needstate mem_none mem_summary mem_dense; do
    echo "=== $arm $(date)"
    run python src/evaluate.py --arm $arm --run-dir $M --backend ollama
    run python src/metrics.py --dialogues $M/dialogues/$arm.jsonl --backend ollama --scale success --scale ip --scale pri
done
echo "=== memory comparison complete $(date)"
