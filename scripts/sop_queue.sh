#!/bin/bash
# Lab server (RTX 3060): the remaining SOP experiments, run after scripts/lab_queue.sh (7B analyses + scaling
# evaluation) has finished. One GPU job at a time, all on Ollama. Every step resumes where it stopped; a failing
# step is logged and the queue moves on.
#   (items 1, 4 and 5 - baselines + 2 x 2, alpha sweep, ExTES - moved to the second RTX 3060:
#    scripts/base_queue.sh; item 4 and 5 read item 1's calibration, so the three stay together)
#   2. memory arms: none / summary / dense / event / needstate + memory metrics          (Outcome 4)
#   3. ES-MemEval question answering per memory type                                    (Outcome 4)
#   (setsid nohup ./scripts/sop_queue.sh > sop_queue.log 2>&1 &)
cd "$(dirname "$0")/.."
source ~/COCCON_NEW/coccon/bin/activate
PREV_PID=${PREV_PID:-1352640}        # lab_queue.sh, which becomes scaling_eval.sh (exec keeps the pid)
echo "queued $(date); waiting for the current lab queue (pid $PREV_PID)"
while kill -0 "$PREV_PID" 2>/dev/null; do sleep 300; done

M=runs/memory_1000
B=(--backend ollama)
run() { echo "--- $* ($(date))"; "$@" || echo "!!! FAILED: $* ($(date))"; }

echo "=== 2. memory arms $(date)"
for arm in mem_none mem_summary mem_dense mem_event mem_needstate; do
    run python src/evaluate.py --arm $arm --run-dir $M "${B[@]}"
    run python src/metrics.py --dialogues $M/dialogues/$arm.jsonl "${B[@]}" --scale success --scale ip --scale pri
done
run python extra/memory_eval.py --run $M --judge "${B[@]}"
run python extra/memory_calibration.py --run $M "${B[@]}"

echo "=== 3. ES-MemEval $(date)"
run python extra/memeval_qa.py --run runs/memeval "${B[@]}"

echo "=== SOP queue complete $(date)"
