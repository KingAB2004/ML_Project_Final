#!/bin/bash
# Lab server (RTX 3060): the remaining SOP experiments, started by scripts/lab_chain.sh after the 7B PVI.
# One GPU job at a time, all on Ollama. Every step resumes where it stopped; a failing step is logged and the
# queue moves on. Order (6 Oct): our memory first, then the baselines + 2 x 2 that were on the second RTX 3060
# (that machine died), then the other memories.
#   2a. mem_needstate (our memory) + Success / IP / PRI                                  (Outcome 4)
#   1.  baselines + 2 x 2: gate calibration, base_instruct, reactive_baseline, cellA-D    (Outcomes 1-2)
#   2b. memory baselines: none / summary / dense / event, then the memory metrics         (Outcome 4)
#   3.  ES-MemEval question answering per memory type                                    (Outcome 4)
#   4.  alpha sweep: the gate at alpha 0.05 and 0.20 on item 1's calibration (0.10 = cellD) (Outcome 3)
#   (ExTES of scripts/base_queue.sh is not scheduled)
#   (setsid nohup ./scripts/sop_queue.sh > sop_queue.log 2>&1 &)
cd "$(dirname "$0")/.."
source ~/COCCON_NEW/coccon/bin/activate
PREV_PID=${PREV_PID:-1352640}        # lab_chain.sh passes a pid that does not exist: start at once
echo "queued $(date); waiting for the current lab queue (pid $PREV_PID)"
while kill -0 "$PREV_PID" 2>/dev/null; do sleep 300; done

M=runs/memory_1000
B=(--backend ollama)
run() { echo "--- $* ($(date))"; "$@" || echo "!!! FAILED: $* ($(date))"; }
mem_arm() {
    run python src/evaluate.py --arm "$1" --run-dir $M "${B[@]}"
    run python src/metrics.py --dialogues $M/dialogues/$1.jsonl "${B[@]}" --scale success --scale ip --scale pri
}

echo "=== 2a. mem_needstate (ours) $(date)"
mem_arm mem_needstate
echo "=== 2a. mem_needstate done $(date)"

# A run directory of its own: runs/full_1000 holds the trained adapters and the dataset pipeline state.
# Without --train, run_all evaluates only the six base-model arms (configs/arms.yaml maps no adapter).
echo "=== 1. baselines + 2x2 $(date)"
run python scripts/run_all.py --scale full "${B[@]}" --run-dir runs/baselines_1000 --only calib,eval,score,report \
    --keep-going
echo "=== 1. baselines + 2x2 done $(date)"

echo "=== 2b. memory baselines $(date)"
for arm in mem_none mem_summary mem_dense mem_event; do
    mem_arm $arm
done
run python extra/memory_eval.py --run $M --judge "${B[@]}"
run python extra/memory_calibration.py --run $M "${B[@]}"

echo "=== 3. ES-MemEval $(date)"
run python extra/memeval_qa.py --run runs/memeval "${B[@]}"

echo "=== 4. alpha sweep $(date)"
R=runs/baselines_1000
for a in 0.05 0.2; do
    run python src/calibrate.py --dialogues $R/dialogues/calib_dec_ungated.jsonl \
        --out $R/conformal/calibration_alpha_$a.json --alpha $a "${B[@]}"
done
for pair in "gate_alpha_005 0.05" "gate_alpha_020 0.2"; do
    set -- $pair
    run python src/evaluate.py --arm $1 --run-dir $R "${B[@]}" --calibration $R/conformal/calibration_alpha_$2.json
    run python src/metrics.py --dialogues $R/dialogues/$1.jsonl "${B[@]}" --scale success --scale ip --scale pri
done
run python src/report.py --run $R

echo "=== SOP queue complete $(date)"
