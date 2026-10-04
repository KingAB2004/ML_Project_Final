#!/bin/bash
# Lab server (RTX 3060): the remaining SOP experiments, run after scripts/lab_queue.sh (7B analyses + scaling
# evaluation) has finished. One GPU job at a time, all on Ollama. Every step resumes where it stopped; a failing
# step is logged and the queue moves on.
#   1. full-size baselines + 2 x 2: gate calibration, base_instruct, reactive_baseline, cells A-D (Outcomes 1-2)
#   2. memory arms: none / summary / dense / event / needstate + memory metrics          (Outcome 4)
#   3. ES-MemEval question answering per memory type                                    (Outcome 4)
#   4. alpha sweep: gate at alpha 0.05 and 0.20 (0.10 is cell D)                         (Outcome 3)
#   5. the gated system on ExTES profiles: violation rate under distribution shift      (SOP III-B)
#   (setsid nohup ./scripts/sop_queue.sh > sop_queue.log 2>&1 &)
cd "$(dirname "$0")/.."
source ~/COCCON_NEW/coccon/bin/activate
PREV_PID=${PREV_PID:-1352640}        # lab_queue.sh, which becomes scaling_eval.sh (exec keeps the pid)
echo "queued $(date); waiting for the current lab queue (pid $PREV_PID)"
while kill -0 "$PREV_PID" 2>/dev/null; do sleep 300; done

R=runs/full_1000
M=runs/memory_1000
B=(--backend ollama)
run() { echo "--- $* ($(date))"; "$@" || echo "!!! FAILED: $* ($(date))"; }

echo "=== 1. baselines + 2x2 $(date)"
run python scripts/run_all.py --scale full "${B[@]}" --run-dir $R --only calib,eval,score,human,report --keep-going

echo "=== 2. memory arms $(date)"
for arm in mem_none mem_summary mem_dense mem_event mem_needstate; do
    run python src/evaluate.py --arm $arm --run-dir $M "${B[@]}"
    run python src/metrics.py --dialogues $M/dialogues/$arm.jsonl "${B[@]}" --scale success --scale ip --scale pri
done
run python extra/memory_eval.py --run $M --judge "${B[@]}"
run python extra/memory_calibration.py --run $M "${B[@]}"

echo "=== 3. ES-MemEval $(date)"
run python extra/memeval_qa.py --run runs/memeval "${B[@]}"

echo "=== 4. alpha sweep $(date)"
for a in 0.05 0.2; do
    run python src/calibrate.py --dialogues $R/dialogues/calib_dec_ungated.jsonl \
        --out $R/conformal/calibration_alpha_$a.json --alpha $a "${B[@]}"
done
for pair in "gate_alpha_005 0.05" "gate_alpha_020 0.2"; do
    set -- $pair
    run python src/evaluate.py --arm $1 --run-dir $R "${B[@]}" --calibration $R/conformal/calibration_alpha_$2.json
    run python src/metrics.py --dialogues $R/dialogues/$1.jsonl "${B[@]}" --scale success --scale ip --scale pri
done

echo "=== 5. ExTES distribution shift $(date)"
run python src/evaluate.py --arm cellD_extes --run-dir $R "${B[@]}" --calibration $R/conformal/calibration.json --limit 100
run python src/metrics.py --dialogues $R/dialogues/cellD_extes.jsonl "${B[@]}" --scale ip --scale pri --scale aels

run python src/report.py --run $R
echo "=== SOP queue complete $(date)"
