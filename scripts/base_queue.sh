#!/bin/bash
# Second RTX 3060 (~/COCCON_BASE): the Ollama-only SOP experiments that need no fine-tuned adapter, moved off the
# lab server so they run while the lab trains the 7B. Same code, data and configs as the lab (lab configs copied
# verbatim: backend ollama, pri.n_rollouts 2). Items 4 and 5 read item 1's calibration, so all three run here;
# the lab's scripts/sop_queue.sh keeps the memory arms and ES-MemEval. Every step resumes where it stopped.
#   1. full-size baselines + 2 x 2: gate calibration, base_instruct, reactive_baseline, cells A-D (Outcomes 1-2)
#   4. alpha sweep: gate at alpha 0.05 and 0.20 (0.10 is cell D)                         (Outcome 3)
#   5. the gated system on ExTES profiles: violation rate under distribution shift      (SOP III-B)
#   (setsid nohup ./scripts/base_queue.sh > runs/base_queue.log 2>&1 &)
cd "$(dirname "$0")/.."
source venv/bin/activate
pgrep -x ollama > /dev/null || { (setsid nohup ollama serve >> runs/ollama.log 2>&1 &); sleep 10; }

R=runs/full_1000
B=(--backend ollama)
run() { echo "--- $* ($(date))"; "$@" || echo "!!! FAILED: $* ($(date))"; }

echo "=== 1. baselines + 2x2 $(date)"
run python scripts/run_all.py --scale full "${B[@]}" --run-dir $R --only calib,eval,score,human,report --keep-going

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
echo "=== base queue complete $(date)"
