#!/bin/bash
# Lab server (RTX 3060): replaces the rest of scripts/sop_queue.sh after the need-state memory fix (7 Oct).
# Waits for the running baselines + 2 x 2 (run_all.py) to finish, then:
#   0.  data quality FIRST: our corpus vs ExTES / ESConv on the paper's Tables 2 (judge), 6 (ESC-RANK), 8
#   4.  alpha sweep at 0.05 / 0.20 on the same code as cells A-D, so the sweep and cell D (alpha 0.10) match
#   --  installs the fixed memory code staged in ~/memfix_stage (linked needs, real confirm / deny / resolve)
#   2.  all five memory arms on the fixed code, in a new run dir (runs/memory_1000 keeps the flat-memory run)
#       + memory metrics (judge) + memory calibration
#   5.  test-profile evaluation of the fine-tuned Qwen2.5-3B and Qwen3-4B (scripts/scaling_eval.sh)
#   3.  ES-MemEval question answering per memory type
# Every step resumes where it stopped; a failing step is logged and the queue moves on.
#   (setsid nohup ./scripts/sop_queue2.sh > sop_queue2.log 2>&1 &)
cd "$(dirname "$0")/.."
source ~/COCCON_NEW/coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WAIT_PIDS=${WAIT_PIDS:-"2265188 2143226"}   # run_all.py (baselines + 2 x 2) and the old sop_queue.sh
STAGE=${STAGE:-$HOME/memfix_stage}
# Never beside the old queue: if sop_queue.sh is not stopped, this waits for it to finish everything.
echo "queued $(date); waiting for pids $WAIT_PIDS"
for p in $WAIT_PIDS; do while kill -0 "$p" 2>/dev/null; do sleep 300; done; done

B=(--backend ollama)
run() { echo "--- $* ($(date))"; "$@" || echo "!!! FAILED: $* ($(date))"; }

echo "=== 0. data quality (first: dataset metrics, paper Tables 2 / 6 / 8) $(date)"
# dialogue samples: 200 of our corpus, 100 ExTES, 100 ESConv (extra/data_quality.py samples, seed 0)
[ -s runs/data_quality/dialogues/ours.jsonl ] || run python extra/data_quality.py samples
for set in ours extes esconv; do      # Table 2: our judge on the paper's scales
    [ -s runs/data_quality/scores/$set/rac.jsonl ] || run python src/metrics.py \
        --dialogues runs/data_quality/dialogues/$set.jsonl --out runs/data_quality/scores/$set "${B[@]}" \
        --scale basic --scale crs --scale rac
done
python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"
for set in ours extes esconv; do      # Table 6: ESC-RANK, the paper's scorer (older transformers on the path)
    PYTHONPATH=$HOME/escrank_pkgs run python scripts/escrank_score.py --set $set
done
run python extra/data_quality.py table

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

echo "=== installing the fixed memory code from $STAGE $(date)"
for f in src/memory.py src/agents.py prompts/agent_analyzer.md extra/memeval_qa.py; do
    [ -f "$STAGE/$f" ] && cp "$STAGE/$f" "$f" && echo "installed $f"
done
python -c "import sys; sys.path[:0] = ['src']; import memory, agents; assert hasattr(memory.NeedStateMemory, 'valid_parent'); print('fixed memory importable')" \
    || { echo "!!! fixed memory code not installed; stopping before the memory arms $(date)"; exit 1; }

M=runs/memory_1000_v2
mem_arm() {
    run python src/evaluate.py --arm "$1" --run-dir $M "${B[@]}"
    run python src/metrics.py --dialogues $M/dialogues/$1.jsonl "${B[@]}" --scale success --scale ip --scale pri
}
echo "=== 2. memory arms (fixed need-state memory) $(date)"
for arm in mem_needstate mem_none mem_summary mem_dense mem_event; do
    mem_arm $arm
done
run python extra/memory_eval.py --run $M --judge "${B[@]}"
run python extra/memory_calibration.py --run $M "${B[@]}"

echo "=== 5. test-profile evaluation: Qwen2.5-3B, Qwen3-4B $(date)"
# polls every 10 min until the Qwen3-4B adapter is uploaded from the laptop, then evaluates it
TRAIN_QUEUE_PID=4194000 SIZES="3B Qwen3-4B" run ./scripts/scaling_eval.sh

echo "=== 3. ES-MemEval $(date)"
run python extra/memeval_qa.py --run runs/memeval "${B[@]}"

echo "=== SOP queue 2 complete $(date)"
