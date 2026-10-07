#!/bin/bash
# Lab server (RTX 3060, 12 GB): idea 1 of the scaling study - test-profile evaluation of every fine-tuned size.
# For each size and arm: roll out the 100 test profiles (evaluate.py, transformers backend), then the judge
# scales and counterfactual PRI (metrics.py, counterfactual.py, Ollama). Every size meets the same seeker: the
# 7B base plays it, and a smaller supporter is loaded beside it (--supporter-base, llm.TransformersBackend).
#
# Waits for the 7B training queue to exit, never stops anything. Evaluates the sizes whose adapters are present
# (0.5B / 3B are copied up from the laptop into runs/scale_qwen2.5_<size>/), then keeps polling for the rest.
# Resumable: a finished size/arm is skipped, a half-finished rollout continues where it stopped.
#   ./scripts/scaling_eval.sh        results: runs/scaling_eval/<eval_tag>/, runs/scaling/extra/scaling/
#   TRAIN_QUEUE_PID=<pid> SIZES="3B Qwen3-4B" ./scripts/scaling_eval.sh     # wait for another queue, these sizes only
cd "$(dirname "$0")/.."
source ~/COCCON_NEW/coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRAIN_QUEUE_PID=${TRAIN_QUEUE_PID:-1165336}
echo "queued $(date); waiting for the 7B training queue (pid $TRAIN_QUEUE_PID)"
while kill -0 "$TRAIN_QUEUE_PID" 2>/dev/null; do sleep 300; done
echo "training queue gone $(date)"

SIZES=${SIZES:-"0.5B 7B 3B"}                         # what is ready first goes first
size_field() {   # size field -> value from extra/_shared.py MODEL_SIZES (base | dir | eval_tag)
    python -c "import sys; sys.path.insert(0, 'extra'); import _shared as s; print({'eval_tag': s.eval_tag('$1')}.get('$2') or s.MODEL_SIZES['$1']['$2'])"
}

free_gpu() {     # the judge stays resident in Ollama for minutes after scoring; the rollout needs the VRAM
    python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"
}

evaluate_one() {  # size arm; returns 0 when its scores exist
    local size=$1 arm=$2 run adapter
    run=runs/scaling_eval/$(size_field "$size" eval_tag)
    adapter=$(size_field "$size" dir)/adapter_$arm
    [ -f "$run/scores/sft_$arm/pri_summary.json" ] && return 0
    [ -f "$adapter/adapter_config.json" ] || return 1
    echo "=== $size $arm start $(date)"
    local base_args=()
    [ "$size" != "7B" ] && base_args=(--supporter-base "$(size_field "$size" base)")
    free_gpu
    python src/evaluate.py --arm "sft_$arm" --run-dir "$run" --adapter "$adapter" --backend transformers \
        "${base_args[@]}" || { echo "=== $size $arm rollout FAILED $(date)"; return 2; }
    python src/metrics.py --dialogues "$run/dialogues/sft_$arm.jsonl" --backend ollama \
        || { echo "=== $size $arm judge FAILED $(date)"; return 2; }
    python src/counterfactual.py --dialogues "$run/dialogues/sft_$arm.jsonl" --backend ollama \
        || { echo "=== $size $arm counterfactual FAILED $(date)"; return 2; }
    echo "=== $size $arm done $(date)"
    python extra/scaling.py > /dev/null 2>&1        # refresh the cross-size report after every finished arm
    return 0
}

while true; do
    pending=0
    for size in $SIZES; do
        for arm in with_thoughts wo_thoughts; do
            evaluate_one "$size" "$arm"
            rc=$?
            [ $rc -eq 1 ] && pending=1
            [ $rc -eq 2 ] && exit 1
        done
    done
    [ $pending -eq 0 ] && break
    echo "waiting for adapters still missing ($(date)); polling every 10 min"
    sleep 600
done
python extra/scaling.py
echo "=== scaling evaluation complete $(date)"
