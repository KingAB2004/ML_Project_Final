#!/bin/bash
# Activation steering vs our pipeline (src/steering.py has the method). Lab RTX 3060, after step 12.
#   1  steering vectors from 400 training-split contexts (untuned 7B, gentle vs intrusive instruction contrast)
#   2  sweep: layers 12 and 16 x strength 0.1 / 0.2 / 0.4, plus the coef-0 control, on 20 calibration people,
#      2-4 sessions each; judge IP + Success; the pre-registered rule picks one setting
#   3  final: the picked setting and the coef-0 control on the 100 test people (2-4 sessions each), all scores
#   4  report against cells A-D and step 12 -> temp_results/15_activation_steering
# Starts once step 12 is complete; NO_WAIT=1 starts at once. Every step resumes where it stopped.
#   (setsid nohup ./scripts/steering.sh > runs/steering.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=runs/steering
V=$R/steering_vectors.pt
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "queued $(date); waiting for step 12"
until [ -n "$NO_WAIT" ] || grep -q "step 12 complete" runs/sft_agents.log 2>/dev/null; do sleep 300; done

echo "=== 1. steering vectors $(date)"
free_gpu
run python src/steering.py build --out $V --n 400

echo "=== 2. sweep on calibration people $(date)"
SWEEP="12:0 12:0.1 12:0.2 12:0.4 16:0.1 16:0.2 16:0.4"
for s in $SWEEP; do
    run python src/evaluate.py --arm steer_calib --run-dir $R --limit 20 --backend transformers --steer $s --steer-vector $V
done
free_gpu
for f in $R/dialogues/steer_calib_L*_c*.jsonl; do
    run python src/metrics.py --dialogues $f --scale ip --scale success --backend ollama
done
run python src/steering.py select --run $R --arm steer_calib
[ -s $R/selected.txt ] || { echo "!!! no setting selected, stopping $(date)"; exit 1; }
PICK=$(cat $R/selected.txt)
LAYER=${PICK%%:*}

echo "=== 3. final on test people: $PICK and its control $LAYER:0 $(date)"
free_gpu
for s in $LAYER:0 $PICK; do
    run python src/evaluate.py --arm steer_mono --run-dir $R --backend transformers --steer $s --steer-vector $V
done
free_gpu
for f in $R/dialogues/steer_mono_L*_c*.jsonl; do
    run python src/metrics.py --dialogues $f --backend ollama
    run python src/counterfactual.py --dialogues $f --backend ollama
done

echo "=== 4. report $(date)"
run python src/steering.py report --run $R --arm steer_mono --out temp_results/15_activation_steering
echo "=== steering complete $(date)"
