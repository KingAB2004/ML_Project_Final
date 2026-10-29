#!/bin/bash
# Preference alignment (DPO) vs our pipeline (src/preference.py has the method). Lab RTX 3060, after the 12c rerun.
#   1  sample: the SFT listener writes 4 replies to each of 1200 training-split contexts
#   2  rank: base Qwen 7B (critic role, Ollama) scores each reply on the IP scale; least vs most intrusive = a pair
#   3  train: DPO on the SFT adapter (beta 0.1, 1 epoch)
#   4  final: SFT and DPO adapters on the 100 test people, 2-4 sessions each, same session seeds; all scores
#   5  report against steering, cells A-D and step 12 -> temp_results/18_preference_alignment_dpo
# Starts once the 12c rerun is complete; NO_WAIT=1 starts at once. Every step resumes where it stopped.
#   (setsid nohup ./scripts/preference.sh > runs/preference.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=runs/preference
SFT=runs/full_1000/adapter_with_thoughts
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "queued $(date); waiting for the 12c rerun"
until [ -n "$NO_WAIT" ] || grep -q "12c rerun complete" runs/sft_agents_12c_rerun.log 2>/dev/null; do sleep 300; done

echo "=== 1. sample $(date)"
free_gpu
run python src/preference.py sample --out $R --n 1200 --adapter $SFT
echo "=== 2. rank $(date)"
[ -s $R/pairs.jsonl ] || run python src/preference.py rank --out $R --backend ollama
echo "=== 3. DPO $(date)"
free_gpu
run python src/preference.py train --out $R --adapter $SFT
[ -s $R/adapter_dpo/adapter_model.safetensors ] || { echo "!!! no DPO adapter, stopping $(date)"; exit 1; }

echo "=== 4. final on test people $(date)"
run python src/evaluate.py --arm pref_mono --run-dir $R --backend transformers --adapter $SFT --tag sft
run python src/evaluate.py --arm pref_mono --run-dir $R --backend transformers --adapter $R/adapter_dpo --tag dpo
free_gpu
for t in dpo sft; do
    run python src/metrics.py --dialogues $R/dialogues/pref_mono_$t.jsonl --backend ollama
    run python src/counterfactual.py --dialogues $R/dialogues/pref_mono_$t.jsonl --backend ollama
done

echo "=== 5. report $(date)"
run python src/preference.py report --out $R
echo "=== preference complete $(date)"
