#!/bin/bash
# Lab RTX 3060, after DPO (scripts/preference.sh): the LLM psychology probes for the DPO model, and the dependence
# judge on the arms whose dialogues live here (step 12, the SFT and DPO multi-session runs).
#   (setsid nohup ./scripts/psych_lab.sh > runs/psych_lab.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "queued $(date); waiting for DPO"
until [ -n "$NO_WAIT" ] || grep -q "preference complete" runs/preference.log 2>/dev/null; do sleep 300; done

free_gpu
[ -s runs/psych/sycophancy/replies_dpo.jsonl ] || run python src/psych.py respond --system dpo
[ -s runs/psych/emobench/dpo.jsonl ] || run python src/psych.py emobench --system dpo
free_gpu
[ -s runs/psych/sycophancy/scores_dpo.jsonl ] || run python src/psych.py score --system dpo --backend ollama
for a in runs/sft_agents:sft_dec_gated_mem runs/preference:pref_mono_sft runs/preference:pref_mono_dpo; do
    [ -s runs/psych/dependence/${a#*:}.jsonl ] || run python src/psych.py dependence --arm $a --backend ollama
done
echo "=== psych_lab complete $(date)"
