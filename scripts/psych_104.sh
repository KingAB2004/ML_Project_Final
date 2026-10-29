#!/bin/bash
# 10.10.10.104, after the cellD_extes scoring: the LLM psychology probes (src/psych.py) for every system that does not
# need the DPO adapter - sycophancy replies, EmoBench, then judge scoring and the dependence judge on the multi-session
# arms present here. The DPO system runs on the lab once it exists (scripts/psych_lab.sh).
#   (setsid nohup ./scripts/psych_104.sh > runs/psych_104.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "queued $(date); waiting for the cellD_extes scoring"
until [ -n "$NO_WAIT" ] || grep -q "cellD_extes rest complete" runs/cellD_extes_rest.log 2>/dev/null; do sleep 300; done

echo "=== replies and EmoBench (transformers) $(date)"
free_gpu
for s in base steered sft dec_gated sft_dec_gated; do
    [ -s runs/psych/sycophancy/replies_$s.jsonl ] || run python src/psych.py respond --system $s
done
for s in base steered sft; do
    [ -s runs/psych/emobench/$s.jsonl ] || run python src/psych.py emobench --system $s
done

echo "=== judge (Ollama) $(date)"
free_gpu
for s in base steered sft dec_gated sft_dec_gated; do
    [ -s runs/psych/sycophancy/scores_$s.jsonl ] || run python src/psych.py score --system $s --backend ollama
done
for a in runs/baselines_1000:cellA_mono_ungated runs/baselines_1000:cellD_dec_gated runs/memory_v2:mem_none \
         runs/memory_v2:mem_needstate runs/steering:steer_mono_L12_c0 runs/steering:steer_mono_L12_c0.2 \
         runs/sft7b_multi:sft7b_multi_nomem runs/sft7b_multi:sft7b_multi_mem; do
    [ -s runs/psych/dependence/${a#*:}.jsonl ] || run python src/psych.py dependence --arm $a --backend ollama
done
echo "=== psych_104 complete $(date)"
