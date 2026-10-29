#!/bin/bash
# Step 12 (10.10.2.41, RTX 3060 12 GB): the fine-tuned supporter as the Generator inside the four-agent pipeline
# (Analyzer, Strategist, Critic on the base weights) with the conformal gate and need-state memory.
#   12a  ungated rollouts on 30 calibration people with the fine-tuned Generator, then calibrate the gate on them
#        (the conformal guarantee needs calibration drafts from the same generator it gates)
#   12b  the gated arm on the 100 test people (2-4 sessions each, same seeds as cell D)
#   12c  judge scores + counterfactual PRI, same scorers as the 2 x 2 cells
# Every step resumes where it stopped.   (setsid nohup ./scripts/sft_agents.sh > runs/sft_agents.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=runs/full_1000/adapter_with_thoughts
R=runs/sft_agents
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "=== 12a. calibration with the fine-tuned Generator $(date)"
free_gpu
run python src/evaluate.py --arm sft_calib_dec_ungated --run-dir $R --limit 30 --adapter $A --backend transformers
[ -s $R/conformal/calibration.json ] || run python src/calibrate.py \
    --dialogues $R/dialogues/sft_calib_dec_ungated.jsonl --out $R/conformal/calibration.json --backend ollama

echo "=== 12b. fine-tuned + four agents + gate + memory, test people $(date)"
free_gpu
run python src/evaluate.py --arm sft_dec_gated_mem --run-dir $R --calibration $R/conformal/calibration.json \
    --adapter $A --backend transformers

echo "=== 12c. scoring $(date)"
run python src/metrics.py --dialogues $R/dialogues/sft_dec_gated_mem.jsonl --backend ollama
run python src/counterfactual.py --dialogues $R/dialogues/sft_dec_gated_mem.jsonl --backend ollama

echo "=== step 12 complete $(date)"
