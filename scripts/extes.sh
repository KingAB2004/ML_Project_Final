#!/bin/bash
# Training-corpus comparison (10.10.2.41): our corpus vs ExTES, both reply-only targets on the untuned
# Qwen2.5-7B-Instruct with the same QLoRA settings. ExTES is sampled to 15k examples (0.7 x our 21.4k
# train + val), whole dialogues, so the comparison is about the data, not its size.
# Starts after step 12 (scripts/sft_agents.sh); NO_WAIT=1 starts at once. Every step resumes where it stopped.
#   (setsid nohup ./scripts/extes.sh > runs/extes.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=runs/corpus_extes
OURS=runs/full_1000/adapter_wo_thoughts
EXTES=$R/adapter_corpus_extes
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

echo "queued $(date); waiting for step 12"
until [ -n "$NO_WAIT" ] || grep -q "step 12 complete" runs/sft_agents.log 2>/dev/null; do sleep 300; done

echo "=== convert ExTES (15k examples) $(date)"
[ -s data/sft/corpus_extes_train.jsonl ] || run python src/convert_corpus.py --set extes --raw data/raw/extes.json --max-examples 15000

echo "=== fine-tune the untuned 7B on ExTES $(date)"
free_gpu
[ -s $EXTES/adapter_model.safetensors ] || run python src/train.py --arm corpus_extes --out $R

ev() {   # arm adapter limit-args...
    local arm=$1 adapter=$2; shift 2
    free_gpu
    run python src/evaluate.py --arm $arm --run-dir $R --adapter $adapter --backend transformers "$@"
    run python src/metrics.py --dialogues $R/dialogues/$arm.jsonl --backend ollama
    run python src/counterfactual.py --dialogues $R/dialogues/$arm.jsonl --backend ollama
}
echo "=== evaluation $(date)"
ev corpus_extes                $EXTES             # ExTES model, our 100 test people (vs sft_wo_thoughts)
ev corpus_extes_extes_profiles $EXTES --limit 100 # ExTES model, 100 ExTES people
ev corpus_ours_extes_profiles  $OURS  --limit 100 # our wo_thoughts model, the same 100 ExTES people

echo "=== ExTES comparison complete $(date)"
