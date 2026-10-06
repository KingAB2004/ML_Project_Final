#!/bin/bash
# Option C: memory-aware fine-tuning. The current corpus trains the supporter with "[memory] none." in every
# input, so the fine-tuned models never learn to use what earlier sessions established. This rebuilds sessions
# 2-4 with the need-state memory brief in the generator prompt, fine-tunes on it, and evaluates the fine-tuned
# supporter with memory across linked sessions. The old corpus, SFT files and adapters are never touched.
#
#   ./scripts/memcorpus.sh data                 1. regenerate sessions 2-4 with memory -> filter -> annotate -> SFT
#   ./scripts/memcorpus.sh train <arm>          2. QLoRA on the memory corpus (arm: with_thoughts | wo_thoughts)
#   ./scripts/memcorpus.sh eval                 3. roll out + score the Option C arms on the 100 test profiles
#   (setsid nohup ./scripts/memcorpus.sh data > runs/memcorpus_data.log 2>&1 &)
#
# data and eval need Ollama (Qwen2.5-7B generator/seeker/analyzer, Mistral-Nemo judge); train and the eval rollouts
# need the GPU to themselves (transformers). Run on a machine holding data/corpus/sessions.jsonl and its LLM cache
# (runs/_cache) so session 1's unchanged prompts replay from the cache. Every step resumes where it stopped.
cd "$(dirname "$0")/.."
for venv in coccon venv .venv; do [ -f $venv/bin/activate ] && { source $venv/bin/activate; break; }; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

C=data/corpus_mem
S=data/sft_mem
R=runs/memcorpus
B=(--backend ollama)
run() { echo "--- $* ($(date))"; "$@" || { echo "!!! FAILED: $* ($(date))"; exit 1; }; }
free_gpu() { python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"; }

case "$1" in
data)
    pgrep -x ollama > /dev/null || { (setsid nohup ollama serve >> runs/ollama.log 2>&1 &); sleep 10; }
    echo "=== 1a. memory-aware sessions (session 1 replayed, sessions 2-4 regenerated) $(date)"
    run python src/sessions.py --memory --profiles data/profiles/profiles_trainval.jsonl \
        --source data/corpus/sessions.jsonl --sessions $C/sessions.jsonl "${B[@]}"
    run python src/sessions.py --check --sessions $C/sessions.jsonl
    echo "=== 1b. filter + annotate $(date)"
    run python src/filter.py --sessions $C/sessions.jsonl --out $C/sessions_filtered.jsonl "${B[@]}"
    run python src/annotate.py --sessions $C/sessions_filtered.jsonl --out $C/sessions_annotated.jsonl "${B[@]}"
    echo "=== 1c. SFT files $(date)"
    for arm in with_thoughts wo_thoughts; do
        run python src/build_sft.py --arm $arm --sessions $C/sessions_annotated.jsonl --out $S
    done
    # did the regenerated replies use memory? (old corpus: 27 of 12,444 later-session replies, 0.2%)
    for f in data/sft/with_thoughts_train.jsonl $S/with_thoughts_train.jsonl; do [ -f "$f" ] && python - "$f" <<'PY'
import json, re, sys
cue = re.compile(r"last time|you mentioned|you said before|last session|when we (last )?talked|you told me|"
                 r"since we (last )?(spoke|talked)", re.I)
rows = [json.loads(l) for l in open(sys.argv[1]) if "[elapsed]" in l]
hit = sum(bool(cue.search(r["target"].split("<response>")[-1])) for r in rows)
print(f"{sys.argv[1]}: {hit}/{len(rows)} later-session replies refer to an earlier session "
      f"({100 * hit / max(1, len(rows)):.1f}%)")
PY
    done
    echo "=== data complete $(date)" ;;
train)
    arm=${2:?usage: memcorpus.sh train with_thoughts|wo_thoughts}
    free_gpu
    run python src/train.py --arm "$arm" --sft-dir $S --out $R
    cp $R/train_summary.json $R/train_summary_$arm.json
    echo "=== train $arm complete $(date)" ;;
eval)
    # sftmem_zero_shot uses the existing adapter; the others the memory-trained ones
    declare -A adapter=([sftmem_zero_shot]=runs/full_1000/adapter_with_thoughts
                        [sftmem_with_thoughts]=$R/adapter_with_thoughts
                        [sftmem_wo_thoughts]=$R/adapter_wo_thoughts
                        [sftmem_no_memory]=$R/adapter_with_thoughts)
    for arm in sftmem_zero_shot sftmem_with_thoughts sftmem_wo_thoughts sftmem_no_memory; do
        [ -f ${adapter[$arm]}/adapter_config.json ] || { echo "skip $arm: no adapter ${adapter[$arm]}"; continue; }
        free_gpu
        run python src/evaluate.py --arm $arm --run-dir $R --adapter ${adapter[$arm]} --backend transformers
        run python src/metrics.py --dialogues $R/dialogues/$arm.jsonl "${B[@]}" --scale success --scale ip --scale pri
    done
    run python extra/memory_eval.py --run $R --judge "${B[@]}"
    run python src/report.py --run $R
    echo "=== eval complete $(date)" ;;
*)
    sed -n 2,15p "$0"; exit 1 ;;
esac
