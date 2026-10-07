#!/bin/bash
# Scaling study on a local 8 GB card (RTX 4060 laptop): the same QLoRA recipe as the 7B adapters trained on the
# lab server (configs/default.yaml train:, same SFT files from the 1000-profile corpus), on Qwen2.5 0.5B and
# 3B Instruct (1.5B dropped for time). Each size runs both arms
# (with_thoughts, then wo_thoughts) before the next size starts: smallest first.
# Resumable: a size/arm whose adapter already exists is skipped.
#
#   ./scripts/scale_local.sh            # all sizes, both arms
#   SIZES="0.5B" ARMS="with_thoughts" ./scripts/scale_local.sh
#   SIZES=Qwen3-4B ./scripts/scale_local.sh   # the Qwen3 control (thinking off in the chat template)
cd "$(dirname "$0")/.."
source coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SIZES=${SIZES:-"0.5B"}   # 3B: scripts/scale_3b.sh
ARMS=${ARMS:-"with_thoughts wo_thoughts"}

for f in data/sft/with_thoughts_train.jsonl data/sft/wo_thoughts_train.jsonl; do
    [ -s "$f" ] || { echo "missing or empty $f - copy the dataset from the lab server first"; exit 1; }
done

for size in $SIZES; do
    for arm in $ARMS; do
        # base model and run dir from extra/_shared.py MODEL_SIZES (Qwen2.5 sizes and the Qwen3 controls)
        read -r model out < <(python -c "import sys; sys.path.insert(0, 'extra'); from _shared import MODEL_SIZES as M; print(M['$size']['base'], M['$size']['dir'])")
        if [ -f "$out/adapter_${arm}/adapter_config.json" ]; then
            echo "skip $size $arm: adapter exists"
            continue
        fi
        echo "=== $size $arm start $(date)"
        python src/train.py --arm "$arm" --model "$model" --out "$out" \
            || { echo "=== $size $arm FAILED $(date)"; exit 1; }
        # train.py writes one train_summary.json per run dir; keep one per arm
        mv "$out/train_summary.json" "$out/train_summary_${arm}.json"
        echo "=== $size $arm done $(date)"
    done
done

# ideas 2 and 3 (PVI, LoRA geometry) on whatever sizes are now complete
./scripts/local_analyses.sh
