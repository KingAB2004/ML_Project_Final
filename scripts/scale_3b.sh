#!/bin/bash
# 3B half of the local scaling study: Qwen2.5-3B-Instruct, with_thoughts then wo_thoughts, same recipe as
# scale_local.sh. Runs detached (survives closing the terminal); a finished arm is skipped on a rerun.
#   ./scripts/scale_3b.sh          then follow:  tail -f runs/scale_3b.log
cd "$(dirname "$0")/.."
if pgrep -f "src/train.py|extra/pvi.py|extra/lora_geometry.py" > /dev/null; then
    echo "the GPU is busy (training or a local analysis) - wait for it"; exit 1
fi
(SIZES=3B setsid nohup ./scripts/scale_local.sh >> runs/scale_3b.log 2>&1 &)
echo "3B started $(date); log: runs/scale_3b.log"
