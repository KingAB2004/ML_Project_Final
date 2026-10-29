#!/bin/bash
# Run a queue script once the GPU is usable (the lab's driver needs a fix first). Polls every 5 min.
#   (setsid nohup ./scripts/wait_gpu_then.sh ./scripts/sft_agents.sh > runs/sft_agents.log 2>&1 &)
cd "$(dirname "$0")/.."
echo "waiting for a usable GPU $(date)"
until nvidia-smi >/dev/null 2>&1 && coccon/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; do sleep 300; done
echo "GPU ready $(date)"
exec "$@"
