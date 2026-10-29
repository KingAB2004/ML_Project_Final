#!/bin/bash
# 10.10.10.104: ESC-RANK (paper Table 6) on the data-quality samples, after scripts/queue_104.sh frees the GPU,
# then the data-quality table again. Needs ~/escrank_venv (transformers 4.41, sentencepiece 0.2.0: 0.2.2 rejects
# InternLM2's tokenizer) and the InternLM2-chat-7b base in the HF cache.
#   (setsid nohup ./scripts/escrank_104.sh > runs/escrank_104.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
echo "queued $(date); waiting for queue_104"
until grep -q "queue_104 complete" runs/queue_104.log 2>/dev/null; do sleep 300; done
python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"
for set in ours extes esconv; do
    echo "--- escrank $set $(date)"
    ESCRANK_ENABLE=1 ~/escrank_venv/bin/python scripts/escrank_score.py --set $set || echo "!!! FAILED escrank $set"
done
python extra/data_quality.py table
echo "=== escrank_104 complete $(date)"
