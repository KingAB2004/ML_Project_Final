#!/bin/bash
# Lab server, 6 Oct: 7B PVI rerun, then the memory arms + memory metrics, then ES-MemEval (scripts/sop_queue.sh).
# Lets the running 7B wo_thoughts test evaluation finish, then stops scaling_eval.sh before its 3B step (the 3B
# is not evaluated here). Never touches anything else.
#   (setsid nohup ./scripts/lab_chain.sh > lab_chain.log 2>&1 &)
cd "$(dirname "$0")/.."
source ~/COCCON_NEW/coccon/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EVAL_PID=1352640
DONE=runs/scaling_eval/qwen2.5_7B/scores/sft_wo_thoughts/pri_summary.json
echo "queued $(date); waiting for $DONE"
while [ ! -f $DONE ] && kill -0 $EVAL_PID 2>/dev/null; do sleep 20; done
kill $EVAL_PID 2>/dev/null; sleep 3
pkill -f "runs/scaling_eval/qwen2.5_3B"
echo "7B wo_thoughts test evaluation done; scaling_eval.sh stopped $(date)"

echo "=== 7B PVI $(date)"
python -c "import sys; sys.path.insert(0, 'src'); from train import free_gpu_from_ollama; free_gpu_from_ollama()"
python extra/pvi.py --sizes 7B || echo "!!! pvi 7B FAILED $(date)"
echo "=== 7B PVI done $(date)"

PREV_PID=4194000 exec ./scripts/sop_queue.sh      # no such pid: starts at once
