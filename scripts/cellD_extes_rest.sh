#!/bin/bash
# 10.10.10.104, after the ESC-RANK rerun: the rest of cell D on ExTES people (queue_104 scored success, ip, pri, aels
# only) - CRS, RAC, basic and counterfactual PRI, so it has the same scores as the other cells.
#   (setsid nohup ./scripts/cellD_extes_rest.sh > runs/cellD_extes_rest.log 2>&1 &)
cd "$(dirname "$0")/.."
source coccon/bin/activate
D=runs/baselines_1000/dialogues/cellD_extes.jsonl
run() { echo "--- $* ($(date))"; local t=$SECONDS; "$@" || echo "!!! FAILED: $* ($(date))"; echo "    took $((SECONDS - t)) s"; }
echo "queued $(date); waiting for the ESC-RANK rerun"
until grep -q "escrank rerun complete" runs/escrank_rerun.log 2>/dev/null; do sleep 300; done
run python src/metrics.py --dialogues $D --scale crs --scale rac --scale basic --backend ollama
run python src/counterfactual.py --dialogues $D --backend ollama
echo "=== cellD_extes rest complete $(date)"
