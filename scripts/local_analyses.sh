#!/bin/bash
# Laptop (RTX 4060, 8 GB): ideas 2 and 3 of the scaling study, for every size whose two adapters exist
# trained here (0.5B, 3B). The 7B's GPU analyses run on the lab server (scripts/lab_queue.sh); their outputs
# are copied into runs/scaling/extra/ and picked up by the CPU-only steps. Each step keeps what it
# already computed, so rerunning after a new size arrives only does the new work.
#   ./scripts/local_analyses.sh        results: runs/scaling/extra/{lora_geometry,pvi,scaling}/report.md
cd "$(dirname "$0")/.."
source coccon/bin/activate
exec 9> runs/.local_analyses.lock
flock 9                                            # one analysis pass on the GPU at a time
while pgrep -f "src/train.py" > /dev/null; do sleep 300; done   # never beside a training run
S=${ANALYSIS_SIZES:-"0.5B 3B"}                        # GPU steps for these sizes; summaries cover every size
echo "=== local analyses start $(date)"
python extra/lora_geometry.py --sizes $S --truncate || echo "lora_geometry FAILED"
python extra/lora_geometry.py || echo "lora_geometry summary FAILED"   # CPU: every size with adapters, cached curves
python extra/pvi.py --sizes $S || echo "pvi FAILED"
python extra/scaling.py || echo "scaling FAILED"
echo "=== local analyses done $(date)"
