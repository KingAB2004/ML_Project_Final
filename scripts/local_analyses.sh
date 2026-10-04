#!/bin/bash
# Laptop (RTX 4060, 8 GB): ideas 2 and 3 of the scaling study, for every size whose two adapters exist
# (0.5B and 3B trained here; 7B once its adapters are copied into runs/full_1000/). Each step keeps what it
# already computed, so rerunning after a new size arrives only does the new work.
#   ./scripts/local_analyses.sh        results: runs/scaling/extra/{lora_geometry,pvi,scaling}/report.md
cd "$(dirname "$0")/.."
source coccon/bin/activate
exec 9> runs/.local_analyses.lock
flock 9                                            # one analysis pass on the GPU at a time
while pgrep -f "src/train.py" > /dev/null; do sleep 300; done   # never beside a training run
echo "=== local analyses start $(date)"
python extra/lora_geometry.py --truncate || echo "lora_geometry FAILED"
python extra/pvi.py || echo "pvi FAILED"
python extra/scaling.py || echo "scaling FAILED"
echo "=== local analyses done $(date)"
