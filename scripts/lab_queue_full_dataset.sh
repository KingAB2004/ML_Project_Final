#!/bin/bash
# Lab server (RTX 3060): build the full 1000-profile dataset with the current code in ~/COCCON_FULL, once the
# v3 run in ~/COCCON_NEW has finished (Phase C exits last). Waits, never stops anything. Resumable: rerun this
# script and finished steps are skipped (runs/full_1000/pipeline_state.json).
cd ~/COCCON_FULL
echo "queued $(date); waiting for the v3 run (phaseC50.sh, run_v3.sh) to exit"
while pgrep -f "phaseC50.sh|run_v3.sh" > /dev/null; do sleep 120; done
echo "v3 finished $(date); starting the 1000-profile dataset"
source ~/COCCON_NEW/coccon/bin/activate
python scripts/run_all.py --scale full --backend ollama --train --run-dir runs/full_1000 --only check,corpus,sft
status=$?
echo "dataset run exited with $status at $(date)"
wc -l data/corpus/*.jsonl data/sft/*.jsonl 2>/dev/null
echo "profile-jargon check (v3 had 54): $(grep -ci 'terminal need' data/corpus/annotations.jsonl 2>/dev/null)"
exit $status
