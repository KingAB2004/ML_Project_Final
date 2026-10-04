#!/bin/bash
# v3 at 50 profiles, one GPU job at a time: B (corpus) -> C (fine-tune + evaluate) -> A (baselines).
cd ~/COCCON_NEW
./phaseB50.sh > phaseB50.log 2>&1 || { echo "phase B failed - see phaseB50.log"; exit 1; }
./phaseC50.sh > phaseC50.log 2>&1 || echo "phase C failed - see phaseC50.log"
./phaseA50.sh > phaseA50.log 2>&1 || echo "phase A failed - see phaseA50.log"
cd ~/COCCON_NEW && source coccon/bin/activate && python src/report.py --run runs/v3_50 --out runs/v3_50/results_v3.md
echo "run_v3 finished $(date)"
