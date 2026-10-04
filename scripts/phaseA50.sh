#!/bin/bash
# Phase A (v3): the six baseline arms on the 40 held-out test profiles. The conformal gate is fitted on
# cell C dialogues of 25 CALIBRATION profiles, so it is never calibrated on the profiles it is scored on.
set -e
cd ~/COCCON_NEW && source coccon/bin/activate
RUN=runs/v3_50; N=40
python src/evaluate.py --arm calib_dec_ungated --run-dir $RUN --limit 25
python src/calibrate.py --dialogues $RUN/dialogues/calib_dec_ungated.jsonl --out $RUN/conformal/calibration.json
ARMS="base_instruct reactive_baseline cellA_mono_ungated cellB_mono_gated cellC_dec_ungated cellD_dec_gated"
for a in $ARMS; do
  python src/evaluate.py --arm $a --run-dir $RUN --limit $N --calibration $RUN/conformal/calibration.json
done
python ./leakcheck.py $RUN
for a in $ARMS; do
  python src/metrics.py --dialogues $RUN/dialogues/$a.jsonl
  python src/counterfactual.py --dialogues $RUN/dialogues/$a.jsonl
done
python src/report.py --run $RUN --out $RUN/results_v3.md
mkdir -p reports && cp $RUN/results_v3.md reports/results.md
echo PHASE_A_DONE
