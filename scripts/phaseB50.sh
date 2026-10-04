#!/bin/bash
# Phase B (50 profiles): the training corpus from 45 train + 5 val profiles. Test and calibration
# profiles never enter it. Resumable: LLM calls replay from runs/_cache, finished sessions are skipped.
set -e
cd ~/COCCON_NEW && source coccon/bin/activate
P=data/profiles/profiles_corpus50.jsonl
{ head -n 45 data/profiles/profiles_train.jsonl; head -n 5 data/profiles/profiles_val.jsonl; } > $P
echo "corpus profiles: $(wc -l < $P)"
python src/dialogue.py --profiles $P
python src/sessions.py --profiles $P
python src/sessions.py --check
python src/filter.py
python src/annotate.py
python src/build_sft.py --arm with_thoughts
python src/build_sft.py --arm wo_thoughts
wc -l data/corpus/*.jsonl data/sft/*.jsonl
echo PHASE_B_DONE
