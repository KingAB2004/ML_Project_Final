#!/bin/bash
# Phase C (50 profiles): QLoRA fine-tune the supporter on the Phase B corpus, then roll it out on the 40
# test profiles (transformers backend - Ollama cannot load an adapter) and score it with the same judge.
set -e
cd ~/COCCON_NEW && source coccon/bin/activate
RUN=runs/v3_50
test -s data/sft/with_thoughts_train.jsonl || { echo "no SFT data - run phase B first"; exit 1; }
while pgrep -f phaseA50.sh > /dev/null; do sleep 120; done
while pgrep -f dl_model.sh > /dev/null; do echo "waiting for the base model download"; sleep 120; done
for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done
# fewer fragmented blocks: the 7B QLoRA step runs within ~1 GB of the 12 GB card
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf $RUN/adapter_with_thoughts
python src/train.py --arm with_thoughts --out $RUN
# Point the arm at its adapter only for this phase: cells A-D default to supporter sft_with_thoughts, and
# the baseline phase must keep running them on the base model.
cp configs/arms.yaml configs/arms.yaml.pre_c; trap 'mv configs/arms.yaml.pre_c configs/arms.yaml' EXIT
sed -i "s|^  sft_with_thoughts: null.*|  sft_with_thoughts: $RUN/adapter_with_thoughts|" configs/arms.yaml
grep -q "sft_with_thoughts: $RUN/adapter_with_thoughts" configs/arms.yaml
python src/evaluate.py --arm sft_with_thoughts --run-dir $RUN --limit 40 --backend transformers
python src/metrics.py --dialogues $RUN/dialogues/sft_with_thoughts.jsonl
python src/counterfactual.py --dialogues $RUN/dialogues/sft_with_thoughts.jsonl
echo PHASE_C_DONE
