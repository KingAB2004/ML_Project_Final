#!/bin/bash
# Copy the 1000-profile run's results from runs/ (ignored by git) into results/v3_full1000/ (tracked), so the
# results folder always holds the latest numbers. Rerun whenever a new size, analysis or evaluation finishes.
# Model weights are never copied (too large for git); everything else that is a result is.
#   ./scripts/collect_results.sh
cd "$(dirname "$0")/.."
D=results/v3_full1000
source coccon/bin/activate
python extra/dataset_stats.py --out "$D/dataset/dataset_stats.json"

for run in runs/scale_qwen2.5_*/ runs/full_1000/; do
    [ -d "$run" ] || continue
    size=$(basename "$run"); size=${size#scale_qwen2.5_}; [ "$size" = "full_1000" ] && size=7B
    out="$D/training/$size"
    mkdir -p "$out"
    cp "$run"/train_summary*.json "$out"/ 2>/dev/null
    for adapter in "$run"/adapter_*/; do
        arm=$(basename "$adapter"); arm=${arm#adapter_}
        last=$(ls -d "$adapter"checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        [ -n "$last" ] && cp "$last/trainer_state.json" "$out/trainer_state_$arm.json"
        [ -f "$adapter/adapter_config.json" ] && cp "$adapter/adapter_config.json" "$out/adapter_config_$arm.json"
    done
    cp runs/scaling/extra/scaling/training_curves_$size.png "$out"/ 2>/dev/null
done
# training logs without the progress bars
for log in runs/scale_local.log runs/scale_3b.log runs/full_1000/full_train.log; do
    [ -f "$log" ] && tr '\r' '\n' < "$log" | grep -v -E "it/s\]|s/it\]|Loading weights|^\s*$" > "$D/training/$(basename "$log" .log)_clean.log"
done

mkdir -p "$D/analyses"
cp -r runs/scaling/extra/. "$D/analyses/"
if [ -d runs/scaling_eval ]; then
    mkdir -p "$D/test_eval"
    cp -r runs/scaling_eval/. "$D/test_eval/"
fi
# SOP evaluations (lab RTX 3060): baselines + 2 x 2 + alpha sweep, memory arms. The LLM call log is left out.
# memory_calibration only once the lab's own (Ollama judge) run lands: a local run without --backend is echo.
for run in baselines_1000 memory_1000; do
    [ -d runs/$run ] || continue
    mkdir -p "$D/$run"
    (cd runs/$run && tar cf - --exclude=calls.jsonl --exclude=pipeline_state.json --exclude=memory_calibration .) |
        tar xf - -C "$D/$run"
done
echo "collected into $D"
