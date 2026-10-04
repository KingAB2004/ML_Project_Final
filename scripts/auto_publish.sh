#!/bin/bash
# Keep GitHub up to date while the remaining results come in: every 30 min, collect the latest results into
# results/v3_full1000/ and, if anything changed, commit and push it. Only results/ and runs/scaling/ are
# committed, never work in progress elsewhere. Stops once every pending result is in (7B and 3B training,
# all six test evaluations) or after 6 days.
#   (setsid nohup ./scripts/auto_publish.sh > runs/auto_publish.log 2>&1 &)
cd "$(dirname "$0")/.."
D=results/v3_full1000
deadline=$(( $(date +%s) + 6 * 86400 ))
complete() {
    [ -f $D/training/7B/trainer_state_wo_thoughts.json ] && [ -f $D/training/3B/trainer_state_wo_thoughts.json ] \
        && [ "$(ls $D/test_eval/*/scores/*/pri_summary.json 2>/dev/null | wc -l)" -ge 6 ]
}
while [ "$(date +%s)" -lt "$deadline" ]; do
    ./scripts/collect_results.sh > /dev/null 2>&1
    git add results runs/scaling
    if ! git diff --cached --quiet; then
        git commit -q -m "Update results $(date '+%Y-%m-%d %H:%M')" && git push -q origin main \
            && echo "$(date) pushed: $(git log -1 --format=%h)" || echo "$(date) push FAILED"
    fi
    complete && { echo "$(date) all results in; stopping"; exit 0; }
    sleep 1800
done
echo "$(date) deadline reached; stopping"
