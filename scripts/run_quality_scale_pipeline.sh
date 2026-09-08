#!/usr/bin/env bash
# Runs persistently in Zima tmux. All GPU work is inside the guarded TP3 serve.
set -euo pipefail
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/quality/20260908-scale-v1
attempt=k275-quality-v1
candidate=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260907T224500Z/candidate/attempts/$attempt
evidence=/mnt/unas-models/ZAI/GLM-5.3-EXL3-K275-quality-experiments-20260908-scale-v1
py=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/eval-venv-numpy2.3.3/bin/python
cd "$project"
deadline=$((SECONDS+900))
while [[ ! -s "$candidate/READY.json" ]]; do
    [[ ! -s "$candidate/FAILED.json" && $SECONDS -lt $deadline ]] || exit 2
    sleep 5
done
jq -e '.passed and .dcp==3 and .max_num_batched_tokens==256' "$candidate/READY.json" >/dev/null
if [[ ! -s "$root/CALIBRATION_CAPTURE.json" ]]; then
    "$py" scripts/quality_scale_campaign.py --capture-only --attempt "$attempt" --root "$root" --evidence "$evidence"
fi
exec "$py" scripts/quality_scale_campaign.py --attempt "$attempt" --root "$root" --evidence "$evidence"
