#!/usr/bin/env bash
# Zima tmux controller; waits for the guarded Spark candidate, then runs tests.
set -euo pipefail
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/quality/20260908-scale-v2
candidate=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260907T224500Z/candidate/attempts/k275-quality-v2
py=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/eval-venv-numpy2.3.3/bin/python
cd "$project"
deadline=$((SECONDS+900))
while [[ ! -s "$candidate/READY.json" ]]; do
    [[ ! -s "$candidate/FAILED.json" && $SECONDS -lt $deadline ]] || exit 2
    sleep 5
done
jq -e '.passed and .dcp==3 and .max_num_batched_tokens==128' "$candidate/READY.json" >/dev/null
exec "$py" scripts/quality_scale_campaign.py --attempt k275-quality-v2 --expected-rows 2063 \
    --expert-fit "$root/activation-scale-fit-layer33.json" --root "$root" \
    --evidence /mnt/unas-models/ZAI/GLM-5.3-EXL3-K275-quality-experiments-20260908-scale-v2
