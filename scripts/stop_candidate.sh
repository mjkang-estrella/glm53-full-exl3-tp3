#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
stopped_at=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
for rank in 0 1 2; do
    node=${nodes[$rank]}
    ssh -F zima-ssh-config "$node" \
        "docker stop --time 60 'glm53-k3-candidate-rank$rank' >/dev/null 2>&1 || true; \
         tmux kill-session -t 'glm53-k3-candidate-watchdog-$rank' 2>/dev/null || true; \
         tmux kill-session -t 'glm53-k3-candidate-log-$rank' 2>/dev/null || true" &
done
wait || true
mkdir -p "$state"
printf '%s\n' "$stopped_at" > "$state/stopped-at.txt"
if [[ -s "$state/started-at.txt" ]]; then
    python3 "$project/scripts/kernel_audit.py" --since "$(cat "$state/started-at.txt")" \
        --until "$stopped_at" --output "$state/final-kernel-audit.json"
fi
