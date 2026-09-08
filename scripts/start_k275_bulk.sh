#!/usr/bin/env bash
set -euo pipefail

run_stamp=${1:?run stamp required}
[[ "$run_stamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid run stamp" >&2; exit 2; }
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3
state="$state_root/k275/bulk/$run_stamp"
publish_root="/mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2-rotating-uneven-v1-$run_stamp"
manager="glm53-k275-manager-${run_stamp:9:6}"

test -s "$state_root/k275/selection/k2-experts.json"
for node_service in \
    "mj-spark-1 glm53-exl3-head" \
    "mj-spark-2 glm53-exl3-worker" \
    "mj-spark-3 minimax-h3-comfy"; do
    read -r node service <<<"$node_service"
    running=$(ssh -F "$project/zima-ssh-config" "$node" "docker inspect -f '{{.State.Running}}' '$service' 2>/dev/null || echo missing")
    [ "$running" = false ] || { echo "$node protected service is not stopped: $service" >&2; exit 1; }
done

mkdir -p "$state"
if [ ! -e "$state/RUN_CONFIG.json" ]; then
    jq -n --arg run_stamp "$run_stamp" --arg publish_root "$publish_root" \
        --arg selection "$state_root/k275/selection/k2-experts.json" \
        --arg base_checkpoint "/mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z" \
        '{schema:"glm53-full-exl3-tp3.k275-run-config.v1",run_stamp:$run_stamp,target_bpw:"2.75",base_checkpoint:$base_checkpoint,publish_root:$publish_root,selection:$selection,geometry_id:"rotating-uneven-768-640-640-v1",bits:{k2:64,k3:192},service_policy:"leave_3bpw_candidate_and_Flash_H3_stopped"}' \
        > "$state/RUN_CONFIG.json"
fi
if [ ! -e "$state/bulk-window-start.txt" ]; then
    date -u +%Y-%m-%dT%H:%M:%S.%3NZ > "$state/bulk-window-start.txt"
fi

tmux kill-session -t "$manager" 2>/dev/null || true
tmux new-session -d -s "$manager" \
    "cd '$project' && exec python3 scripts/k275_bulk_manager.py --run-stamp '$run_stamp' >> '$state/manager.log' 2>&1"
echo "$run_stamp"
