#!/usr/bin/env bash
# Resume the Zima manager after a control-process interruption while protected
# services remain stopped. It never starts or restores a protected service.
set -euo pipefail

cd "$(dirname "$0")/.."
run_stamp=${1:?bulk run stamp required}
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/bulk/$run_stamp"
config="$state/RUN_CONFIG.json"
test -s "$config"
test ! -e "$state/COMPLETE.json"

nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
for index in 0 1 2; do
    running=$(ssh -F zima-ssh-config "${nodes[$index]}" \
        "docker inspect -f '{{.State.Running}}' '${services[$index]}'")
    if [ "$running" != false ]; then
        echo "protected service $service is running on ${nodes[$index]}; use restart_bulk_after_block.sh to capture and stop it without a canary" >&2
        exit 2
    fi
    available=$(ssh -F zima-ssh-config "${nodes[$index]}" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 )) || { echo "${nodes[$index]} reserve below 12 GiB" >&2; exit 2; }
done

manager_session=$(jq -r '.manager_session' "$config")
if tmux has-session -t "$manager_session" 2>/dev/null; then
    echo "$manager_session already running"
    exit 0
fi
rollback_stamp=$(jq -r '.rollback_stamp' "$config")
publish_root=$(jq -r '.publish_root' "$config")
qualification_seconds=$(jq -r '.qualification_seconds' "$config")
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
manager_command=$(printf '%q ' bash "$project/scripts/run_bulk_manager_guarded.sh" \
    "$run_stamp" "$rollback_stamp" "$publish_root" "$qualification_seconds")
tmux new-session -d -s "$manager_session" "$manager_command >> '$state/manager.log' 2>&1"
printf '%s\n' "$manager_session"
