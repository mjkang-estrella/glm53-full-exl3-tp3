#!/usr/bin/env bash
# Last-resort guard around the Python coordinator. Any nonzero manager exit
# stops only experimental workers and writes durable diagnostics. Protected
# Flash/H3 services are never started by this guard.
set -uo pipefail

cd "$(dirname "$0")/.."
run_stamp=${1:?run stamp required}
rollback_stamp=${2:?rollback stamp required}
publish_root=${3:?publish root required}
qualification_seconds=${4:?qualification seconds required}
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/bulk/$run_stamp"
queue="$state/queue"

python3 scripts/bulk_lease_manager.py --run-stamp "$run_stamp" \
    --rollback-stamp "$rollback_stamp" --publish-root "$publish_root" \
    --qualification-seconds "$qualification_seconds"
code=$?
if (( code == 0 )); then
    exit 0
fi

# Bound every target by the run-specific local sessions and the exact
# experimental container/session namespaces used by this project.
for node_number in 1 2 3; do
    tmux kill-session -t "glm53-k3-lease-${run_stamp:9:6}-$node_number" \
        2>/dev/null || true
done
for node in mj-spark-1 mj-spark-2 mj-spark-3; do
    ssh -F zima-ssh-config "$node" \
        "for name in \$(docker ps --format '{{.Names}}' | awk '/^glm53-tp3-k3-uneven-L[0-9][0-9][0-9]$/ {print}'); do docker stop --timeout 30 \"\$name\" >/dev/null 2>&1 || true; done; tmux list-sessions -F '#{session_name}' 2>/dev/null | awk '/^glm53-k3-${run_stamp:9:6}-L/ {print}' | xargs -r -n1 tmux kill-session -t" &
done
wait || true

stamp=$(date -u +%Y%m%dT%H%M%SZ)
incident="$state/guard-incidents/$stamp"
mkdir -p "$incident"
for node in mj-spark-1 mj-spark-2 mj-spark-3; do
    ssh -F zima-ssh-config "$node" \
        "docker ps -a --format '{{.Names}}|{{.State}}|{{.Status}}' | awk '/^(glm53-tp3-k3-uneven-L|glm53-exl3-head|glm53-exl3-worker|minimax-h3-comfy)/ {print}'; awk '/MemAvailable:|SwapFree:/ {print}' /proc/meminfo" \
        > "$incident/$node.txt" 2>&1 || true
done
jq -n --arg run_stamp "$run_stamp" --arg at "$stamp" --argjson exit_code "$code" \
    --arg queue "$queue" \
    '{schema:"glm53-full-exl3-tp3.manager-guard-incident.v2",run_stamp:$run_stamp,
      recorded_at:$at,manager_exit_code:$exit_code,experimental_workers_stopped:true,
      automatic_service_restore_attempted:false,
      service_policy:"leave_flash_and_h3_stopped",queue:$queue}' \
    > "$incident/INCIDENT.json.tmp"
mv "$incident/INCIDENT.json.tmp" "$incident/INCIDENT.json"
printf '%s\n' "manager exited $code; experimental workers stopped; Flash/H3 were not restored; inspect $incident" \
    > "$incident/README.txt"
exit "$code"
