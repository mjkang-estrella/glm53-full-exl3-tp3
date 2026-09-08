#!/usr/bin/env bash
# Recover an interrupted bulk run without starting Flash/H3. The recovery
# archives prior control evidence, stops only owned experimental/restore
# processes, verifies resumable artifacts, and establishes a new strict
# encoder kernel window before launching one persistent Zima manager.
set -euo pipefail

cd "$(dirname "$0")/.."
run_stamp=${1:?bulk run stamp required}
[[ "$run_stamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid run stamp" >&2; exit 2; }

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3
state="$state_root/bulk/$run_stamp"
config="$state/RUN_CONFIG.json"
test -s "$config"
test ! -e "$state/COMPLETE.json"

# Stop only a restore process whose argv exactly identifies this bulk run.
mapfile -t restore_pids < <(
    ps -eo pid=,args= | awk -v stamp="$run_stamp" \
        '$2 ~ /(^|\/)bash$/ && $3 ~ /(^|\/)restore_protected_services[.]sh$/ && $4 == stamp {print $1}'
)
for pid in "${restore_pids[@]}"; do
    kill -TERM "$pid"
done
for _ in $(seq 1 12); do
    remaining=0
    for pid in "${restore_pids[@]}"; do kill -0 "$pid" 2>/dev/null && remaining=1; done
    (( remaining == 0 )) && break
    sleep 1
done
for pid in "${restore_pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && { echo "owned restore process $pid did not stop" >&2; exit 2; }
done

# The manager lock is authoritative even if an old tmux session was renamed.
if ! flock -n "$state/manager.lock" -c true; then
    echo "another bulk lease manager holds the run lock" >&2
    exit 2
fi
old_session=$(jq -r '.manager_session' "$config")
if tmux has-session -t "$old_session" 2>/dev/null; then
    echo "bulk manager session $old_session is still running" >&2
    exit 2
fi

nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
handoff=0
new_session=
recovery_failure() {
    code=$?
    trap - EXIT
    if (( code != 0 && handoff == 0 )); then
        [ -z "$new_session" ] || tmux kill-session -t "$new_session" 2>/dev/null || true
        for node_number in 1 2 3; do
            tmux kill-session -t "glm53-k3-lease-${run_stamp:9:6}-$node_number" \
                2>/dev/null || true
        done
        for node in "${nodes[@]}"; do
            ssh -F zima-ssh-config "$node" \
                "for name in \$(docker ps --format '{{.Names}}' | awk '/^glm53-tp3-k3-uneven-L[0-9][0-9][0-9]$/ {print}'); do docker stop --timeout 30 \"\$name\" >/dev/null 2>&1 || true; done; tmux list-sessions -F '#{session_name}' 2>/dev/null | awk '/^glm53-k3-${run_stamp:9:6}-L/ {print}' | xargs -r -n1 tmux kill-session -t" &
        done
        wait || true
        failed_at=$(date -u +%Y%m%dT%H%M%SZ)
        failure_dir="$state/recovery-failures/$failed_at"
        mkdir -p "$failure_dir"
        jq -n --arg run_stamp "$run_stamp" --arg failed_at "$failed_at" \
            --argjson exit_code "$code" \
            '{schema:"glm53-full-exl3-tp3.bulk-recovery-failure.v2",
              run_stamp:$run_stamp,failed_at:$failed_at,exit_code:$exit_code,
              experimental_workers_stopped:true,
              automatic_service_restore_attempted:false,
              service_policy:"leave_flash_and_h3_stopped"}' \
            > "$failure_dir/FAILURE.json.tmp"
        mv "$failure_dir/FAILURE.json.tmp" "$failure_dir/FAILURE.json"
    fi
    exit "$code"
}
trap recovery_failure EXIT

epoch=$(date -u +%Y%m%dT%H%M%SZ)
epoch_dir="$state/recovery-epochs/$epoch"
mkdir -p "$epoch_dir/prior-control-state"
for name in BLOCKED.json BLOCKED.md STATUS.json STATUS.md RUN_CONFIG.json \
    service-baseline.json bulk-window-start.txt UNCAUGHT-MANAGER-EXIT.txt; do
    [ ! -f "$state/$name" ] || cp "$state/$name" "$epoch_dir/prior-control-state/$name"
done

# Remove competing experimental control processes and containers, while
# retaining every local encoded artifact and source staging directory.
for node_number in 1 2 3; do
    tmux kill-session -t "glm53-k3-lease-${run_stamp:9:6}-$node_number" \
        2>/dev/null || true
done
for node in "${nodes[@]}"; do
    ssh -F zima-ssh-config "$node" \
        "for name in \$(docker ps --format '{{.Names}}' | awk '/^glm53-tp3-k3-uneven-L[0-9][0-9][0-9]$/ {print}'); do docker stop --timeout 30 \"\$name\" >/dev/null; done; tmux list-sessions -F '#{session_name}' 2>/dev/null | awk '/^glm53-k3-${run_stamp:9:6}-L/ {print}' | xargs -r -n1 tmux kill-session -t" &
done
wait

rollback_stamp=$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/capture_inventory.sh "$rollback_stamp" > "$epoch_dir/rollback-stamp.txt"
test -s "$state_root/rollback/$rollback_stamp/mj-zima/ARTIFACT_SHA256SUMS"

# Freeze the corrected recovery scripts on every worker before reconciliation.
for node in "${nodes[@]}"; do
    rsync -a --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude outputs/ --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" \
        "$node:$project/"
done

service_state_tmp="$epoch_dir/pre-stop-service-state.json.tmp"
jq -n '{schema:"glm53-full-exl3-tp3.recovery-service-state.v1",containers:{}}' \
    > "$service_state_tmp"
for index in 0 1 2; do
    node=${nodes[$index]}
    service=${services[$index]}
    values=$(ssh -F zima-ssh-config "$node" \
        "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.State.Restarting}} {{.RestartCount}}' '$service'")
    read -r running oom restarting restart_count <<< "$values"
    [ "$oom" = false ] && [ "$restarting" = false ] || {
        echo "$node:$service has an OOM-killed or restarting state" >&2
        exit 1
    }
    jq --arg key "$node:$service" --argjson running "$running" \
        --argjson oom "$oom" --argjson restarting "$restarting" \
        --argjson restart_count "$restart_count" \
        '.containers[$key]={running:$running,oom_killed:$oom,restarting:$restarting,restart_count:$restart_count}' \
        "$service_state_tmp" > "$service_state_tmp.next"
    mv "$service_state_tmp.next" "$service_state_tmp"
done
mv "$service_state_tmp" "$epoch_dir/pre-stop-service-state.json"

# No readiness or generation canary is required for recovery. If a preserved
# service happens to be running, stop it and retain its immutable inventory.
for index in 0 1 2; do
    node=${nodes[$index]}
    service=${services[$index]}
    running=$(ssh -F zima-ssh-config "$node" \
        "docker inspect -f '{{.State.Running}}' '$service'")
    if [ "$running" = true ]; then
        ssh -F zima-ssh-config "$node" \
            "docker stop --timeout 60 '$service' >/dev/null" &
    fi
done
wait

for index in 0 1 2; do
    node=${nodes[$index]}
    service=${services[$index]}
    values=$(ssh -F zima-ssh-config "$node" \
        "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}}' '$service'")
    read -r running oom <<< "$values"
    [ "$running" = false ] && [ "$oom" = false ] || {
        echo "$node:$service is not safely stopped" >&2
        exit 1
    }
    available=$(ssh -F zima-ssh-config "$node" \
        "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 )) || {
        echo "$node reserve below 12 GiB" >&2
        exit 1
    }
done

# Current hardware health is assessed only after service shutdown. Historical
# lifecycle events remain in their original evidence and are not reused as the
# encoder window.
quiet_start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$quiet_start" > "$epoch_dir/quiet-health-start.txt"
sleep 15
quiet_end=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
python3 scripts/kernel_audit.py --since "$quiet_start" --until "$quiet_end" \
    --output "$epoch_dir/pre-window-strict-kernel-audit.json"

publish_root=$(jq -r '.publish_root' "$config")
python3 scripts/reconcile_bulk.py --run-stamp "$run_stamp" \
    --publish-root "$publish_root" --repair \
    --output "$epoch_dir/RECONCILIATION.json"

bulk_start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$bulk_start" > "$state/bulk-window-start.txt.tmp"
mv "$state/bulk-window-start.txt.tmp" "$state/bulk-window-start.txt"
new_session="glm53-k3-manager-${epoch:9:6}"
jq --arg rollback_stamp "$rollback_stamp" --arg manager_session "$new_session" \
    --arg recovery_epoch "$epoch" \
    '.rollback_stamp=$rollback_stamp | .manager_session=$manager_session | .recovery_epoch=$recovery_epoch |
     .automatic_service_restore=false' \
    "$config" > "$config.tmp"
mv "$config.tmp" "$config"
jq -n --arg epoch "$epoch" --arg rollback_stamp "$rollback_stamp" \
    --arg manager_session "$new_session" --arg bulk_start "$bulk_start" \
    --arg reconciliation "$epoch_dir/RECONCILIATION.json" \
    '{schema:"glm53-full-exl3-tp3.bulk-recovery-epoch.v2",recovered_at:$epoch,
      rollback_stamp:$rollback_stamp,manager_session:$manager_session,
      encoder_window_start:$bulk_start,reconciliation:$reconciliation,
      protected_service_canary_required:false,automatic_service_restore:false}' \
    > "$epoch_dir/RECOVERY.json.tmp"
mv "$epoch_dir/RECOVERY.json.tmp" "$epoch_dir/RECOVERY.json"

bash scripts/resume_bulk.sh "$run_stamp"
for _ in $(seq 1 18); do
    if tmux has-session -t "$new_session" 2>/dev/null && \
        jq -e --arg start "$bulk_start" '.state == "RUNNING" and .updated_at >= $start' \
            "$state/STATUS.json" >/dev/null 2>&1; then
        break
    fi
    sleep 5
done
tmux has-session -t "$new_session" 2>/dev/null || {
    echo "recovery manager did not remain running" >&2
    exit 1
}
jq -e --arg start "$bulk_start" '.state == "RUNNING" and .updated_at >= $start' \
    "$state/STATUS.json" >/dev/null
handoff=1
trap - EXIT
printf '%s\n' "$epoch"
