#!/usr/bin/env bash
# Final Phase-1 gate and one-time bulk lifecycle transition.
set -euo pipefail

cd "$(dirname "$0")/.."
run_stamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
real_stamp=${2:?passing real uneven-layer stamp required}
runtime_stamp=${3:?passing runtime-gate stamp required}
capacity=${4:?passing capacity projection required}
[[ "$run_stamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid run stamp" >&2; exit 2; }

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3
real_dir="$state_root/qualification/real-uneven-layer-$real_stamp"
runtime_dir="$state_root/qualification/runtime-gate-$runtime_stamp"
state="$state_root/bulk/$run_stamp"
publish_root="/mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-$run_stamp"
manager_session="glm53-k3-manager-${run_stamp:9:6}"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)

test ! -e "$state"
test ! -e "$publish_root"
test -w /mnt/unas-models/ZAI
jq -e '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1" and (.rotations|length) == 3 and all(.rotations[];.passed)' "$runtime_dir/SUMMARY.json" >/dev/null
jq -e '.passed == true and .windows.candidate_runtime.fault_count == 0 and .windows.post_ready.fault_count == 0' "$runtime_dir/lifecycle-audit.json" >/dev/null
jq -e '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1" and .experts == 256' "$real_dir/layer-verification.json" >/dev/null
jq -e '.passed == true' "$real_dir/encoder-window-kernel-audit.json" >/dev/null
jq -e '.passed == true' "$real_dir/h3-restore-kernel-audit.json" >/dev/null
test -s "$real_dir/PASSED"
jq -e '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1" and .context_tokens == 32768 and .minimum_optimistic_reserve_gib >= 12' "$capacity" >/dev/null
qualification_seconds=$(jq -r '.elapsed_expert_seconds' "$real_dir/remote-layer/LAYER_RECEIPT.json")
python3 - "$qualification_seconds" <<'PY'
import math, sys
value=float(sys.argv[1])
assert math.isfinite(value) and value > 0
PY

# A new rollback inventory and real generation canary are captured immediately
# before the protected services are stopped.
rollback_stamp=$(date -u +%Y%m%dT%H%M%SZ)
bash scripts/capture_inventory.sh "$rollback_stamp" >/dev/null
rollback="$state_root/rollback/$rollback_stamp"
test -s "$rollback/mj-zima/ARTIFACT_SHA256SUMS"
mkdir -p "$state/evidence"
[ ! -f "$state_root/STATUS.md" ] || cp "$state_root/STATUS.md" "$state/evidence/prior-STATUS.md"
[ ! -f "$state_root/BLOCKED.md" ] || cp "$state_root/BLOCKED.md" "$state/evidence/prior-BLOCKED.md"
cp "$capacity" "$state/evidence/capacity-projection.json"
cp "$runtime_dir/SUMMARY.json" "$state/evidence/runtime-summary.json"
cp "$runtime_dir/lifecycle-audit.json" "$state/evidence/runtime-lifecycle-audit.json"
cp "$real_dir/layer-verification.json" "$state/evidence/real-layer-verification.json"
cp "$real_dir/encoder-window-kernel-audit.json" "$state/evidence/real-layer-kernel-audit.json"
ssh -F zima-ssh-config mj-spark-1 \
    "bash '$project/scripts/verify_flash_generation.sh' '$state_root/bulk/$run_stamp/pre-stop-generation.json'" \
    > "$state/evidence/pre-stop-generation-summary.json"
scp -q -F zima-ssh-config \
    "mj-spark-1:$state_root/bulk/$run_stamp/pre-stop-generation.json" \
    "$state/evidence/pre-stop-generation.json"
h3_http=$(curl --connect-timeout 3 --max-time 10 -sS -o /dev/null -w '%{http_code}' http://192.168.0.234:8188/system_stats)
[ "$h3_http" = 200 ]

baseline_tmp="$state/service-baseline.json.tmp"
jq -n '{schema:"glm53-full-exl3-tp3.bulk-service-baseline.v1",containers:{}}' > "$baseline_tmp"
for index in 0 1 2; do
    node=${nodes[$index]}
    service=${services[$index]}
    values=$(ssh -F zima-ssh-config "$node" \
        "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' '$service'")
    read -r running oom restart_count <<< "$values"
    [ "$running" = true ] && [ "$oom" = false ]
    jq --arg key "$node:$service" --argjson restart "$restart_count" \
        '.containers[$key]={running:true,oom_killed:false,restart_count:$restart}' \
        "$baseline_tmp" > "$baseline_tmp.next"
    mv "$baseline_tmp.next" "$baseline_tmp"
done
mv "$baseline_tmp" "$state/service-baseline.json"

# Freeze the exact project sent to all nodes before changing service state.
for node in "${nodes[@]}"; do
    rsync -a --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude outputs/ --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" \
        "$node:$project/"
done
transition_start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
handoff=0
stop_experiment_on_failed_start() {
    code=$?
    trap - EXIT
    if (( code != 0 && handoff == 0 )); then
        for node in "${nodes[@]}"; do
            ssh -F zima-ssh-config "$node" \
                "for name in \$(docker ps --format '{{.Names}}' | awk '/^glm53-tp3-k3-uneven-L[0-9][0-9][0-9]$/ {print}'); do docker stop --timeout 30 \"\$name\" >/dev/null 2>&1 || true; done" &
        done
        wait || true
        jq -n --argjson exit_code "$code" --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
            '{schema:"glm53-full-exl3-tp3.bulk-start-failure.v2",failed_at:$at,
              exit_code:$exit_code,experimental_workers_stopped:true,
              automatic_service_restore_attempted:false,
              service_policy:"leave_flash_and_h3_stopped"}' \
            > "$state/START_FAILURE.json.tmp"
        mv "$state/START_FAILURE.json.tmp" "$state/START_FAILURE.json"
    fi
    exit "$code"
}
trap stop_experiment_on_failed_start EXIT
for index in 0 1 2; do
    ssh -F zima-ssh-config "${nodes[$index]}" \
        "docker stop --time 60 '${services[$index]}' >/dev/null" &
done
wait

stopped_ok=1
for index in 0 1 2; do
    node=${nodes[$index]}
    service=${services[$index]}
    running=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '$service'")
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    [ "$running" = false ] || stopped_ok=0
    (( available >= 12 * 1024 * 1024 * 1024 )) || stopped_ok=0
done
if (( stopped_ok == 0 )); then
    echo "bulk preflight could not stop services with 12 GiB reserve" >&2
    exit 1
fi

bulk_start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$bulk_start" > "$state/bulk-window-start.txt"
mkdir "$publish_root"
jq -n --arg run_stamp "$run_stamp" --arg rollback_stamp "$rollback_stamp" \
    --arg publish_root "$publish_root" --arg capacity "$capacity" \
    --arg real_dir "$real_dir" --arg runtime_dir "$runtime_dir" \
    --arg manager_session "$manager_session" --argjson qualification_seconds "$qualification_seconds" \
    '{schema:"glm53-full-exl3-tp3.bulk-run-config.v1",geometry_id:"rotating-uneven-768-640-640-v1",
      run_stamp:$run_stamp,rollback_stamp:$rollback_stamp,publish_root:$publish_root,
      capacity_projection:$capacity,real_qualification:$real_dir,runtime_gate:$runtime_dir,
      manager_session:$manager_session,qualification_seconds:$qualification_seconds}' \
    > "$state/RUN_CONFIG.json"

manager_command=$(printf '%q ' bash "$project/scripts/run_bulk_manager_guarded.sh" \
    "$run_stamp" "$rollback_stamp" "$publish_root" "$qualification_seconds")
tmux new-session -d -s "$manager_session" \
    "$manager_command >> '$state/manager.log' 2>&1"
for _ in $(seq 1 12); do
    [ -s "$state/STATUS.json" ] && break
    tmux has-session -t "$manager_session" 2>/dev/null || break
    sleep 5
done
if [ ! -s "$state/STATUS.json" ] || ! tmux has-session -t "$manager_session" 2>/dev/null; then
    echo "bulk manager failed to establish durable status" >&2
    exit 1
fi
handoff=1
trap - EXIT
printf '%s\n' "$run_stamp"
