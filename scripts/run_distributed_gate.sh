#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
stamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
rollback_stamp=${2:?rollback inventory stamp required}
master=192.168.0.238
port=29603
central="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp"
mkdir -p "$central/rotations" "$central/timeline"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
rotation_layers=(3 4 5)
utc_now() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }

for index in 0 1 2; do
    node=${nodes[$index]}
    service=${services[$index]}
    ssh -F zima-ssh-config "$node" "test -s '/home/mj-kang/Dev/state/glm53-full-exl3-tp3/rollback/$rollback_stamp/$node/containers/$service.inspect.json'"
    rsync -a --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" "$node:/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/"
    ssh -F zima-ssh-config "$node" 'mkdir -p /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/runtime'
    rsync -a /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/runtime/exl3.py -e "ssh -F zima-ssh-config" "$node:/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/runtime/exl3.py"
done

restored=0
restore_services() {
    if (( restored )); then return; fi
    restored=1
    # Worker first, then head; these are the exact preserved containers.
    ssh -F zima-ssh-config mj-spark-2 'docker start glm53-exl3-worker >/dev/null 2>&1 || true' &
    p2=$!
    ssh -F zima-ssh-config mj-spark-3 'docker start minimax-h3-comfy >/dev/null 2>&1 || true' &
    p3=$!
    wait "$p2" "$p3" || true
    ssh -F zima-ssh-config mj-spark-1 'docker start glm53-exl3-head >/dev/null 2>&1 || true' || true
}
stop_gate_containers() {
    for index in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$index]}" "docker stop --time 30 glm53-tp3-runtime-gate-rank$index >/dev/null 2>&1 || true" &
    done
    wait || true
}
trap 'stop_gate_containers; restore_services' EXIT

head_restart_before=$(ssh -F zima-ssh-config mj-spark-1 \
    "docker inspect -f '{{.RestartCount}}' glm53-exl3-head")
worker_restart_before=$(ssh -F zima-ssh-config mj-spark-2 \
    "docker inspect -f '{{.RestartCount}}' glm53-exl3-worker")
h3_restart_before=$(ssh -F zima-ssh-config mj-spark-3 \
    "docker inspect -f '{{.RestartCount}}' minimax-h3-comfy")

for index in 0 1 2; do
    ssh -F zima-ssh-config "${nodes[$index]}" "docker stop --time 60 '${services[$index]}' >/dev/null" &
done
wait
for index in 0 1 2; do
    node=${nodes[$index]}
    service=${services[$index]}
    state=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '$service'")
    [ "$state" = false ] || { echo "$node protected service still running" >&2; exit 2; }
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 )) || { echo "$node reserve below 12 GiB" >&2; exit 2; }
done

candidate_start=$(utc_now)
printf '%s\n' "$candidate_start" > "$central/timeline/candidate-runtime-start.txt"

failed=0
for layer in "${rotation_layers[@]}"; do
    rotation="$central/rotations/layer-$(printf '%03d' "$layer")"
    mkdir -p "$rotation/ranks"
    for index in 0 1 2; do
        rotation_port=$((port + layer - 3))
        ssh -F zima-ssh-config "${nodes[$index]}" \
            "bash /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/scripts/start_runtime_gate_rank.sh '$index' '$master' '$rotation_port' '$stamp' '$layer'"
    done
    deadline=$(( $(date +%s) + 1800 ))
    while :; do
        active=0
        for index in 0 1 2; do
            node=${nodes[$index]}
            running=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' glm53-tp3-runtime-gate-rank$index 2>/dev/null || echo false")
            [ "$running" = true ] && active=$((active + 1))
        done
        [ "$active" -eq 0 ] && break
        if (( $(date +%s) >= deadline )); then echo "runtime layer $layer gate timeout" >&2; failed=1; stop_gate_containers; break; fi
        sleep 5
    done
    for index in 0 1 2; do
        node=${nodes[$index]}
        code=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.ExitCode}}' glm53-tp3-runtime-gate-rank$index 2>/dev/null || echo 125")
        printf '%s\n' "$code" > "$rotation/rank-$index.exit-code"
        scp -q -F zima-ssh-config "$node:/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/runtime-gate-$stamp-layer-$(printf '%03d' "$layer")-rank$index.log" "$rotation/ranks/rank-$index.log" || true
        scp -q -F zima-ssh-config "$node:/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp/rotations/layer-$(printf '%03d' "$layer")/rank-$index.json" "$rotation/ranks/rank-$index.json" || true
        scp -q -F zima-ssh-config "$node:/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp/rotations/layer-$(printf '%03d' "$layer")/watchdog-rank$index/metrics.csv" "$rotation/ranks/rank-$index.metrics.csv" || true
        scp -q -F zima-ssh-config "$node:/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp/rotations/layer-$(printf '%03d' "$layer")/watchdog-rank$index/STOP.json" "$rotation/ranks/rank-$index.watchdog-stop.json" 2>/dev/null || true
        [ "$code" = 0 ] || failed=1
        grep -Fq 'NCCL INFO Using network IB' "$rotation/ranks/rank-$index.log" || {
            printf '%s\n' "layer $layer rank $index did not bind NCCL to IB/RoCE" >&2
            failed=1
        }
        if [ -e "$rotation/ranks/rank-$index.watchdog-stop.json" ]; then
            printf '%s\n' "layer $layer rank $index watchdog stopped the runtime gate" >&2
            failed=1
        fi
    done
    if ls "$rotation"/ranks/rank-*.json >/dev/null 2>&1; then
        jq -s --argjson layer "$layer" \
            '{schema:"glm53-full-exl3-tp3.distributed-runtime-rotation-summary.v2",layer:$layer,passed:(length==3 and all(.[];.passed==true)),ranks:.}' \
            "$rotation"/ranks/rank-*.json > "$rotation/SUMMARY.json"
        jq -e '.passed == true' "$rotation/SUMMARY.json" >/dev/null || failed=1
    else
        failed=1
    fi
    stop_gate_containers
done

candidate_end=$(utc_now)
printf '%s\n' "$candidate_end" > "$central/timeline/candidate-runtime-end.txt"

stop_gate_containers
flash_restore_start=$(utc_now)
printf '%s\n' "$flash_restore_start" > "$central/timeline/flash-restore-start.txt"
restore_services
trap - EXIT

# Verify the preserved services themselves, not merely container states.
worker_death_observed=false
flash_ready=
for _ in $(seq 1 180); do
    head_code=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' http://192.168.0.238:8888/health || true)
    h3_code=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' http://192.168.0.234:8188/system_stats || true)
    head_running=$(ssh -F zima-ssh-config mj-spark-1 "docker inspect -f '{{.State.Running}}' glm53-exl3-head 2>/dev/null || echo false")
    worker=$(ssh -F zima-ssh-config mj-spark-2 "docker inspect -f '{{.State.Running}}' glm53-exl3-worker 2>/dev/null || echo false")
    if [ "$head_running" != true ] || [ "$worker" != true ]; then
        worker_death_observed=true
        break
    fi
    if [ "$head_code" = 200 ] && [ "$h3_code" = 200 ] && [ "$worker" = true ]; then
        flash_ready=$(utc_now)
        printf '%s\n' "$flash_ready" > "$central/timeline/flash-ready.txt"
        break
    fi
    sleep 5
done
printf '%s %s %s\n' "$head_code" "$worker" "$h3_code" > "$central/restored-health.txt"
if [ "$head_code" != 200 ] || [ "$h3_code" != 200 ] || [ "$worker" != true ] || [ -z "$flash_ready" ]; then failed=1; fi
if [ "$head_code" = 200 ] && [ "$worker" = true ]; then
    remote_canary="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp/restored-generation.json"
    if ssh -F zima-ssh-config mj-spark-1 \
        "bash /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/scripts/verify_flash_generation.sh '$remote_canary'" \
        > "$central/restored-generation-summary.json"; then
        scp -q -F zima-ssh-config "mj-spark-1:$remote_canary" "$central/restored-generation.json"
    else
        printf '%s\n' 'restored Flash endpoint failed a real generation' >&2
        failed=1
    fi
fi

# A ready service must stay kernel-clean through and after the real canary.
sleep 15
post_ready_end=$(utc_now)
printf '%s\n' "$post_ready_end" > "$central/timeline/post-ready-end.txt"

head_after=$(ssh -F zima-ssh-config mj-spark-1 \
    "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' glm53-exl3-head 2>/dev/null || echo 'false true -1'")
worker_after=$(ssh -F zima-ssh-config mj-spark-2 \
    "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' glm53-exl3-worker 2>/dev/null || echo 'false true -1'")
h3_after=$(ssh -F zima-ssh-config mj-spark-3 \
    "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' minimax-h3-comfy 2>/dev/null || echo 'false true -1'")
read -r head_running head_oom head_restart_after <<< "$head_after"
read -r worker_running worker_oom worker_restart_after <<< "$worker_after"
read -r h3_running h3_oom h3_restart_after <<< "$h3_after"

jq -n \
    --arg candidate_start "$candidate_start" --arg candidate_end "$candidate_end" \
    --arg flash_start "$flash_restore_start" --arg flash_ready "$flash_ready" \
    --arg post_end "$post_ready_end" \
    '{schema:"glm53-full-exl3-tp3.lifecycle-timeline.v1",
      candidate_runtime:{start:$candidate_start,end:$candidate_end},
      protected_flash_rollback:{start:$flash_start,
        ready:(if $flash_ready == "" then null else $flash_ready end),
        timeout_seconds:900},
      post_ready:{start:(if $flash_ready == "" then null else $flash_ready end),end:$post_end}}' \
    > "$central/timeline.json"

jq -n \
    --arg head_running "$head_running" --arg head_oom "$head_oom" \
    --argjson head_before "$head_restart_before" --argjson head_after "$head_restart_after" \
    --arg worker_running "$worker_running" --arg worker_oom "$worker_oom" \
    --argjson worker_before "$worker_restart_before" --argjson worker_after "$worker_restart_after" \
    --arg h3_running "$h3_running" --arg h3_oom "$h3_oom" \
    --argjson h3_before "$h3_restart_before" --argjson h3_after "$h3_restart_after" \
    --arg death "$worker_death_observed" --arg flash_http "${head_code:-0}" \
    --arg h3_http "${h3_code:-0}" \
    '{schema:"glm53-full-exl3-tp3.restored-service-state.v1",
      containers:{
        "mj-spark-1:glm53-exl3-head":{running:($head_running=="true"),oom_killed:($head_oom=="true"),restart_count_before:$head_before,restart_count_after:$head_after},
        "mj-spark-2:glm53-exl3-worker":{running:($worker_running=="true"),oom_killed:($worker_oom=="true"),restart_count_before:$worker_before,restart_count_after:$worker_after},
        "mj-spark-3:minimax-h3-comfy":{running:($h3_running=="true"),oom_killed:($h3_oom=="true"),restart_count_before:$h3_before,restart_count_after:$h3_after}},
      worker_death_observed:($death=="true"),
      health:{flash_http:($flash_http|tonumber),h3_http:($h3_http|tonumber)}}' \
    > "$central/service-state.json"

if [ ! -s "$central/restored-generation.json" ]; then
    jq -n '{schema:"glm53-full-exl3-tp3.missing-generation.v1",choices:[]}' \
        > "$central/restored-generation.json"
fi
if ! python3 scripts/kernel_audit.py \
    --timeline "$central/timeline.json" \
    --service-state "$central/service-state.json" \
    --generation "$central/restored-generation.json" \
    --output "$central/lifecycle-audit.json"; then
    failed=1
fi

if (( failed == 0 )); then
    jq -s '{schema:"glm53-full-exl3-tp3.distributed-runtime-summary.v2",geometry_id:"rotating-uneven-768-640-640-v1",passed:(length==3 and all(.[];.passed==true)),rotations:.}' "$central"/rotations/layer-*/SUMMARY.json > "$central/SUMMARY.json"
    jq -e '.passed == true' "$central/SUMMARY.json" >/dev/null
    date -u +%FT%TZ > "$central/PASSED"
else
    date -u +%FT%TZ > "$central/FAILED"
    exit 1
fi
