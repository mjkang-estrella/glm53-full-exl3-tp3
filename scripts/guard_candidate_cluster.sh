#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
attempt=${2:?candidate attempt required}
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || { echo "invalid attempt label" >&2; exit 2; }

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt"
guard="$state/guard"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
prefix="glm53-k3-cand-$attempt"
reserve=$((12 * 1024 * 1024 * 1024))
started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)

test -s "$state/READY.json"
jq -e '.passed == true' "$state/READY.json" >/dev/null
test ! -e "$guard/STOP.json"
mkdir -p "$guard"

stop_group() {
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        ssh -F zima-ssh-config "$node" \
            "docker stop --time 60 '$prefix-rank$rank' >/dev/null 2>&1 || true; \
             tmux kill-session -t '$prefix-watchdog-$rank' 2>/dev/null || true; \
             tmux kill-session -t '$prefix-log-$rank' 2>/dev/null || true" &
    done
    wait || true
}

health_failures=0
while :; do
    now=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    reason=
    rows=()
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        cname="$prefix-rank$rank"
        raw=$(ssh -F zima-ssh-config -o BatchMode=yes -o ConnectTimeout=5 "$node" \
            "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}' '$cname' 2>/dev/null || echo 'ssh_or_container_error true 125'; \
             awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo; \
             test -e '$state/rank-$rank/watchdog/STOP.json' && echo watchdog_stop || echo watchdog_clean" \
            2>/dev/null || printf 'ssh_or_container_error true 125\n0\nwatchdog_stop\n')
        status=$(sed -n '1p' <<<"$raw")
        available=$(sed -n '2p' <<<"$raw")
        watchdog=$(sed -n '3p' <<<"$raw")
        read -r running oom exit_code <<<"$status"
        [[ "$available" =~ ^[0-9]+$ ]] || available=0
        rows+=("$(jq -n --argjson rank "$rank" --arg node "$node" \
            --arg running "$running" --arg oom "$oom" --arg exit "$exit_code" \
            --arg watchdog "$watchdog" --argjson available "$available" \
            '{rank:$rank,node:$node,running:$running,oom_killed:$oom,exit_code:$exit,watchdog:$watchdog,available_bytes:$available}')")
        if [[ "$watchdog" != watchdog_clean ]]; then
            reason="rank_${rank}_watchdog_stop"
        elif [[ "$running" != true || "$oom" != false ]]; then
            reason="rank_${rank}_worker_not_healthy"
        elif (( available < reserve )); then
            reason="rank_${rank}_reserve_breach"
        fi
    done
    health=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' \
        http://192.168.0.238:8893/health 2>/dev/null || true)
    if [[ "$health" == 200 ]]; then
        health_failures=0
    else
        health_failures=$((health_failures + 1))
        (( health_failures < 3 )) || reason="endpoint_health_failed_three_times"
    fi
    ranks_json=$(printf '%s\n' "${rows[@]}" | jq -s .)
    jq -n --arg schema "glm53-full-exl3-tp3.candidate-guard-heartbeat.v1" \
        --arg started "$started" --arg updated "$now" --arg health "${health:-0}" \
        --argjson health_failures "$health_failures" --argjson ranks "$ranks_json" \
        '{schema:$schema,started_at:$started,updated_at:$updated,health_http:($health|tonumber),consecutive_health_failures:$health_failures,ranks:$ranks}' \
        > "$guard/HEARTBEAT.json.tmp"
    mv "$guard/HEARTBEAT.json.tmp" "$guard/HEARTBEAT.json"
    if [[ -n "$reason" ]]; then
        jq -n --arg reason "$reason" --arg at "$now" --argjson heartbeat "$(cat "$guard/HEARTBEAT.json")" \
            '{schema:"glm53-full-exl3-tp3.candidate-guard-stop.v1",reason:$reason,detected_at:$at,heartbeat:$heartbeat,automatic_service_restore_attempted:false}' \
            > "$guard/STOP.json.tmp"
        mv "$guard/STOP.json.tmp" "$guard/STOP.json"
        stop_group
        python3 "$project/scripts/kernel_audit.py" --since "$started" --until "$now" \
            --output "$guard/kernel-audit.json" || true
        exit 2
    fi
    sleep 5
done
