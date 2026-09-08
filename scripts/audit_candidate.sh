#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
label=${2:-audit}
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
candidate="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate"
audit_stamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$candidate/audits/$label-$audit_stamp"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
mkdir -p "$output/logs" "$output/ranks"
started=$(cat "$candidate/started-at.txt")
ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
failed=0
for rank in 0 1 2; do
    node=${nodes[$rank]}
    ssh -F zima-ssh-config "$node" "docker logs 'glm53-k3-candidate-rank$rank'" \
        > "$output/logs/rank-$rank.log" 2>&1 || failed=1
    ssh -F zima-ssh-config "$node" "docker inspect 'glm53-k3-candidate-rank$rank'" \
        > "$output/ranks/rank-$rank.inspect.json" 2> "$output/ranks/rank-$rank.inspect.err" || failed=1
    row=$(ssh -F zima-ssh-config "$node" \
        "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.State.Restarting}} {{.RestartCount}}' 'glm53-k3-candidate-rank$rank' 2>/dev/null || echo 'false true true -1'")
    read -r running oom restarting restart_count <<< "$row"
    [[ "$running" == true && "$oom" == false && "$restarting" == false && "$restart_count" == 0 ]] || failed=1
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 )) || failed=1
    printf '%s\n' "$available" > "$output/ranks/rank-$rank.mem-available-bytes"
    ssh -F zima-ssh-config "$node" \
        "nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used --format=csv" \
        > "$output/ranks/rank-$rank.gpu.csv" || failed=1
    remote_watchdog="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/rank-$rank/watchdog"
    scp -q -F zima-ssh-config "$node:$remote_watchdog/metrics.csv" "$output/ranks/rank-$rank.metrics.csv" || failed=1
    if scp -q -F zima-ssh-config "$node:$remote_watchdog/STOP.json" "$output/ranks/rank-$rank.watchdog-STOP.json" 2>/dev/null; then
        failed=1
    fi
done
python3 scripts/verify_candidate_logs.py --logs "$output/logs" --output "$output/log-verification.json" || failed=1
python3 scripts/kernel_audit.py --since "$started" --until "$ended" --output "$output/kernel-audit.json" || failed=1
health=$(curl --connect-timeout 2 --max-time 10 -sS -o "$output/health.body" -w '%{http_code}' http://192.168.0.238:8893/health || true)
[[ "$health" == 200 ]] || failed=1
jq -n --arg started "$started" --arg ended "$ended" --arg label "$label" --arg health "$health" --argjson failed "$failed" \
    '{schema:"glm53-full-exl3-tp3.candidate-audit.v1",label:$label,started_at:$started,ended_at:$ended,health_http:($health|tonumber),passed:($failed==0)}' \
    > "$output/SUMMARY.json.tmp"
mv "$output/SUMMARY.json.tmp" "$output/SUMMARY.json"
if (( failed )); then
    bash scripts/stop_candidate.sh "$stamp" || true
    exit 2
fi
printf '%s\n' "$output"
