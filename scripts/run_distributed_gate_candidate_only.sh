#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
stamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
nccl_channels=${2:-4}
nccl_buffsize=${3:-1048576}
[[ "$nccl_channels" =~ ^[0-9]+$ ]] && (( nccl_channels >= 1 && nccl_channels <= 64 )) || { echo "invalid NCCL channel count" >&2; exit 2; }
[[ "$nccl_buffsize" =~ ^[0-9]+$ ]] && (( nccl_buffsize >= 131072 && nccl_buffsize <= 16777216 )) || { echo "invalid NCCL buffer size" >&2; exit 2; }
master=192.168.0.238
port=29613
central="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260906T034145Z/runtime-regression-$stamp"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
layers=(3 4 5)
mkdir -p "$central/rotations"

stop_gate() {
    for rank in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "docker stop --time 30 'glm53-tp3-runtime-gate-rank$rank' >/dev/null 2>&1 || true" &
    done
    wait || true
}
trap stop_gate EXIT

for rank in 0 1 2; do
    node=${nodes[$rank]}
    running=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '${services[$rank]}'")
    [[ "$running" == false ]]
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 ))
    rsync -a --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" \
        "$node:/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/"
done

started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$started" > "$central/started-at.txt"
python3 scripts/kernel_audit.py --since "$started" --output "$central/preflight-kernel-audit.json"
failed=0
for layer in "${layers[@]}"; do
    rotation="$central/rotations/layer-$(printf '%03d' "$layer")"
    mkdir -p "$rotation/ranks"
    for rank in 0 1 2; do
        bash_port=$(( port + layer - 3 ))
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "NCCL_CHANNELS='$nccl_channels' NCCL_BUFFER_BYTES='$nccl_buffsize' bash /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/scripts/start_runtime_gate_rank.sh '$rank' '$master' '$bash_port' '$stamp' '$layer'"
    done
    deadline=$(( $(date +%s) + 1800 ))
    while :; do
        active=0
        for rank in 0 1 2; do
            running=$(ssh -F zima-ssh-config "${nodes[$rank]}" \
                "docker inspect -f '{{.State.Running}}' 'glm53-tp3-runtime-gate-rank$rank' 2>/dev/null || echo false")
            [[ "$running" == true ]] && active=$((active + 1))
        done
        (( active == 0 )) && break
        (( $(date +%s) < deadline )) || { failed=1; stop_gate; break; }
        sleep 5
    done
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        code=$(ssh -F zima-ssh-config "$node" \
            "docker inspect -f '{{.State.ExitCode}}' 'glm53-tp3-runtime-gate-rank$rank' 2>/dev/null || echo 125")
        printf '%s\n' "$code" > "$rotation/ranks/rank-$rank.exit-code"
        scp -q -F zima-ssh-config \
            "$node:/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/runtime-gate-$stamp-layer-$(printf '%03d' "$layer")-rank$rank.log" \
            "$rotation/ranks/rank-$rank.log" || true
        scp -q -F zima-ssh-config \
            "$node:/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp/rotations/layer-$(printf '%03d' "$layer")/rank-$rank.json" \
            "$rotation/ranks/rank-$rank.json" || true
        scp -q -F zima-ssh-config \
            "$node:/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp/rotations/layer-$(printf '%03d' "$layer")/watchdog-rank$rank/metrics.csv" \
            "$rotation/ranks/rank-$rank.metrics.csv" || true
        [[ "$code" == 0 ]] || failed=1
        grep -Fq 'NCCL INFO Using network IB' "$rotation/ranks/rank-$rank.log" || failed=1
    done
    if ls "$rotation"/ranks/rank-*.json >/dev/null 2>&1; then
        jq -s --argjson layer "$layer" \
            '{schema:"glm53-full-exl3-tp3.distributed-runtime-rotation-summary.v2",layer:$layer,passed:(length==3 and all(.[];.passed==true)),ranks:.}' \
            "$rotation"/ranks/rank-*.json > "$rotation/SUMMARY.json"
        jq -e '.passed == true' "$rotation/SUMMARY.json" >/dev/null || failed=1
    else
        failed=1
    fi
    stop_gate
done
ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$ended" > "$central/ended-at.txt"
python3 scripts/kernel_audit.py --since "$started" --until "$ended" \
    --output "$central/kernel-audit.json" || failed=1
if (( failed == 0 )); then
    jq -s '{schema:"glm53-full-exl3-tp3.distributed-runtime-summary.v2",geometry_id:"rotating-uneven-768-640-640-v1",passed:(length==3 and all(.[];.passed==true)),rotations:.}' \
        "$central"/rotations/layer-*/SUMMARY.json > "$central/SUMMARY.json"
    jq -e '.passed == true' "$central/SUMMARY.json" >/dev/null
    date -u +%Y-%m-%dT%H:%M:%S.%3NZ > "$central/PASSED"
else
    jq -n --arg at "$ended" '{schema:"glm53-full-exl3-tp3.runtime-regression-failure.v1",failed_at:$at,automatic_service_restore_attempted:false}' \
        > "$central/FAILED.json"
    exit 2
fi
trap - EXIT
