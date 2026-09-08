#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
stamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)-lazy-gate-v1}
execution=${2:-chunked_fused}
cache_experts=${3:-8}
rank_pack_basename=${4:-}
cuda_allocator_conf=${5:-expandable_segments:True}
arena_block_slots=${6:-16}
uva_resident=${GLM53_LAZY_K3_UVA:-0}
[[ "$execution" == chunked_fused || "$execution" == shared_layer_fused || "$execution" == resident_fused || "$execution" == resident_uva ]] || { echo "invalid gate execution" >&2; exit 2; }
[[ "$cuda_allocator_conf" == "expandable_segments:True" || "$cuda_allocator_conf" == "expandable_segments:False" || "$cuda_allocator_conf" == "backend:cudaMallocAsync" ]] || { echo "unsupported CUDA allocator configuration" >&2; exit 2; }
[[ "$arena_block_slots" =~ ^[0-9]+$ ]] && (( arena_block_slots >= 1 && arena_block_slots <= 64 )) || { echo "arena block slots must be 1..64" >&2; exit 2; }
master=192.168.0.238
port=29657
central="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260906T034145Z/candidate/memory-designs/$stamp"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
mkdir -p "$central/rotations"

stop_gate() {
    for rank in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "docker stop --time 30 'glm53-lazy-runtime-gate-rank$rank' >/dev/null 2>&1 || true; \
             tmux kill-session -t 'glm53-lazy-gate-watchdog-$rank' 2>/dev/null || true; \
             tmux kill-session -t 'glm53-lazy-gate-log-$rank' 2>/dev/null || true" &
    done
    wait || true
}
trap stop_gate EXIT

started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$started" > "$central/started-at.txt"
for rank in 0 1 2; do
    node=${nodes[$rank]}
    [[ "$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '${services[$rank]}'")" == false ]]
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 ))
    ssh -F zima-ssh-config "$node" "! docker ps --format '{{.Names}}' | grep -Eq '^(glm53-k3-|glm53-tp3-k3-uneven|glm53-lazy-runtime-gate)'; test -s /home/mj-kang/Dev/models/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z/ASSEMBLY_COMPLETE.json"
    rsync -a --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" "$node:/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/"
done
python3 scripts/kernel_audit.py --since "$started" --output "$central/preflight-kernel-audit.json"

failed=0
for layer in 3 4 5; do
    rotation="$central/rotations/layer-$(printf '%03d' "$layer")"
    mkdir -p "$rotation/ranks"
    for rank in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "bash /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/scripts/start_lazy_runtime_gate_rank.sh '$rank' '$master' '$((port + layer - 3))' '$stamp' '$layer' '$execution' '$cache_experts' '$rank_pack_basename' '$cuda_allocator_conf' '$arena_block_slots' '$uva_resident'" &
    done
    wait
    deadline=$(( $(date +%s) + 900 ))
    while :; do
        active=0
        for rank in 0 1 2; do
            row=$(ssh -F zima-ssh-config "${nodes[$rank]}" "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}' 'glm53-lazy-runtime-gate-rank$rank'")
            read -r running oom exit_code <<< "$row"
            [[ "$oom" == false ]] || failed=1
            [[ "$running" == true ]] && active=$((active + 1))
            [[ "$running" == true || "$exit_code" == 0 ]] || failed=1
            ssh -F zima-ssh-config "${nodes[$rank]}" "test ! -e '$rotation/watchdog-rank$rank/STOP.json'" || failed=1
        done
        (( active == 0 )) && break
        (( failed == 0 && $(date +%s) < deadline )) || { failed=1; stop_gate; break; }
        sleep 3
    done
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        code=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.ExitCode}}' 'glm53-lazy-runtime-gate-rank$rank' 2>/dev/null || echo 125")
        printf '%s\n' "$code" > "$rotation/ranks/rank-$rank.exit-code"
        scp -q -F zima-ssh-config "$node:$rotation/rank-$rank.json" "$rotation/ranks/rank-$rank.json" || true
        scp -q -F zima-ssh-config "$node:$rotation/watchdog-rank$rank/metrics.csv" "$rotation/ranks/rank-$rank.metrics.csv" || true
        scp -q -F zima-ssh-config "$node:/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-20260906T034145Z/memory-designs/$stamp-layer-$(printf '%03d' "$layer")-rank$rank.log" "$rotation/ranks/rank-$rank.log" || true
        [[ "$code" == 0 ]] || failed=1
        test -s "$rotation/ranks/rank-$rank.json" || failed=1
        grep -Fq 'NCCL INFO Using network IB' "$rotation/ranks/rank-$rank.log" || failed=1
    done
    if (( failed == 0 )); then
        jq -s --argjson layer "$layer" '{schema:"glm53-full-exl3-tp3.lazy-runtime-rotation-summary.v1",layer:$layer,passed:(length==3 and all(.[];.passed==true)),ranks:.}' "$rotation"/ranks/rank-*.json > "$rotation/SUMMARY.json"
        jq -e '.passed == true' "$rotation/SUMMARY.json" >/dev/null || failed=1
    fi
    stop_gate
    (( failed == 0 )) || break
done
ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$ended" > "$central/ended-at.txt"
python3 scripts/kernel_audit.py --since "$started" --until "$ended" --output "$central/kernel-audit.json" || failed=1
if (( failed == 0 )); then
    jq -s '{schema:"glm53-full-exl3-tp3.lazy-runtime-summary.v1",passed:(length==3 and all(.[];.passed==true)),rotations:.}' "$central"/rotations/layer-*/SUMMARY.json > "$central/SUMMARY.json"
    jq -e '.passed == true' "$central/SUMMARY.json" >/dev/null
    touch "$central/PASSED"
else
    jq -n --arg at "$ended" '{schema:"glm53-full-exl3-tp3.lazy-runtime-failure.v1",failed_at:$at,candidates_stopped:true,automatic_service_restore_attempted:false}' > "$central/FAILED.json.tmp"
    mv "$central/FAILED.json.tmp" "$central/FAILED.json"
    exit 2
fi
trap - EXIT
