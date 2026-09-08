#!/usr/bin/env bash
set -euo pipefail

rank=${1:?rank required}
master=${2:?master address required}
port=${3:?master port required}
stamp=${4:?stamp required}
layer=${5:?rotation layer required}
execution=${6:-chunked_fused}
cache_experts=${7:-8}
rank_pack_basename=${8:-}
cuda_allocator_conf=${9:-expandable_segments:True}
arena_block_slots=${10:-16}
uva_resident=${11:-0}
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
runtime=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/runtime/exl3.py
model=/home/mj-kang/Dev/models/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z
output="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260906T034145Z/candidate/memory-designs/$stamp/rotations/layer-$(printf '%03d' "$layer")"
log="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-20260906T034145Z/memory-designs/$stamp-layer-$(printf '%03d' "$layer")-rank$rank.log"
watchdog="$output/watchdog-rank$rank"
container="glm53-lazy-runtime-gate-rank$rank"

[[ "$cuda_allocator_conf" == "expandable_segments:True" || "$cuda_allocator_conf" == "expandable_segments:False" || "$cuda_allocator_conf" == "backend:cudaMallocAsync" ]] || {
    echo "unsupported CUDA allocator configuration" >&2
    exit 2
}
[[ "$arena_block_slots" =~ ^[0-9]+$ ]] && (( arena_block_slots >= 1 && arena_block_slots <= 64 )) || {
    echo "arena block slots must be 1..64" >&2
    exit 2
}
[[ "$uva_resident" == 0 || "$uva_resident" == 1 ]] || { echo "invalid UVA resident flag" >&2; exit 2; }

[[ "$(sha256sum "$runtime" | awk '{print $1}')" == c9e765e13747cde82840c7af44945b7f06a1dee176df472dcebd1d858f9a5843 ]]
test -s "$model/ASSEMBLY_COMPLETE.json"
if docker ps --format '{{.Names}}' | grep -Eq '^(glm53-exl3-(head|worker)|minimax-h3-comfy)$'; then
    echo "protected GPU service remains active" >&2
    exit 2
fi
available=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
(( available >= 12 * 1024 * 1024 * 1024 ))
mkdir -p "$output" "$(dirname "$log")" "$watchdog"
rank_pack_args=()
if [[ -n "$rank_pack_basename" ]]; then
    rank_pack_root="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename/rank-$rank"
    jq -e '.passed == true and .rank == '"$rank"' and .layers == 76' "$rank_pack_root/COMPLETE.json" >/dev/null
    rank_pack_args+=(
        -e GLM53_LAZY_K3_PACKED_ROOT=/rank-packs
        -v "$rank_pack_root:/rank-packs:ro"
    )
fi
docker rm "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --gpus all --network host --ipc host --shm-size 8g \
    --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONPATH=/workspace -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e PYTORCH_CUDA_ALLOC_CONF="$cuda_allocator_conf" \
    -e GLM53_LAZY_K3_EXECUTION="$execution" \
    -e GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER="$cache_experts" \
    -e GLM53_LAZY_K3_ARENA_BLOCK_SLOTS="$arena_block_slots" \
    -e GLM53_LAZY_K3_UVA="$uva_resident" \
    "${rank_pack_args[@]}" \
    -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
    -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
    -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_SOCKET_IFNAME=enP7s7 -e NCCL_DEBUG=INFO \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=1 -e NCCL_BUFFSIZE=131072 \
    -v "$project:/workspace:ro" -v "$runtime:/runtime-exl3.py:ro" \
    -v "$model:/model:ro" -v "$output:/output" \
    --entrypoint python3 glm53-exl3:e2-c190db1 /workspace/tests/lazy_distributed_runtime_gate.py \
    --rank "$rank" --layer "$layer" --master-addr "$master" --master-port "$port" \
    --runtime-exl3 /runtime-exl3.py --lazy-patch /workspace/runtime/lazy_k3_patch.py \
    --model-dir /model --output-dir /output >/dev/null
tmux kill-session -t "glm53-lazy-gate-watchdog-$rank" 2>/dev/null || true
tmux new-session -d -s "glm53-lazy-gate-watchdog-$rank" \
    "python3 '$project/scripts/watchdog.py' --container '$container' --state-dir '$watchdog' --reserve-gib 12 --interval 2 > '$watchdog/watchdog.log' 2>&1"
tmux kill-session -t "glm53-lazy-gate-log-$rank" 2>/dev/null || true
tmux new-session -d -s "glm53-lazy-gate-log-$rank" "docker logs -f '$container' > '$log' 2>&1"
