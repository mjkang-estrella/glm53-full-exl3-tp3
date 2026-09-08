#!/usr/bin/env bash
set -euo pipefail

rank=${1:?rank required}
master=${2:?master address required}
port=${3:?master port required}
stamp=${4:?stamp required}
layer=${5:?rotation layer required}
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
runtime=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/runtime/exl3.py
output="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/runtime-gate-$stamp/rotations/layer-$(printf '%03d' "$layer")"
log="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/runtime-gate-$stamp-layer-$(printf '%03d' "$layer")-rank$rank.log"
watchdog="$output/watchdog-rank$rank"
container="glm53-tp3-runtime-gate-rank$rank"
nccl_channels=${NCCL_CHANNELS:-4}
nccl_buffsize=${NCCL_BUFFER_BYTES:-1048576}
[[ "$nccl_channels" =~ ^[0-9]+$ ]] && (( nccl_channels >= 1 && nccl_channels <= 64 )) || { echo "invalid NCCL channel count" >&2; exit 2; }
[[ "$nccl_buffsize" =~ ^[0-9]+$ ]] && (( nccl_buffsize >= 131072 && nccl_buffsize <= 16777216 )) || { echo "invalid NCCL buffer size" >&2; exit 2; }

if [ "$(sha256sum "$runtime" | awk '{print $1}')" != c9e765e13747cde82840c7af44945b7f06a1dee176df472dcebd1d858f9a5843 ]; then
    echo "sealed EXL3 runtime hash differs" >&2
    exit 2
fi
if docker ps --format '{{.Names}}' | grep -Eq '^(glm53-exl3-(head|worker)|minimax-h3-comfy)$'; then
    echo "protected GPU service remains active" >&2
    exit 2
fi
available=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
if (( available < 12 * 1024 * 1024 * 1024 )); then echo "host reserve below 12 GiB" >&2; exit 2; fi
mkdir -p "$output" "$(dirname "$log")" "$watchdog"
docker rm "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --gpus all --network host --ipc host --shm-size 8g \
    --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
    -e PYTHONPATH=/workspace:/workspace/encoder-r10 \
    -e VLLM_GLM53_EXL3_TP3_GEOMETRY_ID=rotating-uneven-768-640-640-v1 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
    -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
    -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_SOCKET_IFNAME=enP7s7 -e NCCL_DEBUG=INFO \
    -e NCCL_MIN_NCHANNELS="$nccl_channels" -e NCCL_MAX_NCHANNELS="$nccl_channels" -e NCCL_BUFFSIZE="$nccl_buffsize" \
    -v "$project:/workspace:ro" -v "$runtime:/runtime-exl3.py:ro" -v "$output:/output" \
    --entrypoint python3 glm53-exl3:e2-c190db1 /workspace/tests/distributed_runtime_gate.py \
    --rank "$rank" --layer "$layer" --world-size 3 --master-addr "$master" --master-port "$port" \
    --runtime-exl3 /runtime-exl3.py --tp3-patch /workspace/runtime/tp3_rank_slice_patch.py \
    --numeric-core /workspace/encoder-r10/lineage/encode_tr3_v31.py \
    --numeric-core-sha256 e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032 \
    --extension /usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so \
    --extension-sha256 7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2 \
    --output-dir /output >/dev/null
tmux kill-session -t "glm53-runtime-watchdog-$rank" 2>/dev/null || true
tmux new-session -d -s "glm53-runtime-watchdog-$rank" \
    "python3 '$project/scripts/watchdog.py' --container '$container' --state-dir '$watchdog' > '$watchdog/watchdog.log' 2>&1"
tmux kill-session -t "glm53-runtime-log-$rank" 2>/dev/null || true
tmux new-session -d -s "glm53-runtime-log-$rank" \
    "docker logs -f '$container' > '$log' 2>&1"
printf '%s\n' "$container"
