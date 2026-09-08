#!/usr/bin/env bash
set -euo pipefail

stamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
rollback_stamp=${2:?rollback inventory stamp required}
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification
log=/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/synthetic-uneven-codec-$stamp.log
container=glm53-tp3-k3-synthetic-uneven
protected=minimax-h3-comfy
rollback="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/rollback/$rollback_stamp/$(hostname -s)"
runtime=/home/mj-kang/Dev/experiment/glm53-exl3-2spark-c190db1/overlay/exl3.py
mkdir -p "$state" "$(dirname "$log")"

if [ ! -s "$rollback/containers/$protected.inspect.json" ] || [ ! -x "$rollback/rollback-commands.sh" ]; then
    echo "missing rollback inventory for $protected" >&2
    exit 2
fi
if [ ! -f "$runtime" ]; then echo "sealed EXL3 runtime overlay missing" >&2; exit 2; fi
if [ "$(sha256sum "$runtime" | awk '{print $1}')" != c9e765e13747cde82840c7af44945b7f06a1dee176df472dcebd1d858f9a5843 ]; then
    echo "EXL3 runtime overlay hash mismatch" >&2
    exit 2
fi

was_running=$(docker inspect -f '{{.State.Running}}' "$protected")
restore() {
    docker rm "$container" >/dev/null 2>&1 || true
    if [ "$was_running" = true ] && [ "$(docker inspect -f '{{.State.Running}}' "$protected" 2>/dev/null || true)" != true ]; then
        docker start "$protected" >/dev/null
    fi
}
trap restore EXIT
if [ "$was_running" = true ]; then docker stop --time 60 "$protected" >/dev/null; fi
for _ in $(seq 1 30); do
    available=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
    if (( available >= 12 * 1024 * 1024 * 1024 )); then break; fi
    sleep 2
done
if (( available < 12 * 1024 * 1024 * 1024 )); then echo "12 GiB reserve not recovered" >&2; exit 2; fi

docker rm "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --gpus all --ipc host --shm-size 8g --network none \
    -e PYTHONPATH=/workspace:/workspace/encoder-r10 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
    -v "$project:/workspace:ro" \
    -v "$runtime:/runtime-exl3.py:ro" \
    -v "$state:/state" \
    --entrypoint python3 glm53-exl3:e2-c190db1 /workspace/tests/synthetic_codec.py \
    --numeric-core /workspace/encoder-r10/lineage/encode_tr3_v31.py \
    --numeric-core-sha256 e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032 \
    --extension /usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so \
    --extension-sha256 7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2 \
    --runtime-exl3 /runtime-exl3.py --output "/state/synthetic-uneven-codec-$stamp.json" >/dev/null
docker logs -f "$container" > "$log" 2>&1 &
log_pid=$!
set +e
python3 "$project/scripts/watchdog.py" --container "$container" --state-dir "$state/synthetic-uneven-codec-$stamp-watchdog"
watchdog_status=$?
exit_code=$(docker wait "$container")
wait "$log_pid"
set -e
docker inspect "$container" > "$state/synthetic-uneven-codec-$stamp-container.json"
if (( watchdog_status != 0 )) || [ "$exit_code" != 0 ]; then
    echo "synthetic codec gate failed: watchdog=$watchdog_status container=$exit_code" >&2
    exit 1
fi
test -s "$state/synthetic-uneven-codec-$stamp.json"
