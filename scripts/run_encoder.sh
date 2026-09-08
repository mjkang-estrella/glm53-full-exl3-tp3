#!/usr/bin/env bash
set -euo pipefail

layer=${1:?layer required}
experts=${2:-0-255}
stamp=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
bits=${4:-3}
[[ "$bits" == 2 || "$bits" == 3 ]] || { echo "bits must be 2 or 3" >&2; exit 2; }
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
source_dir="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/source/layer-$(printf '%03d' "$layer")"
geometry=rotating-uneven-768-640-640-v1
variant="$geometry"
[[ "$bits" == 3 ]] || variant="$geometry-bits${bits}"
output="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/encoded/$variant"
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/workers/$variant/layer-$(printf '%03d' "$layer")-$stamp"
log="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/$variant-layer-$(printf '%03d' "$layer")-$stamp.log"
inventory=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/source-inventory.json
container="glm53-tp3-k${bits}-uneven-L$(printf '%03d' "$layer")"
image=glm53-exl3:e2-c190db1

mkdir -p "$output" "$state" "$(dirname "$log")"
if [ ! -f "$source_dir/STAGED_OK" ]; then echo "source layer is not staged and verified" >&2; exit 2; fi
if [ ! -f "$inventory" ]; then echo "sealed source inventory is missing" >&2; exit 2; fi
if docker ps --format '{{.Names}}' | grep -Eq '^(glm53-exl3-(head|worker)|minimax-h3-comfy)$'; then
    echo "refusing encoder beside a protected GPU service" >&2
    exit 2
fi
available=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
if (( available < 12 * 1024 * 1024 * 1024 )); then echo "host memory reserve is below 12 GiB" >&2; exit 2; fi
docker rm "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --gpus all --ipc host --shm-size 8g --network none \
    -e PYTHONPATH=/workspace:/workspace/encoder-r10 \
    -e TP3K_BITS="$bits" \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
    -v "$project:/workspace:ro" \
    -v "$source_dir:/source:ro" \
    -v "$inventory:/source-inventory.json:ro" \
    -v "$output:/work" \
    --entrypoint python3 "$image" -m tp3k3.encoder \
    --source /source --source-inventory /source-inventory.json --output /work \
    --layer "$layer" --experts "$experts" \
    --numeric-core /workspace/encoder-r10/lineage/encode_tr3_v31.py \
    --numeric-core-sha256 e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032 \
    --extension /usr/local/lib/python3.12/dist-packages/exllamav3_ext.cpython-312-aarch64-linux-gnu.so \
    --extension-sha256 7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2

docker logs -f "$container" > "$log" 2>&1 &
log_pid=$!
set +e
python3 "$project/scripts/watchdog.py" --container "$container" --state-dir "$state"
watchdog_status=$?
wait_status=$(docker wait "$container")
wait "$log_pid"
set -e
docker inspect "$container" > "$state/container-inspect.json"
printf '%s\n' "$wait_status" > "$state/container-exit-code.txt"
oom_killed=$(docker inspect -f '{{.State.OOMKilled}}' "$container")
printf '%s\n' "$oom_killed" > "$state/container-oom-killed.txt"
if (( watchdog_status != 0 )) || [ "$wait_status" != 0 ] || [ "$oom_killed" != false ]; then
    exit 1
fi
