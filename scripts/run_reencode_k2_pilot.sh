#!/usr/bin/env bash
# Spark 2 host-side launcher. Invoke over SSH from a durable Zima tmux job.
set -euo pipefail
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
attempt=${1:?attempt required, e.g. v2}
[[ "$attempt" =~ ^v[0-9]+$ ]]
stage=${2:-encode}
[[ "$stage" == encode || "$stage" == validate ]]
state=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/quality/20260908-reencode-$attempt
output=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/quality-reencode/20260908
container=glm53-k275-reencode-layer33-20260908-$attempt
if docker ps --format '{{.Names}}' | grep -Eq '^(glm53-k3-cand-|glm53-exl3-|minimax-h3-comfy|glm53-tp3-k[23]-|glm53-k275-reencode-)'; then
  echo 'refusing encoding beside serving or an encoder' >&2; exit 2
fi
test ! -e "$output/pilot-$attempt"
test -f /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/source/layer-033/STAGED_OK
mkdir -p "$state" "$output"
available=$(awk '/MemAvailable:/ {printf "%.0f",$2*1024}' /proc/meminfo)
(( available >= 40 * 1024 * 1024 * 1024 ))
date -u -Iseconds > "$state/started-at.txt"
script=reencode_k2_pilot.py
extra=(--per-tier 8)
if [[ "$stage" == validate ]]; then
  script=validate_layer_reselection.py
  extra=(--pilot /output/pilot-v2)
fi
docker run -d --name "$container" --gpus all --ipc host --network none \
  -e PYTHONPATH=/workspace:/workspace/encoder-r10:/workspace/scripts \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -v "$project:/workspace:ro" \
  -v /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/source/layer-033:/source:ro \
  -v /home/mj-kang/Dev/state/glm53-full-exl3-tp3/source-inventory.json:/inventory.json:ro \
  -v /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/quality-calibration/20260908:/activations:ro \
  -v /home/mj-kang/Dev/models/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z:/k3:ro \
  -v /home/mj-kang/Dev/models/GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z:/k275:ro \
  -v "$output:/output" --entrypoint python3 glm53-exl3:e2-c190db1 \
  /workspace/scripts/"$script" --source /source --inventory /inventory.json \
  --k3 /k3 --k275 /k275 --activations /activations --output /output/pilot-$attempt "${extra[@]}"
docker logs -f "$container" > "$state/encoder.log" 2>&1 &
log_pid=$!
set +e
python3 "$project/scripts/watchdog.py" --container "$container" --state-dir "$state/watchdog"
guard=$?
exit_code=$(docker wait "$container")
wait "$log_pid"
set -e
docker inspect "$container" > "$state/container-inspect.json"
printf '%s\n' "$exit_code" > "$state/exit-code.txt"
[[ "$guard" == 0 && "$exit_code" == 0 ]]
test -s "$output/pilot-$attempt/RESULTS.json"
