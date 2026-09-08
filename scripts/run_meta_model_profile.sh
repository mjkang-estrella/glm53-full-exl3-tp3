#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
replica_basename=${2:?local replica basename required}
attempt=${3:?unique meta-profile attempt label required}
placement=${4:-tp3}
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || { echo "invalid attempt label" >&2; exit 2; }
case "$placement" in
    tp3) tensor_parallel=3; pipeline_parallel=1; rank_slices=1; pp3_all_slices=0; lazy_k3=0 ;;
    tp3-lazy) tensor_parallel=3; pipeline_parallel=1; rank_slices=0; pp3_all_slices=0; lazy_k3=1 ;;
    pp3) tensor_parallel=1; pipeline_parallel=3; rank_slices=0; pp3_all_slices=1; lazy_k3=0 ;;
    *) echo "placement must be tp3, tp3-lazy, or pp3" >&2; exit 2 ;;
esac

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/memory-designs/$attempt"
logs="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/memory-designs/$attempt"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
prefix="glm53-k3-meta-$attempt"
head_ip=192.168.0.238
master_port=29656
started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
[[ ! -e "$state" ]] || { echo "meta profile state already exists" >&2; exit 2; }
mkdir -p "$state/ranks" "$logs"
printf '%s\n' "$started" > "$state/started-at.txt"

stop_all() {
    for rank in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "docker stop --time 30 '$prefix-rank$rank' >/dev/null 2>&1 || true; \
             tmux kill-session -t '$prefix-watchdog-$rank' 2>/dev/null || true; \
             tmux kill-session -t '$prefix-log-$rank' 2>/dev/null || true" &
    done
    wait || true
}

collect() {
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        remote="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/memory-designs/$attempt/rank-$rank"
        ssh -F zima-ssh-config "$node" "docker inspect '$prefix-rank$rank'" > "$state/ranks/rank-$rank.inspect.json" 2> "$state/ranks/rank-$rank.inspect.err" || true
        ssh -F zima-ssh-config "$node" "docker logs '$prefix-rank$rank'" > "$logs/rank-$rank.log" 2>&1 || true
        scp -q -F zima-ssh-config "$node:$remote/profile.json" "$state/ranks/rank-$rank.profile.json" 2>/dev/null || true
        scp -q -F zima-ssh-config "$node:$remote/watchdog/metrics.csv" "$state/ranks/rank-$rank.metrics.csv" 2>/dev/null || true
        scp -q -F zima-ssh-config "$node:$remote/watchdog/STOP.json" "$state/ranks/rank-$rank-watchdog-STOP.json" 2>/dev/null || true
    done
}

fail() {
    code=$?
    trap - EXIT
    failed_at=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    collect
    stop_all
    python3 scripts/kernel_audit.py --since "$started" --until "$failed_at" --output "$state/kernel-audit.json" || true
    python3 - "$state/FAILED.json" "$failed_at" "$code" <<'PY'
import json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1]); value = {"schema":"glm53-full-exl3-tp3.meta-model-profile-failure.v1","failed_at":sys.argv[2],"exit_code":int(sys.argv[3]),"candidates_stopped":True,"automatic_service_restore_attempted":False}
fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w") as handle: json.dump(value, handle, sort_keys=True, indent=2); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(temp, path)
PY
    exit "$code"
}
trap fail EXIT

for rank in 0 1 2; do
    node=${nodes[$rank]}
    [[ "$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '${services[$rank]}'")" == false ]]
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 ))
    ssh -F zima-ssh-config "$node" "test -s '/home/mj-kang/Dev/models/$replica_basename/ASSEMBLY_COMPLETE.json'; ! docker ps --format '{{.Names}}' | grep -Eq '^(glm53-k3-|glm53-tp3-k3-uneven)'; ! docker inspect '$prefix-rank$rank' >/dev/null 2>&1"
    rsync -a --exclude logs/ --exclude state/ "$project/" -e "ssh -F zima-ssh-config" "$node:$project/"
done
python3 scripts/kernel_audit.py --since "$started" --output "$state/prelaunch-kernel-audit.json"

for rank in 0 1 2; do
    node=${nodes[$rank]}
    remote="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/memory-designs/$attempt/rank-$rank"
    ssh -F zima-ssh-config "$node" "mkdir -p '$remote/watchdog' '/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/memory-designs'; \
docker run -d --name '$prefix-rank$rank' --gpus all --network host --ipc host --shm-size 4g \
 --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
 -e PYTHONPATH=/runtime:/workspace -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
 -e VLLM_GLM53_EXL3_TP3_FULL_MODEL=1 -e VLLM_GLM53_EXL3_TP3_RANK_SLICES='$rank_slices' \
 -e VLLM_GLM53_EXL3_PP3_ALL_SLICES='$pp3_all_slices' -e GLM53_META_PROFILE_PLACEMENT='$placement' \
 -e VLLM_GLM53_EXL3_LAZY_K3='$lazy_k3' -e GLM53_LAZY_K3_MODEL_DIR=/model \
 -e VLLM_GLM53_EXL3_TP3_GEOMETRY_ID=rotating-uneven-768-640-640-v1 \
 -e VLLM_GLM53_META_PROFILE=1 -e GLM53_META_PROFILE_OUTPUT=/output/profile.json \
 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
 -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
 -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_DEBUG=INFO \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
 -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=1 -e NCCL_BUFFSIZE=131072 \
 -e MASTER_ADDR='$head_ip' -e MASTER_PORT='$master_port' -e WORLD_SIZE=3 -e RANK='$rank' -e LOCAL_RANK=0 \
 -e MODEL_DIR=/model \
 -v '/home/mj-kang/Dev/models/$replica_basename:/model:ro' -v '$project:/workspace:ro' -v '$project/runtime:/runtime:ro' -v '$remote:/output' \
 --entrypoint python3 glm53-exl3:e2-c190db1 /runtime/meta_model_profile.py >/dev/null; \
tmux new-session -d -s '$prefix-watchdog-$rank' \"python3 '$project/scripts/watchdog.py' --container '$prefix-rank$rank' --state-dir '$remote/watchdog' --reserve-gib 12 --interval 2 > '$remote/watchdog/watchdog.log' 2>&1\"; \
tmux new-session -d -s '$prefix-log-$rank' \"docker logs -f '$prefix-rank$rank' > '/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/memory-designs/$attempt-rank-$rank.log' 2>&1\"" &
done
wait

deadline=$(( $(date +%s) + 900 ))
while :; do
    complete=0
    for rank in 0 1 2; do
        row=$(ssh -F zima-ssh-config "${nodes[$rank]}" "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}' '$prefix-rank$rank'")
        read -r running oom code <<< "$row"
        [[ "$oom" == false ]] || exit 2
        if [[ "$running" == false ]]; then
            [[ "$code" == 0 ]] || exit 2
            complete=$((complete + 1))
        fi
        ssh -F zima-ssh-config "${nodes[$rank]}" "test ! -e '/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/memory-designs/$attempt/rank-$rank/watchdog/STOP.json'"
    done
    (( complete == 3 )) && break
    (( $(date +%s) < deadline )) || { echo "meta model profile timeout" >&2; exit 2; }
    sleep 5
done

ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
collect
stop_all
python3 scripts/kernel_audit.py --since "$started" --until "$ended" --output "$state/kernel-audit.json"
python3 scripts/verify_meta_model_profile.py --state "$state" --logs "$logs" --placement "$placement" --output "$state/SUMMARY.json"
printf '%s\n' "$ended" > "$state/ended-at.txt"
touch "$state/PASSED"
trap - EXIT
