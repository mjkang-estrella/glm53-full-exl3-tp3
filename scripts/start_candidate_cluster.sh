#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
replica_basename=${2:?local replica basename required}
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate"
logs="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/candidate"
replication="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/replication/SUMMARY.json"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
host_ips=(192.168.0.238 192.168.0.110 192.168.0.234)
port=8893
master_port=29653
head_ip=192.168.0.238
started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
mkdir -p "$state/ranks" "$logs"
printf '%s\n' "$started" > "$state/started-at.txt"

write_status() {
    local phase=$1 health=${2:-0}
    jq -n --arg phase "$phase" --arg started "$started" \
        --arg updated "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        --arg health "$health" \
        '{schema:"glm53-full-exl3-tp3.candidate-status.v1",phase:$phase,started_at:$started,updated_at:$updated,health_http:($health|tonumber)}' \
        > "$state/STATUS.json.tmp"
    mv "$state/STATUS.json.tmp" "$state/STATUS.json"
}

write_status preflight
jq -e '.passed == true and (.ranks|length) == 3 and all(.ranks[];.passed)' "$replication" >/dev/null

stop_candidates() {
    for rank in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "docker stop --time 60 'glm53-k3-candidate-rank$rank' >/dev/null 2>&1 || true; \
             tmux kill-session -t 'glm53-k3-candidate-watchdog-$rank' 2>/dev/null || true; \
             tmux kill-session -t 'glm53-k3-candidate-log-$rank' 2>/dev/null || true" &
    done
    wait || true
}

diagnose_failure() {
    code=$?
    trap - EXIT
    failed_at=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        ssh -F zima-ssh-config "$node" "docker inspect 'glm53-k3-candidate-rank$rank'" \
            > "$state/ranks/rank-$rank.inspect.json" 2> "$state/ranks/rank-$rank.inspect.err" || true
        ssh -F zima-ssh-config "$node" "docker logs 'glm53-k3-candidate-rank$rank'" \
            > "$logs/rank-$rank-final.log" 2>&1 || true
        scp -q -F zima-ssh-config "$node:/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/rank-$rank/watchdog/STOP.json" \
            "$state/ranks/rank-$rank-watchdog-STOP.json" 2>/dev/null || true
    done
    stop_candidates
    python3 "$project/scripts/kernel_audit.py" --since "$started" --until "$failed_at" \
        --output "$state/failure-kernel-audit.json" || true
    jq -n --arg at "$failed_at" --argjson exit_code "$code" \
        '{schema:"glm53-full-exl3-tp3.candidate-failure.v1",failed_at:$at,exit_code:$exit_code,candidates_stopped:true,automatic_service_restore_attempted:false}' \
        > "$state/FAILED.json.tmp"
    mv "$state/FAILED.json.tmp" "$state/FAILED.json"
    write_status failed
    exit "$code"
}
trap diagnose_failure EXIT

for rank in 0 1 2; do
    node=${nodes[$rank]}
    service=${services[$rank]}
    running=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '$service'")
    [[ "$running" == false ]]
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 ))
    ssh -F zima-ssh-config "$node" \
        "test -s '/home/mj-kang/Dev/models/$replica_basename/ASSEMBLY_COMPLETE.json'; \
         ! docker ps --format '{{.Names}}' | grep -Eq '^glm53-tp3-k3-uneven-L[0-9][0-9][0-9]$'; \
         ! docker inspect 'glm53-k3-candidate-rank$rank' >/dev/null 2>&1"
    rsync -a --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude outputs/ --exclude logs/ --exclude state/ "$project/" -e "ssh -F zima-ssh-config" "$node:$project/"
done

python3 "$project/scripts/kernel_audit.py" --since "$started" \
    --output "$state/prelaunch-kernel-audit.json"
write_status launching

for rank in 0 1 2; do
    node=${nodes[$rank]}
    host_ip=${host_ips[$rank]}
    remote_state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/rank-$rank"
    remote_log="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/candidate-rank-$rank.log"
    ssh -F zima-ssh-config "$node" "mkdir -p '$remote_state/watchdog' '$remote_state/capture' \"\$(dirname '$remote_log')\" \
 '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/vllm' \
 '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/triton'; \
docker run -d --name 'glm53-k3-candidate-rank$rank' --gpus all --network host --ipc host --shm-size 32g \
 --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
 -e PYTHONPATH=/runtime:/workspace \
 -e VLLM_GLM53_EXL3_TP3_FULL_MODEL=1 -e VLLM_GLM53_EXL3_TP3_RANK_SLICES=1 \
 -e GLM53_K3_LOGIT_CAPTURE_ROOT=/capture \
 -e VLLM_GLM53_EXL3_TP3_GEOMETRY_ID=rotating-uneven-768-640-640-v1 \
 -e EXL3_FUSED_MOE=1 -e EXL3_FAT_KERNEL=1 -e EXL3_TEMP_ROWS_FUSED=128 \
 -e EXL3_MOE_ROW_TILE=0 -e EXL3_FAT_BATCHED=0 -e EXL3_FAT_SORTED=0 \
 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
 -e VLLM_CACHE_ROOT=/root/.cache/vllm -e TRITON_CACHE_DIR=/root/.triton/cache \
 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
 -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
 -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_DEBUG=INFO \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
 -e VLLM_HOST_IP='$host_ip' -e MODEL_DIR=/model -e SERVED_MODEL_NAME=GLM-5.3-K3-TP3-CANDIDATE \
 -e PORT='$port' -e NODE_RANK='$rank' -e HEAD_IP='$head_ip' -e MASTER_PORT='$master_port' \
 -e MAX_MODEL_LEN=32768 -e GPU_MEM_UTIL=0.86 -e MAX_NUM_SEQS=1 \
 -e MAX_NUM_BATCHED_TOKENS=1024 -e KV_CACHE_DTYPE=fp8 \
 -v '/home/mj-kang/Dev/models/$replica_basename:/model:ro' -v '$project:/workspace:ro' \
 -v '$project/runtime:/runtime:ro' \
 -v '$remote_state/capture:/capture' \
 -v '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/vllm:/root/.cache/vllm' \
 -v '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/triton:/root/.triton/cache' \
 --entrypoint bash glm53-exl3:e2-c190db1 /runtime/run_candidate_node.sh >/dev/null; \
 tmux kill-session -t 'glm53-k3-candidate-watchdog-$rank' 2>/dev/null || true; \
 tmux new-session -d -s 'glm53-k3-candidate-watchdog-$rank' \"python3 '$project/scripts/watchdog.py' --container 'glm53-k3-candidate-rank$rank' --state-dir '$remote_state/watchdog' --reserve-gib 12 > '$remote_state/watchdog/watchdog.log' 2>&1\"; \
 tmux kill-session -t 'glm53-k3-candidate-log-$rank' 2>/dev/null || true; \
 tmux new-session -d -s 'glm53-k3-candidate-log-$rank' \"docker logs -f 'glm53-k3-candidate-rank$rank' > '$remote_log' 2>&1\""
done

deadline=$(( $(date +%s) + 3600 ))
write_status loading
while :; do
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        status_row=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}' 'glm53-k3-candidate-rank$rank'")
        read -r running oom exit_code <<< "$status_row"
        [[ "$running" == true && "$oom" == false ]] || {
            printf '%s\n' "candidate rank $rank stopped during startup: $status_row" >&2
            exit 2
        }
        ssh -F zima-ssh-config "$node" "test ! -e '/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/rank-$rank/watchdog/STOP.json'"
    done
    code=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' "http://$head_ip:$port/health" || true)
    [[ "$code" == 200 ]] && break
    write_status loading "${code:-0}"
    (( $(date +%s) < deadline )) || { printf '%s\n' 'candidate startup timeout' >&2; exit 2; }
    sleep 10
done

ready=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
for rank in 0 1 2; do
    node=${nodes[$rank]}
    ssh -F zima-ssh-config "$node" "docker inspect 'glm53-k3-candidate-rank$rank'" > "$state/ranks/rank-$rank.ready-inspect.json"
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 )) || { printf '%s\n' "$node reserve below 12 GiB after ready" >&2; exit 2; }
    printf '%s\n' "$available" > "$state/ranks/rank-$rank.ready-mem-available-bytes"
    ssh -F zima-ssh-config "$node" "nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used --format=csv" > "$state/ranks/rank-$rank.ready-gpu.csv"
done
python3 "$project/scripts/kernel_audit.py" --since "$started" --until "$ready" \
    --output "$state/startup-kernel-audit.json"
jq -n --arg started "$started" --arg ready "$ready" --arg endpoint "http://$head_ip:$port" \
    --arg model "$replica_basename" \
    '{schema:"glm53-full-exl3-tp3.candidate-ready.v1",passed:true,started_at:$started,ready_at:$ready,endpoint:$endpoint,replica_basename:$model,tp:3,dcp:1,pp:1,max_model_len:32768,max_num_seqs:1,max_num_batched_tokens:1024,kv_cache_dtype:"fp8",graphs:false,mtp:false,prefix_cache:false}' \
    > "$state/READY.json.tmp"
mv "$state/READY.json.tmp" "$state/READY.json"
write_status ready 200
trap - EXIT
