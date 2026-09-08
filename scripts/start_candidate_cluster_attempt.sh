#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
replica_basename=${2:?local replica basename required}
attempt=${3:?unique candidate attempt label required}
nccl_channels=${4:-4}
nccl_buffsize=${5:-1048576}
candidate_mode=${6:-eager}
lazy_cache_experts=${7:-32}
lazy_cache_storage=${8:-arena}
execute_model_timeout_seconds=${9:-3600}
lazy_execution=${10:-chunked_fused}
kv_cache_memory_bytes=${11:-}
rank_pack_basename=${12:-}
cuda_allocator_conf=${13:-expandable_segments:True}
arena_block_slots=${14:-16}
resident_min_available_bytes=${GLM53_RESIDENT_MIN_AVAILABLE_BYTES:-8589934592}
watchdog_reserve_gib=${GLM53_WATCHDOG_RESERVE_GIB:-8}
uva_resident=${GLM53_LAZY_K3_UVA:-0}
max_num_batched_tokens=${GLM53_LAZY_MAX_NUM_BATCHED_TOKENS:-1024}
enforce_eager=${GLM53_ENFORCE_EAGER:-1}
graph_mode=${GLM53_CUDA_GRAPH_MODE:-}
[[ -z "$graph_mode" || "$graph_mode" == FULL_DECODE_ONLY ]] || { echo "unsupported CUDA graph mode" >&2; exit 2; }
spinwait_ms=${GLM53_SPINWAIT_MS:-stock}
[[ "$spinwait_ms" == stock || "$spinwait_ms" == 16 ]] || { echo "unsupported experiment spin window" >&2; exit 2; }
spec_method=${GLM53_SPEC_METHOD:-}
spec_tokens=${GLM53_SPEC_TOKENS:-}
spec_draft_tp=${GLM53_SPEC_DRAFT_TP:-}
dcp_size=${GLM53_DCP_SIZE:-1}
cp_kv_interleave_size=${GLM53_CP_KV_INTERLEAVE_SIZE:-1}
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || { echo "invalid attempt label" >&2; exit 2; }
[[ "$nccl_channels" =~ ^[0-9]+$ ]] && (( nccl_channels >= 1 && nccl_channels <= 64 )) || { echo "invalid NCCL channel count" >&2; exit 2; }
[[ "$nccl_buffsize" =~ ^[0-9]+$ ]] && (( nccl_buffsize >= 131072 && nccl_buffsize <= 16777216 )) || { echo "invalid NCCL buffer size" >&2; exit 2; }
[[ "$candidate_mode" == eager || "$candidate_mode" == lazy ]] || { echo "candidate mode must be eager or lazy" >&2; exit 2; }
[[ "$lazy_cache_storage" == legacy || "$lazy_cache_storage" == arena ]] || { echo "lazy cache storage must be legacy or arena" >&2; exit 2; }
[[ "$execute_model_timeout_seconds" =~ ^[0-9]+$ ]] && (( execute_model_timeout_seconds >= 300 && execute_model_timeout_seconds <= 7200 )) || { echo "invalid execute-model timeout" >&2; exit 2; }
[[ "$lazy_execution" == python_loop || "$lazy_execution" == chunked_fused || "$lazy_execution" == shared_layer_fused || "$lazy_execution" == resident_fused || "$lazy_execution" == resident_uva ]] || { echo "invalid lazy execution mode" >&2; exit 2; }
[[ "$lazy_cache_experts" =~ ^[0-9]+$ ]] || { echo "invalid lazy cache expert count" >&2; exit 2; }
if [[ "$lazy_execution" == shared_layer_fused || "$lazy_execution" == resident_fused ]]; then
    (( lazy_cache_experts == 256 )) || { echo "shared-layer execution requires 256 arena experts" >&2; exit 2; }
else
    if [[ "$lazy_execution" == resident_uva ]]; then
        (( lazy_cache_experts == 256 )) || { echo "resident-UVA execution requires 256 arena experts" >&2; exit 2; }
    else
        (( lazy_cache_experts >= 1 && lazy_cache_experts <= 192 )) || { echo "invalid lazy cache expert count" >&2; exit 2; }
    fi
fi
[[ "$resident_min_available_bytes" =~ ^[0-9]+$ ]] && (( resident_min_available_bytes >= 4 * 1024 * 1024 * 1024 )) || { echo "invalid resident host reserve" >&2; exit 2; }
[[ "$watchdog_reserve_gib" =~ ^[0-9]+$ ]] && (( watchdog_reserve_gib >= 4 && watchdog_reserve_gib <= 64 )) || { echo "invalid watchdog reserve" >&2; exit 2; }
[[ "$uva_resident" == 0 || "$uva_resident" == 1 ]] || { echo "invalid UVA resident flag" >&2; exit 2; }
[[ "$max_num_batched_tokens" =~ ^[0-9]+$ ]] && (( max_num_batched_tokens >= 1 && max_num_batched_tokens <= 8192 )) || { echo "invalid max batched tokens" >&2; exit 2; }
[[ "$enforce_eager" == 0 || "$enforce_eager" == 1 ]] || { echo "invalid enforce-eager flag" >&2; exit 2; }
if [[ -n "$spec_method" ]]; then
    [[ "$spec_method" == mtp ]] || { echo "unsupported speculative method" >&2; exit 2; }
    [[ "$spec_tokens" =~ ^[0-9]+$ ]] && (( spec_tokens >= 1 && spec_tokens <= 4 )) || { echo "speculative tokens must be 1..4" >&2; exit 2; }
else
    spec_tokens=0
fi
if [[ -n "$spec_draft_tp" ]]; then
    [[ "$spec_draft_tp" == 1 || "$spec_draft_tp" == 3 ]] || { echo "invalid speculative draft TP" >&2; exit 2; }
    [[ "$spec_method" == mtp && "$spec_tokens" =~ ^[0-9]+$ ]] || { echo "speculative draft TP requires MTP" >&2; exit 2; }
fi
[[ "$dcp_size" =~ ^[0-9]+$ ]] && (( dcp_size >= 1 && dcp_size <= 3 )) || { echo "invalid DCP size" >&2; exit 2; }
[[ "$cp_kv_interleave_size" =~ ^[0-9]+$ ]] && (( cp_kv_interleave_size >= 1 && cp_kv_interleave_size <= 64 )) || { echo "invalid CP KV interleave size" >&2; exit 2; }
if [[ "$lazy_execution" == resident_uva ]]; then
    [[ "$candidate_mode" == lazy && "$lazy_cache_storage" == arena && "$uva_resident" == 1 && -n "$rank_pack_basename" ]] || {
        echo "resident-UVA requires lazy mode, arena storage, UVA=1, and a rank pack" >&2
        exit 2
    }
fi

if [[ "$candidate_mode" == lazy ]]; then
    rank_slices=0
    lazy_k3=1
    fused_moe=0
    if [[ "$lazy_execution" == resident_uva ]]; then
        gpu_mem_util=${GLM53_LAZY_GPU_MEMORY_UTILIZATION:-0.12}
        kv_cache_memory_bytes=${kv_cache_memory_bytes:-1812613120}
        exl3_fat_kernel=0
    else
        gpu_mem_util=0.28
        kv_cache_memory_bytes=${kv_cache_memory_bytes:-3221225472}
        exl3_fat_kernel=1
    fi
else
    rank_slices=1
    lazy_k3=0
    fused_moe=1
    gpu_mem_util=0.86
    kv_cache_memory_bytes=${kv_cache_memory_bytes:-0}
    exl3_fat_kernel=1
fi
[[ "$kv_cache_memory_bytes" =~ ^[0-9]+$ ]] || { echo "invalid KV cache memory byte count" >&2; exit 2; }
if (( kv_cache_memory_bytes != 0 )); then
    min_kv_cache_bytes=2147483648
    [[ "$lazy_execution" == resident_uva ]] && min_kv_cache_bytes=1073741824
    (( kv_cache_memory_bytes >= min_kv_cache_bytes && kv_cache_memory_bytes <= 8589934592 )) || {
        echo "fixed KV cache is outside the supported range" >&2
        exit 2
    }
fi
if [[ -n "$rank_pack_basename" ]]; then
    [[ "$rank_pack_basename" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || { echo "invalid rank-pack basename" >&2; exit 2; }
fi
[[ "$cuda_allocator_conf" == "expandable_segments:True" || "$cuda_allocator_conf" == "expandable_segments:False" || "$cuda_allocator_conf" == "backend:cudaMallocAsync" ]] || {
    echo "unsupported CUDA allocator configuration" >&2
    exit 2
}
[[ "$arena_block_slots" =~ ^[0-9]+$ ]] && (( arena_block_slots >= 1 && arena_block_slots <= 64 )) || {
    echo "arena block slots must be 1..64" >&2
    exit 2
}

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
candidate_root="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate"
state="$candidate_root/attempts/$attempt"
logs="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/candidate-attempts/$attempt"
replication="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/replication/SUMMARY.json"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
host_ips=(192.168.0.238 192.168.0.110 192.168.0.234)
port=8893
master_port=29654
head_ip=192.168.0.238
container_prefix="glm53-k3-cand-$attempt"
started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)

if [[ -e "$state" ]]; then
    echo "candidate attempt state already exists: $state" >&2
    exit 2
fi
mkdir -p "$state/ranks" "$logs" "$candidate_root"
printf '%s\n' "$started" > "$state/started-at.txt"

write_status() {
    local phase=$1 health=${2:-0}
    jq -n --arg phase "$phase" --arg attempt "$attempt" --arg started "$started" \
        --arg updated "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        --arg health "$health" --arg mode "$candidate_mode" \
        --argjson channels "$nccl_channels" \
        --argjson buffsize "$nccl_buffsize" \
        --argjson lazy_cache_experts "$lazy_cache_experts" \
        --arg lazy_cache_storage "$lazy_cache_storage" \
        --argjson execute_model_timeout_seconds "$execute_model_timeout_seconds" \
        --arg lazy_execution "$lazy_execution" \
        --argjson gpu_memory_utilization "$gpu_mem_util" \
        --argjson kv_cache_memory_bytes "$kv_cache_memory_bytes" \
        --arg rank_pack_basename "$rank_pack_basename" \
        --arg cuda_allocator_conf "$cuda_allocator_conf" \
        --argjson arena_block_slots "$arena_block_slots" \
        '{schema:"glm53-full-exl3-tp3.candidate-status.v8",attempt:$attempt,mode:$mode,phase:$phase,started_at:$started,updated_at:$updated,health_http:($health|tonumber),gpu_memory_utilization:$gpu_memory_utilization,kv_cache_memory_bytes:$kv_cache_memory_bytes,rank_local_pack_basename:(if $rank_pack_basename=="" then null else $rank_pack_basename end),lazy_cache_experts_per_layer:$lazy_cache_experts,lazy_cache_policy:"lfu",lazy_cache_storage:$lazy_cache_storage,lazy_execution:$lazy_execution,lazy_arena_block_slots:$arena_block_slots,execute_model_timeout_seconds:$execute_model_timeout_seconds,cuda_allocator_conf:$cuda_allocator_conf,nccl:{min_nchannels:$channels,max_nchannels:$channels,buffsize:$buffsize}}' \
        > "$state/STATUS.json.tmp"
    mv "$state/STATUS.json.tmp" "$state/STATUS.json"
}

write_status preflight
jq -n --arg attempt "$attempt" --arg state "$state" --arg started "$started" \
    '{schema:"glm53-full-exl3-tp3.candidate-current-attempt.v1",attempt:$attempt,state_dir:$state,started_at:$started}' \
    > "$candidate_root/CURRENT_ATTEMPT.json.tmp"
mv "$candidate_root/CURRENT_ATTEMPT.json.tmp" "$candidate_root/CURRENT_ATTEMPT.json"
jq -e '.passed == true and (.ranks|length) == 3 and all(.ranks[];.passed)' "$replication" >/dev/null

stop_candidates() {
    for rank in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "docker stop --time 60 '$container_prefix-rank$rank' >/dev/null 2>&1 || true; \
             tmux kill-session -t '$container_prefix-watchdog-$rank' 2>/dev/null || true; \
             tmux kill-session -t '$container_prefix-log-$rank' 2>/dev/null || true" &
    done
    wait || true
}

diagnose_failure() {
    code=$?
    trap - EXIT
    failed_at=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        ssh -F zima-ssh-config "$node" "docker inspect '$container_prefix-rank$rank'" \
            > "$state/ranks/rank-$rank.inspect.json" 2> "$state/ranks/rank-$rank.inspect.err" || true
        ssh -F zima-ssh-config "$node" "docker logs '$container_prefix-rank$rank'" \
            > "$logs/rank-$rank-final.log" 2>&1 || true
        remote_watchdog="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt/rank-$rank/watchdog"
        scp -q -F zima-ssh-config "$node:$remote_watchdog/STOP.json" \
            "$state/ranks/rank-$rank-watchdog-STOP.json" 2>/dev/null || true
        scp -q -F zima-ssh-config "$node:$remote_watchdog/metrics.csv" \
            "$state/ranks/rank-$rank.metrics.csv" 2>/dev/null || true
    done
    stop_candidates
    python3 "$project/scripts/kernel_audit.py" --since "$started" --until "$failed_at" \
        --output "$state/failure-kernel-audit.json" || true
    jq -n --arg at "$failed_at" --arg attempt "$attempt" --argjson exit_code "$code" \
        '{schema:"glm53-full-exl3-tp3.candidate-failure.v2",attempt:$attempt,failed_at:$at,exit_code:$exit_code,candidates_stopped:true,automatic_service_restore_attempted:false}' \
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
    (( available >= resident_min_available_bytes ))
    ssh -F zima-ssh-config "$node" \
        "test -s '/home/mj-kang/Dev/models/$replica_basename/ASSEMBLY_COMPLETE.json'; \
         ! docker ps --format '{{.Names}}' | grep -Eq '^(glm53-tp3-k3-uneven-L[0-9][0-9][0-9]|glm53-k3-cand|glm53-k3-candidate|glm53-k3-lazy)'; \
         ! docker inspect '$container_prefix-rank$rank' >/dev/null 2>&1"
    if [[ -n "$rank_pack_basename" ]]; then
        ssh -F zima-ssh-config "$node" \
            "jq -e '.passed == true and .rank == $rank and .layers == 76' '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename/rank-$rank/COMPLETE.json' >/dev/null"
    fi
    rsync -a --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude outputs/ --exclude logs/ --exclude state/ "$project/" -e "ssh -F zima-ssh-config" "$node:$project/"
done

python3 "$project/scripts/kernel_audit.py" --since "$started" --output "$state/prelaunch-kernel-audit.json"
write_status launching

for rank in 0 1 2; do
    node=${nodes[$rank]}
    host_ip=${host_ips[$rank]}
    remote_state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt/rank-$rank"
    remote_log="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/candidate-attempts/$attempt-rank-$rank.log"
    rank_pack_docker_args=
    if [[ -n "$rank_pack_basename" ]]; then
        rank_pack_host="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename/rank-$rank"
        rank_pack_docker_args="-e GLM53_LAZY_K3_PACKED_ROOT=/rank-packs -v $rank_pack_host:/rank-packs:ro"
    fi
    ssh -F zima-ssh-config "$node" "mkdir -p '$remote_state/watchdog' '$remote_state/capture' \"\$(dirname '$remote_log')\" \
 '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/vllm' \
 '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/triton'; \
docker run -d --name '$container_prefix-rank$rank' --gpus all --network host --ipc host --shm-size 32g \
 --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
 -e PYTHONPATH=/runtime:/workspace \
 -e VLLM_GLM53_EXL3_TP3_FULL_MODEL=1 -e VLLM_GLM53_EXL3_TP3_RANK_SLICES='$rank_slices' \
 -e VLLM_GLM53_EXL3_LAZY_K3='$lazy_k3' -e GLM53_LAZY_K3_MODEL_DIR=/model \
 -e GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER='$lazy_cache_experts' \
 -e GLM53_LAZY_K3_CACHE_POLICY=lfu -e GLM53_LAZY_K3_CACHE_STORAGE='$lazy_cache_storage' \
 -e GLM53_LAZY_K3_ARENA_BLOCK_SLOTS='$arena_block_slots' \
 -e GLM53_LAZY_K3_EXECUTION='$lazy_execution' -e GLM53_LAZY_K3_UVA='$uva_resident' \
 -e GLM53_RESIDENT_MIN_AVAILABLE_BYTES='$resident_min_available_bytes' -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
 -e ENFORCE_EAGER='$enforce_eager' -e CUDA_GRAPH_MODE='$graph_mode' -e GLM53_SPINWAIT_MS='$spinwait_ms' -e SPEC_METHOD='$spec_method' -e SPEC_TOKENS='$spec_tokens' -e SPEC_DRAFT_TP='$spec_draft_tp' -e DCP_SIZE='$dcp_size' -e CP_KV_INTERLEAVE_SIZE='$cp_kv_interleave_size' \
 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS='$execute_model_timeout_seconds' \
 -e KV_CACHE_MEMORY_BYTES='$kv_cache_memory_bytes' \
 $rank_pack_docker_args \
 -e GLM53_TP3_SPARSE_MLA_SHADOW_COMPARE=1 \
 -e GLM53_K3_LOGIT_CAPTURE_ROOT=/capture \
 -e VLLM_GLM53_EXL3_TP3_GEOMETRY_ID=rotating-uneven-768-640-640-v1 \
 -e EXL3_FUSED_MOE='$fused_moe' -e EXL3_FAT_KERNEL='$exl3_fat_kernel' -e EXL3_TEMP_ROWS_FUSED=128 \
 -e EXL3_MOE_ROW_TILE=0 -e EXL3_FAT_BATCHED=0 -e EXL3_FAT_SORTED=0 \
 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
 -e VLLM_CACHE_ROOT=/root/.cache/vllm -e TRITON_CACHE_DIR=/root/.triton/cache \
 -e PYTORCH_CUDA_ALLOC_CONF='$cuda_allocator_conf' \
 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
 -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
 -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_DEBUG=INFO \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
 -e NCCL_MIN_NCHANNELS='$nccl_channels' -e NCCL_MAX_NCHANNELS='$nccl_channels' -e NCCL_BUFFSIZE='$nccl_buffsize' \
 -e VLLM_HOST_IP='$host_ip' -e MODEL_DIR=/model -e SERVED_MODEL_NAME=GLM-5.3-K3-TP3-CANDIDATE \
 -e PORT='$port' -e NODE_RANK='$rank' -e HEAD_IP='$head_ip' -e MASTER_PORT='$master_port' \
 -e MAX_MODEL_LEN=32768 -e GPU_MEM_UTIL='$gpu_mem_util' -e MAX_NUM_SEQS=1 \
 -e MAX_NUM_BATCHED_TOKENS='$max_num_batched_tokens' -e KV_CACHE_DTYPE=fp8 \
 -v '/home/mj-kang/Dev/models/$replica_basename:/model:ro' -v '$project:/workspace:ro' \
 -v '$project/runtime:/runtime:ro' \
 -v '$remote_state/capture:/capture' \
 -v '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/vllm:/root/.cache/vllm' \
 -v '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/triton:/root/.triton/cache' \
 --entrypoint bash glm53-exl3:e2-c190db1 /runtime/run_candidate_node.sh >/dev/null; \
 tmux new-session -d -s '$container_prefix-watchdog-$rank' \"python3 '$project/scripts/watchdog.py' --container '$container_prefix-rank$rank' --state-dir '$remote_state/watchdog' --reserve-gib '$watchdog_reserve_gib' > '$remote_state/watchdog/watchdog.log' 2>&1\"; \
 if [[ '$lazy_execution' == resident_uva ]]; then tmux new-session -d -s '$container_prefix-page-cache-$rank' \"GLM53_WATCH_CHECKPOINT='/home/mj-kang/Dev/models/$replica_basename' GLM53_WATCH_RANK_PACK_ROOT='/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename' GLM53_WATCH_STATE='$remote_state/page-cache-watch' '$project/watch_resident_page_cache.sh' '$rank' '$container_prefix'\"; fi; \
 tmux new-session -d -s '$container_prefix-log-$rank' \"docker logs -f '$container_prefix-rank$rank' > '$remote_log' 2>&1\""
done

deadline=$(( $(date +%s) + 3600 ))
write_status loading
while :; do
    for rank in 0 1 2; do
        node=${nodes[$rank]}
        status_row=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.State.ExitCode}}' '$container_prefix-rank$rank'")
        read -r running oom exit_code <<< "$status_row"
        [[ "$running" == true && "$oom" == false ]] || {
            printf '%s\n' "candidate rank $rank stopped during startup: $status_row" >&2
            exit 2
        }
        ssh -F zima-ssh-config "$node" "test ! -e '/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt/rank-$rank/watchdog/STOP.json'"
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
    remote_state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt/rank-$rank"
    page_cache_args=""
    if [[ -n "$rank_pack_basename" ]]; then
        page_cache_args="--rank-pack-dir /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename/rank-$rank"
    fi
    ssh -F zima-ssh-config "$node" \
        "python3 '$project/scripts/evict_model_page_cache.py' \
          --checkpoint-dir '/home/mj-kang/Dev/models/$replica_basename' \
          $page_cache_args \
          --output '$remote_state/page-cache-eviction.json'"
    ssh -F zima-ssh-config "$node" \
        "jq -e '.passed == true and .checkpoint_files > 0' '$remote_state/page-cache-eviction.json' >/dev/null"
    ssh -F zima-ssh-config "$node" "docker inspect '$container_prefix-rank$rank'" > "$state/ranks/rank-$rank.ready-inspect.json"
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= resident_min_available_bytes )) || { printf '%s\n' "$node reserve below configured floor" >&2; exit 2; }
    printf '%s\n' "$available" > "$state/ranks/rank-$rank.ready-mem-available-bytes"
    ssh -F zima-ssh-config "$node" "nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used --format=csv" > "$state/ranks/rank-$rank.ready-gpu.csv"
done
python3 "$project/scripts/kernel_audit.py" --since "$started" --until "$ready" \
    --output "$state/startup-kernel-audit.json"
jq -n --arg attempt "$attempt" --arg started "$started" --arg ready "$ready" \
    --arg endpoint "http://$head_ip:$port" --arg model "$replica_basename" \
    --arg mode "$candidate_mode" \
    --argjson channels "$nccl_channels" --argjson buffsize "$nccl_buffsize" \
    --argjson lazy_cache_experts "$lazy_cache_experts" \
    --argjson max_num_batched_tokens "$max_num_batched_tokens" \
    --argjson enforce_eager "$enforce_eager" --arg spec_method "$spec_method" --argjson spec_tokens "$spec_tokens" --arg spec_draft_tp "$spec_draft_tp" --argjson dcp_size "$dcp_size" --argjson cp_kv_interleave_size "$cp_kv_interleave_size" \
    --arg lazy_cache_storage "$lazy_cache_storage" \
    --argjson execute_model_timeout_seconds "$execute_model_timeout_seconds" \
    --arg lazy_execution "$lazy_execution" \
    --argjson gpu_memory_utilization "$gpu_mem_util" \
    --argjson kv_cache_memory_bytes "$kv_cache_memory_bytes" \
    --arg rank_pack_basename "$rank_pack_basename" \
    --arg cuda_allocator_conf "$cuda_allocator_conf" \
    --argjson arena_block_slots "$arena_block_slots" \
    '{schema:"glm53-full-exl3-tp3.candidate-ready.v8",passed:true,attempt:$attempt,mode:$mode,started_at:$started,ready_at:$ready,endpoint:$endpoint,replica_basename:$model,tp:3,dcp:$dcp_size,pp:1,max_model_len:32768,max_num_seqs:1,max_num_batched_tokens:$max_num_batched_tokens,gpu_memory_utilization:$gpu_memory_utilization,kv_cache_memory_bytes:$kv_cache_memory_bytes,rank_local_pack_basename:(if $rank_pack_basename=="" then null else $rank_pack_basename end),lazy_cache_experts_per_layer:$lazy_cache_experts,lazy_cache_policy:"lfu",lazy_cache_storage:$lazy_cache_storage,lazy_execution:$lazy_execution,lazy_arena_block_slots:$arena_block_slots,enforce_eager:$enforce_eager,spec_method:(if $spec_method=="" then null else $spec_method end),spec_tokens:$spec_tokens,spec_draft_tp:(if $spec_draft_tp=="" then null else ($spec_draft_tp|tonumber) end),cp_kv_interleave_size:$cp_kv_interleave_size,execute_model_timeout_seconds:$execute_model_timeout_seconds,cuda_allocator_conf:$cuda_allocator_conf,kv_cache_dtype:"fp8",graphs:($enforce_eager==0),mtp:($spec_method=="mtp"),prefix_cache:false,nccl:{min_nchannels:$channels,max_nchannels:$channels,buffsize:$buffsize}}' \
    > "$state/READY.json.tmp"
mv "$state/READY.json.tmp" "$state/READY.json"
write_status ready 200
trap - EXIT
