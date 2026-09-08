#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
replica_basename=${2:?local replica basename required}
attempt=${3:?unique lazy full-model attempt required}
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || { echo "invalid attempt label" >&2; exit 2; }
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/memory-designs/$attempt"
logs="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/memory-designs/$attempt"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
prefix="glm53-k3-lazy-$attempt"
head_ip=192.168.0.238
master_port=29658
max_model_len=${GLM53_LAZY_MAX_MODEL_LEN:-4096}
max_num_batched_tokens=${GLM53_LAZY_MAX_NUM_BATCHED_TOKENS:-1024}
gpu_memory_utilization=${GLM53_LAZY_GPU_MEMORY_UTILIZATION:-0.20}
cache_experts=${GLM53_LAZY_CACHE_EXPERTS_PER_LAYER:-32}
shadow_compare=${GLM53_TP3_SPARSE_MLA_SHADOW_COMPARE:-1}
force_empty_think=${GLM53_SMOKE_FORCE_EMPTY_THINK:-1}
cache_storage=${GLM53_LAZY_K3_CACHE_STORAGE:-legacy}
execution=${GLM53_LAZY_K3_EXECUTION:-python_loop}
rank_pack_basename=${GLM53_LAZY_RANK_PACK_BASENAME:-}
kv_cache_memory_bytes=${GLM53_LAZY_KV_CACHE_MEMORY_BYTES:-0}
cpu_offload_gb=${CPU_OFFLOAD_GB:-0}
cpu_offload_params=${CPU_OFFLOAD_PARAMS:-}
nccl_buffsize=${NCCL_BUFFSIZE:-131072}
uva_resident=${GLM53_LAZY_K3_UVA:-0}
cuda_allocator_conf=${GLM53_LAZY_CUDA_ALLOCATOR_CONF:-expandable_segments:True}
arena_block_slots=${GLM53_LAZY_K3_ARENA_BLOCK_SLOTS:-16}
resident_min_available_bytes=${GLM53_RESIDENT_MIN_AVAILABLE_BYTES:-12884901888}
[[ "$execution" == python_loop || "$execution" == chunked_fused || "$execution" == resident_fused || "$execution" == resident_uva ]] || { echo "invalid execution" >&2; exit 2; }
[[ "$kv_cache_memory_bytes" =~ ^[0-9]+$ ]] || { echo "invalid KV bytes" >&2; exit 2; }
[[ "$cpu_offload_gb" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid CPU offload GB" >&2; exit 2; }
[[ "$nccl_buffsize" =~ ^[0-9]+$ ]] || { echo "invalid NCCL buffer size" >&2; exit 2; }
[[ "$uva_resident" == 0 || "$uva_resident" == 1 ]] || { echo "invalid UVA resident flag" >&2; exit 2; }
[[ "$arena_block_slots" =~ ^[0-9]+$ ]] && (( arena_block_slots >= 1 && arena_block_slots <= 64 )) || { echo "invalid arena block slots" >&2; exit 2; }
[[ "$resident_min_available_bytes" =~ ^[0-9]+$ ]] || { echo "invalid resident reserve" >&2; exit 2; }
if [[ "$execution" == resident_fused || "$execution" == resident_uva ]]; then
    [[ "$cache_storage" == arena && "$cache_experts" == 256 && -n "$rank_pack_basename" ]] || {
        echo "resident execution requires arena, cache256, and a rank pack" >&2
        exit 2
    }
fi
started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
[[ ! -e "$state" ]] || { echo "lazy candidate state exists" >&2; exit 2; }
mkdir -p "$state/ranks" "$logs"
printf '%s\n' "$started" > "$state/started-at.txt"
jq -n \
    --arg attempt "$attempt" \
    --argjson max_model_len "$max_model_len" \
    --argjson max_num_batched_tokens "$max_num_batched_tokens" \
    --argjson gpu_memory_utilization "$gpu_memory_utilization" \
    --argjson cache_experts_per_layer "$cache_experts" \
    --argjson sparse_mla_shadow_compare "$shadow_compare" \
    --argjson force_empty_think "$force_empty_think" \
    --arg cache_storage "$cache_storage" \
    --arg execution "$execution" \
    --arg rank_pack_basename "$rank_pack_basename" \
    --argjson kv_cache_memory_bytes "$kv_cache_memory_bytes" \
    --argjson cpu_offload_gb "$cpu_offload_gb" \
    --arg cpu_offload_params "$cpu_offload_params" \
    --argjson nccl_buffsize "$nccl_buffsize" \
    --argjson uva_resident "$uva_resident" \
    --arg cuda_allocator_conf "$cuda_allocator_conf" \
    --argjson arena_block_slots "$arena_block_slots" \
    --argjson resident_min_available_bytes "$resident_min_available_bytes" \
    '{schema:"glm53-full-exl3-tp3.lazy-external-attempt-config.v2",attempt:$attempt,max_model_len:$max_model_len,max_num_batched_tokens:$max_num_batched_tokens,gpu_memory_utilization:$gpu_memory_utilization,cache_experts_per_layer:$cache_experts_per_layer,cache_storage:$cache_storage,execution:$execution,rank_pack_basename:(if $rank_pack_basename=="" then null else $rank_pack_basename end),kv_cache_memory_bytes:$kv_cache_memory_bytes,cpu_offload_gb:$cpu_offload_gb,cpu_offload_params:(if $cpu_offload_params=="" then [] else ($cpu_offload_params|split(",")) end),nccl_buffsize:$nccl_buffsize,uva_resident:$uva_resident,cuda_allocator_conf:$cuda_allocator_conf,arena_block_slots:$arena_block_slots,resident_min_available_bytes:$resident_min_available_bytes,sparse_mla_shadow_compare:$sparse_mla_shadow_compare,force_empty_think:$force_empty_think}' \
    > "$state/CONFIG.json"

stop_all() {
    for rank in 0 1 2; do
        ssh -F zima-ssh-config "${nodes[$rank]}" \
            "docker stop --time 60 '$prefix-rank$rank' >/dev/null 2>&1 || true; \
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
        scp -q -F zima-ssh-config "$node:$remote/result.json" "$state/ranks/rank-$rank.json" 2>/dev/null || true
        scp -q -F zima-ssh-config "$node:$remote/FAILED.json" "$state/ranks/rank-$rank.FAILED.json" 2>/dev/null || true
        scp -q -F zima-ssh-config "$node:$remote/watchdog/metrics.csv" "$state/ranks/rank-$rank.metrics.csv" 2>/dev/null || true
        scp -q -F zima-ssh-config "$node:$remote/watchdog/STOP.json" "$state/ranks/rank-$rank-watchdog-STOP.json" 2>/dev/null || true
        scp -q -F zima-ssh-config "$node:$remote/prelaunch-page-cache.json" "$state/ranks/rank-$rank.prelaunch-page-cache.json" 2>/dev/null || true
    done
}

fail() {
    code=$?
    trap - EXIT
    failed_at=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    collect
    stop_all
    python3 scripts/kernel_audit.py --since "$started" --until "$failed_at" --output "$state/kernel-audit.json" || true
    jq -n --arg at "$failed_at" --argjson code "$code" '{schema:"glm53-full-exl3-tp3.lazy-external-failure.v1",failed_at:$at,exit_code:$code,candidates_stopped:true,automatic_service_restore_attempted:false}' > "$state/FAILED.json.tmp"
    mv "$state/FAILED.json.tmp" "$state/FAILED.json"
    exit "$code"
}
trap fail EXIT

for rank in 0 1 2; do
    node=${nodes[$rank]}
    [[ "$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '${services[$rank]}'")" == false ]]
    available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    (( available >= 12 * 1024 * 1024 * 1024 ))
    ssh -F zima-ssh-config "$node" "test -s '/home/mj-kang/Dev/models/$replica_basename/ASSEMBLY_COMPLETE.json'; ! docker ps --format '{{.Names}}' | grep -Eq '^(glm53-k3-|glm53-tp3-k3-uneven|glm53-lazy-runtime-gate)'; ! docker inspect '$prefix-rank$rank' >/dev/null 2>&1"
    if [[ -n "$rank_pack_basename" ]]; then
        remote_preflight="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/memory-designs/$attempt/rank-$rank"
        ssh -F zima-ssh-config "$node" \
            "mkdir -p '$remote_preflight'; \
             jq -e '.passed == true and .rank == $rank and .layers == 76' '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename/rank-$rank/COMPLETE.json' >/dev/null; \
             python3 '$project/scripts/evict_model_page_cache.py' \
               --checkpoint-dir '/home/mj-kang/Dev/models/$replica_basename' \
               --rank-pack-dir '/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename/rank-$rank' \
               --output '$remote_preflight/prelaunch-page-cache.json'"
    fi
    rsync -a --exclude .git/ --exclude .venv/ --exclude __pycache__/ --exclude outputs/ --exclude logs/ --exclude state/ "$project/" -e "ssh -F zima-ssh-config" "$node:$project/"
done
python3 scripts/kernel_audit.py --since "$started" --output "$state/prelaunch-kernel-audit.json"

for rank in 0 1 2; do
    node=${nodes[$rank]}
    remote="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/memory-designs/$attempt/rank-$rank"
    rank_pack_args=
    if [[ -n "$rank_pack_basename" ]]; then
        rank_pack_host="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$rank_pack_basename/rank-$rank"
        rank_pack_args="-e GLM53_LAZY_K3_PACKED_ROOT=/rank-packs -v $rank_pack_host:/rank-packs:ro"
    fi
    ssh -F zima-ssh-config "$node" "mkdir -p '$remote/watchdog' '/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/memory-designs'; \
	docker run -d --name '$prefix-rank$rank' --gpus all --network host --ipc host --shm-size 16g \
 --device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
 -e PYTHONPATH=/runtime:/workspace -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
 -e VLLM_GLM53_EXL3_TP3_FULL_MODEL=1 -e VLLM_GLM53_EXL3_TP3_RANK_SLICES=0 \
 -e VLLM_GLM53_EXL3_LAZY_K3=1 -e GLM53_LAZY_K3_MODEL_DIR=/model \
	 -e GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER='$cache_experts' \
	 -e GLM53_LAZY_K3_CACHE_STORAGE='$cache_storage' \
	 -e GLM53_LAZY_K3_EXECUTION='$execution' -e GLM53_LAZY_K3_ARENA_BLOCK_SLOTS='$arena_block_slots' \
	 -e GLM53_LAZY_K3_UVA='$uva_resident' \
	 -e GLM53_RESIDENT_MIN_AVAILABLE_BYTES='$resident_min_available_bytes' \
	 -e KV_CACHE_MEMORY_BYTES='$kv_cache_memory_bytes' \
	 -e CPU_OFFLOAD_GB='$cpu_offload_gb' \
	 -e CPU_OFFLOAD_PARAMS='$cpu_offload_params' \
	 -e CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}" \
	 $rank_pack_args \
 -e GLM53_TP3_SPARSE_MLA_SHADOW_COMPARE='$shadow_compare' \
 -e VLLM_GLM53_EXL3_TP3_GEOMETRY_ID=rotating-uneven-768-640-640-v1 \
 -e EXL3_FUSED_MOE=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
	 -e PYTORCH_CUDA_ALLOC_CONF='$cuda_allocator_conf' \
 -e MAX_MODEL_LEN='$max_model_len' -e MAX_NUM_BATCHED_TOKENS='$max_num_batched_tokens' -e GPU_MEMORY_UTILIZATION='$gpu_memory_utilization' -e SMOKE_MAX_OUTPUT_TOKENS=16 -e SMOKE_FORCE_EMPTY_THINK='$force_empty_think' \
 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none \
 -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
 -e NCCL_SOCKET_IFNAME=enP7s7 -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_DEBUG=INFO \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
 -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=1 -e NCCL_BUFFSIZE='$nccl_buffsize' \
 -e MASTER_ADDR='$head_ip' -e MASTER_PORT='$master_port' -e WORLD_SIZE=3 -e RANK='$rank' -e LOCAL_RANK=0 \
 -e MODEL_DIR=/model -e OUTPUT_JSON=/output/result.json \
 -v '/home/mj-kang/Dev/models/$replica_basename:/model:ro' -v '$project:/workspace:ro' -v '$project/runtime:/runtime:ro' -v '$remote:/output' \
 --entrypoint python3 glm53-exl3:e2-c190db1 /runtime/external_candidate_smoke.py >/dev/null; \
tmux new-session -d -s '$prefix-watchdog-$rank' \"python3 '$project/scripts/watchdog.py' --container '$prefix-rank$rank' --state-dir '$remote/watchdog' --reserve-gib 12 --interval 2 > '$remote/watchdog/watchdog.log' 2>&1\"; \
tmux new-session -d -s '$prefix-log-$rank' \"docker logs -f '$prefix-rank$rank' > '/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/memory-designs/$attempt-rank-$rank.log' 2>&1\"" &
done
wait

deadline=$(( $(date +%s) + 3600 ))
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
    (( $(date +%s) < deadline )) || { echo "lazy full candidate timeout" >&2; exit 2; }
    sleep 5
done

ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
collect
stop_all
python3 scripts/kernel_audit.py --since "$started" --until "$ended" --output "$state/kernel-audit.json"
python3 scripts/verify_lazy_external_candidate_smoke.py --state "$state" --logs "$logs" --output "$state/SUMMARY.json"
printf '%s\n' "$ended" > "$state/ended-at.txt"
touch "$state/PASSED"
trap - EXIT
