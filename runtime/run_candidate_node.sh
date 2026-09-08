#!/usr/bin/env bash
set -euo pipefail

required=(MODEL_DIR SERVED_MODEL_NAME PORT NODE_RANK HEAD_IP MASTER_PORT MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS KV_CACHE_DTYPE)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then echo "missing required environment: $name" >&2; exit 2; fi
done
enforce_eager=${ENFORCE_EAGER:-1}
graph_mode=${CUDA_GRAPH_MODE:-}
[[ -z "$graph_mode" || "$graph_mode" == FULL_DECODE_ONLY ]] || { echo "unsupported graph mode" >&2; exit 2; }
if [[ "${GLM53_SPINWAIT_MS:-stock}" != stock ]]; then
    # Use the image's source-validated patch before the serving interpreter
    # imports vLLM. -S avoids the experiment's sitecustomize in this helper.
    python3 -S /opt/glm53/patch_spinwait.py --preflight
    python3 -S /opt/glm53/patch_spinwait.py
fi
spec_method=${SPEC_METHOD:-}
spec_tokens=${SPEC_TOKENS:-}
spec_draft_tp=${SPEC_DRAFT_TP:-}
dcp_size=${DCP_SIZE:-1}
cp_kv_interleave_size=${CP_KV_INTERLEAVE_SIZE:-1}
[[ "$enforce_eager" == 0 || "$enforce_eager" == 1 ]] || { echo "ENFORCE_EAGER must be 0 or 1" >&2; exit 2; }
[[ "$dcp_size" =~ ^[0-9]+$ ]] && (( dcp_size >= 1 && dcp_size <= 3 )) || { echo "DCP_SIZE must be 1..3" >&2; exit 2; }
[[ "$cp_kv_interleave_size" =~ ^[0-9]+$ ]] && (( cp_kv_interleave_size >= 1 && cp_kv_interleave_size <= 64 )) || { echo "CP_KV_INTERLEAVE_SIZE must be 1..64" >&2; exit 2; }
if [[ -n "$spec_method" ]]; then
    [[ "$spec_tokens" =~ ^[0-9]+$ ]] && (( spec_tokens >= 1 && spec_tokens <= 8 )) || { echo "SPEC_TOKENS must be 1..8 when SPEC_METHOD is set" >&2; exit 2; }
fi
if [[ -n "$spec_draft_tp" ]]; then
    [[ "$spec_draft_tp" == 1 || "$spec_draft_tp" == 3 ]] || { echo "SPEC_DRAFT_TP must be 1 or 3" >&2; exit 2; }
    [[ "$spec_method" == mtp && "$spec_tokens" =~ ^[0-9]+$ ]] || { echo "SPEC_DRAFT_TP requires SPEC_METHOD=mtp and SPEC_TOKENS" >&2; exit 2; }
fi

args=(
    serve "$MODEL_DIR"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size 3
    --pipeline-parallel-size 1
    --decode-context-parallel-size "$dcp_size"
    --cp-kv-cache-interleave-size "$cp_kv_interleave_size"
    --nnodes 3
    --node-rank "$NODE_RANK"
    --master-addr "$HEAD_IP"
    --master-port "$MASTER_PORT"
    --distributed-executor-backend mp
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --no-enable-flashinfer-autotune
    --quantization exl3
    --load-format safetensors
    --dtype bfloat16
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-logprobs -1
    --chat-template "$MODEL_DIR/chat_template.jinja"
    --no-enable-prefix-caching
)
if [[ "$enforce_eager" == 1 ]]; then args+=(--enforce-eager); fi
if [[ -n "$graph_mode" ]]; then
    [[ "$enforce_eager" == 0 ]] || { echo "graph mode requires graphs enabled" >&2; exit 2; }
    args+=(--compilation-config "{\"cudagraph_mode\":\"$graph_mode\"}")
fi
if [[ -n "$spec_draft_tp" ]]; then
    args+=(--speculative-config "{\"method\":\"$spec_method\",\"num_speculative_tokens\":$spec_tokens,\"draft_tensor_parallel_size\":$spec_draft_tp}")
elif [[ -n "$spec_method" ]]; then
    args+=(--spec-method "$spec_method" --spec-tokens "$spec_tokens")
fi
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES:-0}
if [[ ! "$kv_cache_memory_bytes" =~ ^[0-9]+$ ]]; then
    echo "KV_CACHE_MEMORY_BYTES must be a non-negative integer" >&2
    exit 2
fi
if (( kv_cache_memory_bytes > 0 )); then
    args+=(--kv-cache-memory-bytes "$kv_cache_memory_bytes")
fi
if [[ "$NODE_RANK" != "0" ]]; then args+=(--headless); fi

echo "GLM53_K3_CANDIDATE_LAUNCH rank=$NODE_RANK model=$MODEL_DIR tp=3 dcp=$dcp_size pp=1 context=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS batched=$MAX_NUM_BATCHED_TOKENS kv=$KV_CACHE_DTYPE kv_bytes=$kv_cache_memory_bytes eager=$enforce_eager mtp=${spec_method:-0}:${spec_tokens:-0} draft_tp=${spec_draft_tp:-0} cp_interleave=$cp_kv_interleave_size prefix_cache=0 max_logprobs=all" >&2
exec vllm "${args[@]}"
