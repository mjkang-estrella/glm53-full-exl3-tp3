#!/usr/bin/env bash
# Run on Zima after intentionally stopping any active GPU models.
set -euo pipefail
label=${1:?new unique attempt label required}
[[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]]
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
for node in mj-spark-1 mj-spark-2 mj-spark-3; do
  compute=$(ssh -F zima-ssh-config "$node" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader')
  [[ -z "$compute" ]] || { echo "$node has a GPU workload; stop it explicitly first" >&2; exit 2; }
done
export GLM53_RESIDENT_MIN_AVAILABLE_BYTES=12884901888
export GLM53_WATCHDOG_RESERVE_GIB=12
export GLM53_LAZY_K3_UVA=1
export GLM53_LAZY_MAX_NUM_BATCHED_TOKENS=128
export GLM53_ENFORCE_EAGER=1
export GLM53_CUDA_GRAPH_MODE=''
export GLM53_SPINWAIT_MS=stock
export GLM53_DCP_SIZE=3
export GLM53_CP_KV_INTERLEAVE_SIZE=1
export GLM53_LAZY_GPU_MEMORY_UTILIZATION=0.12
export GLM53_SPEC_METHOD=mtp
export GLM53_SPEC_TOKENS=3
export GLM53_SPEC_DRAFT_TP=1
exec bash scripts/start_candidate_cluster_attempt.sh \
  20260907T224500Z \
  GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z \
  "$label" 4 1048576 lazy 256 arena 3600 resident_uva 1073741824 \
  20260907T231500Z-k275-mixed-rank-local-v1 expandable_segments:True 16
