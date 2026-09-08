#!/usr/bin/env bash
# Zima: restore the unchanged K275 test profile after offline work exits.
set -euo pipefail
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
for node in mj-spark-1 mj-spark-2 mj-spark-3; do
  active=$(ssh -F zima-ssh-config "$node" "docker ps --format '{{.Names}}'")
  if grep -Eq '^(glm53-k3-cand-|glm53-k275-reencode-|glm53-tp3-k[23]-)' <<< "$active"; then
    echo "experimental GPU process still active on $node" >&2; exit 2
  fi
done
export GLM53_RESIDENT_MIN_AVAILABLE_BYTES=12884901888
export GLM53_WATCHDOG_RESERVE_GIB=12
export GLM53_LAZY_K3_UVA=1
export GLM53_LAZY_MAX_NUM_BATCHED_TOKENS=128
export GLM53_ENFORCE_EAGER=1
export GLM53_DCP_SIZE=3
export GLM53_CP_KV_INTERLEAVE_SIZE=1
export GLM53_LAZY_GPU_MEMORY_UTILIZATION=0.12
export GLM53_SPEC_METHOD=''
export GLM53_SPEC_TOKENS=''
export GLM53_SPEC_DRAFT_TP=''
exec bash scripts/start_candidate_cluster_attempt.sh \
  20260907T224500Z \
  GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z \
  k275-post-reencode-v1 1 524288 lazy 256 arena 3600 resident_uva 1812613120 \
  20260907T231500Z-k275-mixed-rank-local-v1 expandable_segments:True 16
