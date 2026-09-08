#!/usr/bin/env bash
# Run on Zima. Read-only current-attempt check; no generation or restart.
set -euo pipefail
attempt=${1:-speed-qualified-final}
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || exit 2
cd /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
image=sha256:b7ff496564001ee02655ce796370bf823a233a963f04c3d334a549bb5f0628e5
model=GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z
pack=20260907T231500Z-k275-mixed-rank-local-v1
for rank in 0 1 2; do
  node=mj-spark-$((rank+1))
  name=glm53-k3-cand-$attempt-rank$rank
  ssh -F zima-ssh-config "$node" "docker inspect '$name'" |
    jq -e --arg image "$image" --arg model "/home/mj-kang/Dev/models/$model" \
      --arg pack "/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/$pack/rank-$rank" '
      .[0] | .State.Running == true and .State.OOMKilled == false and .Image == $image
      and any(.Mounts[]; .Destination == "/model" and .Source == $model and .RW == false)
      and any(.Mounts[]; .Destination == "/rank-packs" and .Source == $pack and .RW == false)
      and (.Config.Env | contains(["GLM53_LAZY_K3_EXECUTION=resident_uva",
        "GLM53_LAZY_K3_UVA=1", "MAX_MODEL_LEN=32768", "MAX_NUM_SEQS=1",
        "SPEC_METHOD=mtp", "SPEC_TOKENS=3", "SPEC_DRAFT_TP=1", "DCP_SIZE=3",
        "KV_CACHE_MEMORY_BYTES=1073741824", "KV_CACHE_DTYPE=fp8",
        "MAX_NUM_BATCHED_TOKENS=128", "ENFORCE_EAGER=1", "CUDA_GRAPH_MODE=",
        "GLM53_SPINWAIT_MS=stock", "NCCL_MIN_NCHANNELS=4", "NCCL_MAX_NCHANNELS=4",
        "NCCL_BUFFSIZE=1048576", "GLM53_RESIDENT_MIN_AVAILABLE_BYTES=12884901888"]))' >/dev/null
  ssh -F zima-ssh-config "$node" "test ! -e '/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260907T224500Z/candidate/attempts/$attempt/rank-$rank/watchdog/STOP.json' && tmux has-session -t 'glm53-k3-cand-$attempt-watchdog-$rank'"
  echo "$node: running, expected image/mounts/profile, watchdog session present, no STOP receipt"
done
curl -fsS --max-time 10 http://192.168.0.238:8893/health >/dev/null
curl -fsS --max-time 10 http://192.168.0.238:8893/v1/models |
  jq -e 'any(.data[]; .id == "GLM-5.3-K3-TP3-CANDIDATE" and .max_model_len == 32768)' >/dev/null
echo 'LIVE_PROFILE_PASS (not a new generation, memory-window, or quality qualification)'
