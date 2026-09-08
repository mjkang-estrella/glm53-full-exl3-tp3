#!/usr/bin/env bash
set -euo pipefail

checkpoint=${GLM53_WATCH_CHECKPOINT:-/home/mj-kang/Dev/models/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z}
packs_root=${GLM53_WATCH_RANK_PACK_ROOT:-/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/20260906T113200Z-rank-local-v1}
packs=$packs_root/rank-${1:?rank required}
container=${2:?container suffix required}-rank${1}
state=${GLM53_WATCH_STATE:-/home/mj-kang/Dev/state/glm53-full-exl3-tp3/page-cache-watch}
mkdir -p "$state"
while docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; do
  stamp=$(date -u +%Y%m%dT%H%M%S%3NZ)
  python3 /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/scripts/evict_model_page_cache.py \
    --checkpoint-dir "$checkpoint" --rank-pack-dir "$packs" \
    --output "$state/rank-${1}-${stamp}.json" >/dev/null 2>&1 || true
  sleep 4
done
