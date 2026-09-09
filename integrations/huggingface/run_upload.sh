#!/usr/bin/env bash
# Zima only. Start inside tmux; closing an SSH client does not stop the upload.
set -euo pipefail
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3-repro
state=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/hf-public-20260908
mkdir -p "$state"
exec 9>"$state/controller.lock"
flock -n 9 || { echo 'Upload controller already running'; exit 2; }
export HF_XET_CACHE=/home/mj-kang/Dev/cache/glm53-hf-xet-upload-20260908
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_SHARD_CACHE_SIZE_LIMIT=1073741824
export HF_XET_HIGH_PERFORMANCE=0
export HF_XET_NUM_CONCURRENT_RANGE_GETS=2
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
exec /home/mj-kang/Dev/cache/hf-model-upload-20260908-venv/bin/python \
  "$project/integrations/huggingface/upload_models.py" "$@" >>"$state/upload.log" 2>&1
