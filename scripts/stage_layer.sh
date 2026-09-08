#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
node=${1:?node required}
layer=${2:?layer required}
case "$node" in mj-spark-1|mj-spark-2|mj-spark-3) ;; *) echo "invalid node" >&2; exit 2;; esac
if (( layer < 3 || layer > 78 )); then echo "layer must be 3..78" >&2; exit 2; fi

source_root=/mnt/unas-models/ZAI/GLM-5.3-BF16
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3
remote_root="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/source/layer-$(printf '%03d' "$layer")"
list="$state_root/staging/layer-$(printf '%03d' "$layer").files"
mkdir -p "$(dirname "$list")"
python3 scripts/layer_shards.py --index "$source_root/model.safetensors.index.json" --layer "$layer" > "$list.tmp"
printf '%s\n' model.safetensors.index.json >> "$list.tmp"
sort -u "$list.tmp" > "$list"
rm -f "$list.tmp"

ssh -F zima-ssh-config "$node" "mkdir -p '$remote_root'"
rsync -a --partial --append-verify --files-from="$list" -e "ssh -F zima-ssh-config" "$source_root/" "$node:$remote_root/"

manifest=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/manifests/source-structure.json
python3 - "$manifest" "$list" > "$list.sha256" <<'PY'
import json, pathlib, sys
doc=json.loads(pathlib.Path(sys.argv[1]).read_text())
for name in pathlib.Path(sys.argv[2]).read_text().splitlines():
    if name == "model.safetensors.index.json":
        digest=doc["index_sha256"]
    else:
        digest=doc["shards"][name]["lfs_sha256"]
    print(f"{digest}  {name}")
PY
scp -q -F zima-ssh-config "$list.sha256" "$node:$remote_root/STAGED_SHA256SUMS"
ssh -F zima-ssh-config "$node" "cd '$remote_root' && sha256sum -c STAGED_SHA256SUMS && date -u +%FT%TZ > STAGED_OK"
printf '%s\n' "$remote_root"
