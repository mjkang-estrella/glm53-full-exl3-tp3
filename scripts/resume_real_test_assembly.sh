#!/usr/bin/env bash
set -euo pipefail

stamp=${1:-20260906T034145Z}
[[ "$stamp" == 20260906T034145Z ]] || { echo "this sealed runbook only supports 20260906T034145Z" >&2; exit 2; }
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp"
logs="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp"
incoming=/mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z.incomplete
final=/mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z
test ! -e "$final"
test -s /home/mj-kang/Dev/state/glm53-full-exl3-tp3/bulk/20260904T180633Z/COMPLETE.json
cd "$project"
mkdir -p "$state/assembly" "$logs"
exec python3 scripts/assemble_checkpoint.py \
  --source-root /mnt/unas-models/ZAI/GLM-5.3-BF16 \
  --source-inventory /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/reference-uniform-k3/evidence/source-inventory.json \
  --source-structure /home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/uneven-phase0-20260904T162819Z/source-structure.json \
  --tensor-plan /home/mj-kang/Dev/state/glm53-full-exl3-tp3/qualification/uneven-phase0-20260904T162819Z/tensor-plan-v2.json \
  --layer-archive /mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-20260904T180633Z \
  --archive-verification "$state/archive-verification.json" \
  --output-dir "$incoming" \
  --state-dir "$state/assembly"
