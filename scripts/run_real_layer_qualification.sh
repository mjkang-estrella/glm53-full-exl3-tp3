#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
stamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
rollback_stamp=${2:?rollback inventory stamp required}
runtime_gate_stamp=${3:?passing runtime gate stamp required}
layer=3
node=mj-spark-3
geometry=rotating-uneven-768-640-640-v1
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3
remote_state="$state_root/qualification/real-uneven-layer-$stamp"
central="$state_root/qualification/real-uneven-layer-$stamp"
source_inventory=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/reference-uniform-k3/evidence/source-inventory.json

jq -e '.passed == true and .shard_count == 282' "$state_root/manifests/source-sha256.json" >/dev/null
jq -e '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1" and (.rotations | length == 3)' \
    "$state_root/qualification/runtime-gate-$runtime_gate_stamp/SUMMARY.json" >/dev/null
test -s "$state_root/qualification/runtime-gate-$runtime_gate_stamp/PASSED"
test -s "$state_root/rollback/$rollback_stamp/mj-zima/spark-summaries/mj-spark-3.rollback-commands.sh"
ssh -F zima-ssh-config "$node" \
    "test -s '$state_root/rollback/$rollback_stamp/$node/containers/minimax-h3-comfy.inspect.json'"

mkdir -p "$central/remote-layer" "$central/rotation-experts" "$central/logs" "$central/worker-state"
rsync -a --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" \
    "$node:$project/"
ssh -F zima-ssh-config "$node" "mkdir -p '$state_root' '$remote_state'"
rsync -a "$source_inventory" -e "ssh -F zima-ssh-config" \
    "$node:$state_root/source-inventory.json"

# Stage only the shards needed for one expert at all three rotations and the
# complete layer-3 pass. Every staged file is SHA-256 verified before H3 stops.
for rotation_layer in 3 4 5; do
    bash scripts/stage_layer.sh "$node" "$rotation_layer"
done

restored=0
restore_h3() {
    if (( restored )); then return; fi
    restored=1
    ssh -F zima-ssh-config "$node" 'docker start minimax-h3-comfy >/dev/null 2>&1 || true'
}
trap restore_h3 EXIT
ssh -F zima-ssh-config "$node" 'docker stop --time 60 minimax-h3-comfy >/dev/null'
if [ "$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' minimax-h3-comfy")" != false ]; then
    echo 'H3 did not stop for bounded qualification' >&2
    exit 2
fi
encoder_start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$encoder_start" > "$central/encoder-window-start.txt"

run_remote_encoder() {
    local target_layer=$1
    local experts=$2
    local label=$3
    local session="glm53-k3-qual-${stamp}-${label}"
    local exit_path="$remote_state/${label}.exit-code"
    ssh -F zima-ssh-config "$node" "test ! -e '$exit_path'"
    ssh -F zima-ssh-config "$node" \
        "tmux new-session -d -s '$session' \"set +e; bash '$project/scripts/run_encoder.sh' '$target_layer' '$experts' '${stamp}-${label}'; code=\\\$?; printf '%s\\n' \\\"\\\$code\\\" > '${exit_path}.tmp'; mv '${exit_path}.tmp' '$exit_path'\""
    while ! ssh -F zima-ssh-config "$node" "test -s '$exit_path'"; do
        sleep 10
    done
    code=$(ssh -F zima-ssh-config "$node" "cat '$exit_path'")
    if [ "$code" != 0 ]; then
        echo "remote encoder $label failed with exit $code" >&2
        return 1
    fi
}

# Qualify one real expert at every rotation. The layer-3 complete pass then
# validates and skips its already sealed expert rather than re-encoding it.
for rotation_layer in 3 4 5; do
    label="rotation-L$(printf '%03d' "$rotation_layer")-expert000"
    run_remote_encoder "$rotation_layer" 0 "$label"
    remote_receipt="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/encoded/$geometry/layer-$(printf '%03d' "$rotation_layer")/receipts/expert-000.json"
    ssh -F zima-ssh-config "$node" \
        "jq -e '.passed == true and .geometry_id == \"$geometry\" and .expert == 0 and (.slices | length == 9) and .padding_channels == 0' '$remote_receipt' >/dev/null"
    mkdir -p "$central/rotation-experts/layer-$(printf '%03d' "$rotation_layer")"
    scp -q -F zima-ssh-config "$node:$remote_receipt" \
        "$central/rotation-experts/layer-$(printf '%03d' "$rotation_layer")/expert-000.json"
done
run_remote_encoder "$layer" 0-255 full-layer

remote_layer="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/encoded/$geometry/layer-003"
ssh -F zima-ssh-config "$node" \
    "python3 '$project/scripts/verify_layer.py' --layer-dir '$remote_layer' --output '$remote_state/layer-verification.json'"

encoder_end=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$encoder_end" > "$central/encoder-window-end.txt"
python3 scripts/kernel_audit.py --since "$encoder_start" --until "$encoder_end" \
    --output "$central/encoder-window-kernel-audit.json"

rsync -a -e "ssh -F zima-ssh-config" "$node:$remote_layer/receipts/" "$central/remote-layer/receipts/"
scp -q -F zima-ssh-config "$node:$remote_layer/LAYER_RECEIPT.json" "$central/remote-layer/LAYER_RECEIPT.json"
scp -q -F zima-ssh-config "$node:$remote_state/layer-verification.json" "$central/layer-verification.json"
for rotation_layer in 3 4 5; do
    label="rotation-L$(printf '%03d' "$rotation_layer")-expert000"
    scp -q -F zima-ssh-config "$node:$remote_state/$label.exit-code" "$central/$label.exit-code"
done
scp -q -F zima-ssh-config "$node:$remote_state/full-layer.exit-code" "$central/full-layer.exit-code"
scp -q -F zima-ssh-config \
    "$node:/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/$geometry-layer-003-${stamp}-rotation-L003-expert000.log" \
    "$central/logs/rotation-L003-expert000.log"
for rotation_layer in 4 5; do
    scp -q -F zima-ssh-config \
        "$node:/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/$geometry-layer-$(printf '%03d' "$rotation_layer")-${stamp}-rotation-L$(printf '%03d' "$rotation_layer")-expert000.log" \
        "$central/logs/rotation-L$(printf '%03d' "$rotation_layer")-expert000.log"
done
scp -q -F zima-ssh-config \
    "$node:/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/$geometry-layer-003-${stamp}-full-layer.log" \
    "$central/logs/full-layer.log"
for rotation_layer in 3 4 5; do
    rsync -a -e "ssh -F zima-ssh-config" \
        "$node:$state_root/workers/$geometry/layer-$(printf '%03d' "$rotation_layer")-${stamp}-rotation-L$(printf '%03d' "$rotation_layer")-expert000/" \
        "$central/worker-state/rotation-L$(printf '%03d' "$rotation_layer")-expert000/"
done
rsync -a -e "ssh -F zima-ssh-config" \
    "$node:$state_root/workers/$geometry/layer-003-${stamp}-full-layer/" "$central/worker-state/full-layer/"

h3_restore_start=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$h3_restore_start" > "$central/h3-restore-start.txt"
restore_h3
trap - EXIT
for _ in $(seq 1 120); do
    h3_code=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' \
        http://192.168.0.234:8188/system_stats || true)
    [ "$h3_code" = 200 ] && break
    sleep 5
done
printf '%s\n' "$h3_code" > "$central/restored-h3-health.txt"
[ "$h3_code" = 200 ]
sleep 10
h3_restore_end=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
printf '%s\n' "$h3_restore_end" > "$central/h3-restore-end.txt"
python3 scripts/kernel_audit.py --since "$h3_restore_start" --until "$h3_restore_end" \
    --output "$central/h3-restore-kernel-audit.json"
date -u +%FT%TZ > "$central/PASSED"
printf '%s\n' "$central"
