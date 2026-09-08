#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_stamp=${1:?bulk run stamp required}
layer=${2:?layer required}
node=${3:?node required}
lease_id=${4:?lease id required}
publish_root=${5:?UNAS publish root required}

[[ "$run_stamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid run stamp" >&2; exit 2; }
[[ "$lease_id" =~ ^[0-9a-f]{32}$ ]] || { echo "invalid lease id" >&2; exit 2; }
case "$node" in mj-spark-1|mj-spark-2|mj-spark-3) ;; *) echo "invalid node" >&2; exit 2;; esac
(( layer >= 3 && layer <= 78 )) || { echo "invalid layer" >&2; exit 2; }
case "$publish_root" in /mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-*) ;;
    *) echo "invalid publish root" >&2; exit 2;;
esac

geometry=rotating-uneven-768-640-640-v1
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3
source_inventory=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/reference-uniform-k3/evidence/source-inventory.json
bulk_state="$state_root/bulk/$run_stamp"
result="$bulk_state/queue/results/layer-$(printf '%03d' "$layer")-$lease_id.json"
progress="$bulk_state/worker-progress/layer-$(printf '%03d' "$layer")-$lease_id.json"
remote_state="$bulk_state/remote/$node/layer-$(printf '%03d' "$layer")-$lease_id"
remote_layer="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/encoded/$geometry/layer-$(printf '%03d' "$layer")"
remote_source="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/source/layer-$(printf '%03d' "$layer")"
remote_exit="$remote_state/exit-code.txt"
remote_verification="$remote_state/verification.json"
session="glm53-k3-${run_stamp:9:6}-L$(printf '%03d' "$layer")-${lease_id:0:8}"
encoder_stamp="${run_stamp}-bulk-${lease_id}"
target="$publish_root/layer-$(printf '%03d' "$layer")"
incoming="$publish_root/.incoming-layer-$(printf '%03d' "$layer")-$lease_id"
started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
phase=bootstrap
expert_receipts=0

mkdir -p "$(dirname "$result")" "$(dirname "$progress")" "$publish_root"
write_progress() {
    local new_phase=$1
    local detail=${2:-}
    local now tmp
    phase=$new_phase
    now=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    tmp="${progress}.tmp.$$"
    jq -n --arg run_stamp "$run_stamp" --argjson layer "$layer" \
        --arg node "$node" --arg lease_id "$lease_id" --arg phase "$phase" \
        --arg detail "$detail" --arg started "$started" --arg updated "$now" \
        --argjson expert_receipts "$expert_receipts" \
        '{schema:"glm53-full-exl3-tp3.bulk-worker-progress.v1",run_stamp:$run_stamp,
          layer:$layer,node:$node,lease_id:$lease_id,phase:$phase,detail:$detail,
          expert_receipts:$expert_receipts,started_at:$started,updated_at:$updated}' \
        > "$tmp"
    chmod 644 "$tmp"
    mv "$tmp" "$progress"
}
write_result() {
    local passed=$1 code=$2 reason=$3
    local ended tmp
    ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    tmp="${result}.tmp.$$"
    jq -n --argjson passed "$passed" --argjson code "$code" \
        --arg reason "$reason" --arg run_stamp "$run_stamp" \
        --argjson layer "$layer" --arg node "$node" --arg lease_id "$lease_id" \
        --arg started "$started" --arg ended "$ended" --arg target "$target" \
        '{schema:"glm53-full-exl3-tp3.bulk-worker-result.v1",passed:$passed,
          exit_code:$code,reason:$reason,run_stamp:$run_stamp,layer:$layer,node:$node,
          lease_id:$lease_id,started_at:$started,ended_at:$ended,publish_path:$target}' \
        > "$tmp"
    chmod 644 "$tmp"
    mv "$tmp" "$result"
}
failure() {
    code=$?
    trap - EXIT HUP INT TERM
    # An EXIT trap reached with zero is never a passing worker result. This
    # specifically prevents an external TERM during a wait from producing the
    # historical, misleading `passed:false, exit_code:0` receipt.
    if (( code == 0 )); then code=1; fi
    write_progress failed "exit_${code}_phase_${phase}"
    if [ ! -e "$result" ]; then
        write_result false "$code" "bulk_lease_worker_exit_${code}_phase_${phase}"
    fi
    exit "$code"
}
terminated() {
    local code=$1 signal_name=$2
    write_progress stopping "received_${signal_name}"
    exit "$code"
}
trap failure EXIT
trap 'terminated 129 HUP' HUP
trap 'terminated 130 INT' INT
trap 'terminated 143 TERM' TERM

test ! -e "$result"
test -w /mnt/unas-models/ZAI
test -s "$source_inventory"
write_progress syncing_project
rsync -a --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" "$node:$project/"
rsync -a "$source_inventory" -e "ssh -F zima-ssh-config" \
    "$node:$state_root/source-inventory.json"
write_progress staging_source
bash scripts/stage_layer.sh "$node" "$layer"
ssh -F zima-ssh-config "$node" "mkdir -p '$remote_state'"

if ! ssh -F zima-ssh-config "$node" "test -s '$remote_exit'"; then
    write_progress starting_or_resuming_encoder
    if ! ssh -F zima-ssh-config "$node" "tmux has-session -t '$session' 2>/dev/null"; then
        ssh -F zima-ssh-config "$node" \
            "tmux new-session -d -s '$session' \"set +e; bash '$project/scripts/run_encoder.sh' '$layer' 0-255 '$encoder_stamp'; code=\\\$?; printf '%s\\n' \\\"\\\$code\\\" > '${remote_exit}.tmp'; mv '${remote_exit}.tmp' '$remote_exit'\""
    fi
fi

write_progress encoding_or_receipt_validation
while ! ssh -F zima-ssh-config "$node" "test -s '$remote_exit'"; do
    expert_receipts=$(ssh -F zima-ssh-config "$node" \
        "find '$remote_layer/receipts' -maxdepth 1 -type f -name 'expert-*.json' 2>/dev/null | wc -l")
    write_progress encoding_or_receipt_validation
    sleep 20
done
remote_code=$(ssh -F zima-ssh-config "$node" "cat '$remote_exit'")
[ "$remote_code" = 0 ] || { echo "$node layer $layer encoder exit $remote_code" >&2; exit 1; }

expert_receipts=256
write_progress verifying_remote_layer
ssh -F zima-ssh-config "$node" \
    "python3 '$project/scripts/verify_layer.py' --layer-dir '$remote_layer' --output '$remote_verification'"
ssh -F zima-ssh-config "$node" \
    "jq -e '.passed == true and .geometry_id == \"$geometry\" and .layer == $layer and .experts == 256' '$remote_verification' >/dev/null"

if [ -d "$target" ]; then
    write_progress verifying_existing_publication
    (cd "$target" && sha256sum -c ARTIFACT_SHA256SUMS >/dev/null)
    jq -e --argjson layer "$layer" \
        '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1" and .layer == $layer' \
        "$target/VERIFICATION.json" >/dev/null
else
    # Reuse a same-lease partial transfer after a Zima manager interruption.
    write_progress transferring_to_unas
    mkdir -p "$incoming"
    rsync -a --partial -e "ssh -F zima-ssh-config" "$node:$remote_layer/" "$incoming/"
    # Encoder files are deliberately root-owned and immutable on the Spark.
    # Add the independent verifier and publication ledger only in Zima's
    # incoming UNAS directory; never mutate the sealed local layer.
    scp -q -F zima-ssh-config "$node:$remote_verification" "$incoming/VERIFICATION.json.tmp"
    mv "$incoming/VERIFICATION.json.tmp" "$incoming/VERIFICATION.json"
    write_progress checksumming_unas_incoming
    (cd "$incoming" && find . -type f ! -name ARTIFACT_SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > ARTIFACT_SHA256SUMS)
    (cd "$incoming" && sha256sum -c ARTIFACT_SHA256SUMS >/dev/null)
    jq -e --argjson layer "$layer" \
        '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1" and .layer == $layer' \
        "$incoming/VERIFICATION.json" >/dev/null
    write_progress publishing_atomic_rename
    mv "$incoming" "$target"
fi

write_progress writing_completion_receipt
receipt_sha=$(sha256sum "$target/LAYER_RECEIPT.json" | awk '{print $1}')
ledger_sha=$(sha256sum "$target/ARTIFACT_SHA256SUMS" | awk '{print $1}')
published_bytes=$(du -sb "$target" | awk '{print $1}')
ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
tmp="${result}.tmp.$$"
jq -n --arg run_stamp "$run_stamp" --argjson layer "$layer" --arg node "$node" \
    --arg lease_id "$lease_id" --arg started "$started" --arg ended "$ended" \
    --arg target "$target" --arg receipt_sha "$receipt_sha" --arg ledger_sha "$ledger_sha" \
    --argjson published_bytes "$published_bytes" \
    '{schema:"glm53-full-exl3-tp3.bulk-worker-result.v1",passed:true,exit_code:0,
      reason:"sealed_and_published",run_stamp:$run_stamp,layer:$layer,node:$node,
      lease_id:$lease_id,started_at:$started,ended_at:$ended,publish_path:$target,
      layer_receipt_sha256:$receipt_sha,artifact_ledger_sha256:$ledger_sha,
      published_bytes:$published_bytes}' > "$tmp"
chmod 644 "$tmp"
mv "$tmp" "$result"
write_progress published
trap - EXIT HUP INT TERM

# Only the bounded, verified local BF16 staging copy is removed. Encoded
# checkpoint data, qualification evidence, and the immutable UNAS source stay.
ssh -F zima-ssh-config "$node" "rm -rf -- '$remote_source'" || true
