#!/usr/bin/env bash
set -euo pipefail

run_stamp=${1:?run stamp required}
layer=${2:?layer required}
node=${3:?node required}
lease_id=${4:?lease id required}
publish_root=${5:?publish root required}
experts=${6:?K2 expert list required}

[[ "$run_stamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid run stamp" >&2; exit 2; }
[[ "$lease_id" =~ ^[0-9a-f]{32}$ ]] || { echo "invalid lease id" >&2; exit 2; }
case "$node" in mj-spark-1|mj-spark-2|mj-spark-3) ;; *) echo "invalid node" >&2; exit 2;; esac
(( layer >= 3 && layer <= 78 )) || { echo "invalid layer" >&2; exit 2; }
case "$publish_root" in /mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2-rotating-uneven-v1-*) ;; *) echo "invalid publish root" >&2; exit 2;; esac

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3
source_inventory=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/reference-uniform-k3/evidence/source-inventory.json
bulk_state="$state_root/k275/bulk/$run_stamp"
result="$bulk_state/queue/results/layer-$(printf '%03d' "$layer")-$lease_id.json"
progress="$bulk_state/worker-progress/layer-$(printf '%03d' "$layer")-$lease_id.json"
remote_state="$bulk_state/remote/$node/layer-$(printf '%03d' "$layer")-$lease_id"
remote_exit="$remote_state/exit-code.txt"
remote_verification="$remote_state/verification.json"
geometry=rotating-uneven-768-640-640-v1
remote_layer="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/encoded/${geometry}-bits2/layer-$(printf '%03d' "$layer")"
remote_source="/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/source/layer-$(printf '%03d' "$layer")"
session="glm53-k275-${run_stamp:9:6}-L$(printf '%03d' "$layer")-${lease_id:0:8}"
encoder_stamp="${run_stamp}-bulk-${lease_id}"
target="$publish_root/layer-$(printf '%03d' "$layer")"
incoming="$publish_root/.incoming-layer-$(printf '%03d' "$layer")-$lease_id"
started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
expert_receipts=0
phase=bootstrap

mkdir -p "$(dirname "$result")" "$(dirname "$progress")" "$publish_root"
experts_json=$(printf '%s' "$experts" | jq -R 'split(",") | map(tonumber)')
[[ $(jq 'length' <<<"$experts_json") == 64 ]] || { echo "K2 selection must contain 64 experts" >&2; exit 2; }

write_progress() {
    local new_phase=$1 detail=${2:-} now tmp
    phase=$new_phase
    now=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    tmp="${progress}.tmp.$$"
    jq -n --arg run_stamp "$run_stamp" --argjson layer "$layer" \
        --arg node "$node" --arg lease_id "$lease_id" --arg phase "$phase" \
        --arg detail "$detail" --arg started "$started" --arg updated "$now" \
        --argjson expert_receipts "$expert_receipts" --argjson experts "$experts_json" \
        '{schema:"glm53-full-exl3-tp3.k275-worker-progress.v1",run_stamp:$run_stamp,
          layer:$layer,node:$node,lease_id:$lease_id,phase:$phase,detail:$detail,
          k2_experts:$experts,bits:2,expert_receipts:$expert_receipts,
          started_at:$started,updated_at:$updated}' > "$tmp"
    chmod 644 "$tmp"
    mv "$tmp" "$progress"
}

write_result() {
    local passed=$1 code=$2 reason=$3 ended tmp
    ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    tmp="${result}.tmp.$$"
    jq -n --argjson passed "$passed" --argjson code "$code" \
        --arg reason "$reason" --arg run_stamp "$run_stamp" \
        --argjson layer "$layer" --arg node "$node" --arg lease_id "$lease_id" \
        --arg started "$started" --arg ended "$ended" --arg target "$target" \
        --argjson experts "$experts_json" \
        '{schema:"glm53-full-exl3-tp3.k275-worker-result.v1",passed:$passed,
          exit_code:$code,reason:$reason,run_stamp:$run_stamp,layer:$layer,node:$node,
          lease_id:$lease_id,started_at:$started,ended_at:$ended,publish_path:$target,
          bits:2,k2_experts:$experts}' > "$tmp"
    chmod 644 "$tmp"
    mv "$tmp" "$result"
}

failure() {
    code=$?
    trap - EXIT HUP INT TERM
    if (( code == 0 )); then code=1; fi
    write_progress failed "exit_${code}_phase_${phase}"
    [ -e "$result" ] || write_result false "$code" "worker_exit_${code}_phase_${phase}"
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

if [ -d "$target" ] && [ -s "$target/LAYER_K2_RECEIPT.json" ] && \
   [ -s "$target/VERIFICATION.json" ] && [ -s "$target/ARTIFACT_SHA256SUMS" ]; then
    if (cd "$target" && sha256sum -c ARTIFACT_SHA256SUMS >/dev/null) && \
       jq -e --argjson layer "$layer" --argjson experts "$experts_json" \
          '.passed == true and .layer == $layer and .bits == 2 and .expected_experts == $experts' \
          "$target/VERIFICATION.json" >/dev/null; then
        published_bytes=$(du -sb "$target" | awk '{print $1}')
        ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
        jq -n --arg run_stamp "$run_stamp" --argjson layer "$layer" --arg node "$node" \
            --arg lease_id "$lease_id" --arg started "$started" --arg ended "$ended" \
            --arg target "$target" --arg receipt_sha "$(sha256sum "$target/LAYER_K2_RECEIPT.json" | awk '{print $1}')" \
            --arg ledger_sha "$(sha256sum "$target/ARTIFACT_SHA256SUMS" | awk '{print $1}')" \
            --argjson published_bytes "$published_bytes" --argjson experts "$experts_json" \
            '{schema:"glm53-full-exl3-tp3.k275-worker-result.v1",passed:true,exit_code:0,
              reason:"already_published",run_stamp:$run_stamp,layer:$layer,node:$node,
              lease_id:$lease_id,started_at:$started,ended_at:$ended,publish_path:$target,
              bits:2,k2_experts:$experts,layer_receipt_sha256:$receipt_sha,
              artifact_ledger_sha256:$ledger_sha,published_bytes:$published_bytes}' > "$result"
        write_progress published "reused_existing_layer"
        trap - EXIT HUP INT TERM
        exit 0
    fi
fi

write_progress syncing_project
rsync -a --exclude logs/ --exclude state/ ./ -e "ssh -F zima-ssh-config" "$node:$project/"
rsync -a "$source_inventory" -e "ssh -F zima-ssh-config" \
    "$node:$state_root/source-inventory.json"
write_progress staging_source
bash "$project/scripts/stage_layer.sh" "$node" "$layer"
ssh -F zima-ssh-config "$node" "mkdir -p '$remote_state'"

if ! ssh -F zima-ssh-config "$node" "test -s '$remote_exit'"; then
    write_progress starting_or_resuming_encoder
    if ! ssh -F zima-ssh-config "$node" "tmux has-session -t '$session' 2>/dev/null"; then
        ssh -F zima-ssh-config "$node" \
            "tmux new-session -d -s '$session' \"set +e; bash '$project/scripts/run_encoder.sh' '$layer' '$experts' '$encoder_stamp' 2; code=\\\$?; printf '%s\\n' \\\"\\\$code\\\" > '${remote_exit}.tmp'; mv '${remote_exit}.tmp' '$remote_exit'\""
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
[ "$remote_code" = 0 ] || { echo "$node layer $layer K2 encoder exit $remote_code" >&2; exit 1; }

expert_receipts=64
write_progress verifying_remote_subset
ssh -F zima-ssh-config "$node" \
    "python3 '$project/scripts/verify_k275_subset.py' --layer-dir '$remote_layer' --layer '$layer' --experts '$experts' --output '$remote_verification'"
ssh -F zima-ssh-config "$node" \
    "jq -e --argjson layer '$layer' --argjson experts '$experts_json' '.passed == true and .layer == \$layer and .bits == 2 and .expected_experts == \$experts' '$remote_verification' >/dev/null"

write_progress transferring_to_unas
mkdir -p "$incoming"
mkdir -p "$incoming/experts" "$incoming/receipts"
for expert in $(jq -r '.[]' <<<"$experts_json"); do
    rsync -a --partial -e "ssh -F zima-ssh-config" \
        "$node:$remote_layer/experts/expert-$(printf '%03d' "$expert").safetensors" \
        "$incoming/experts/"
    rsync -a --partial -e "ssh -F zima-ssh-config" \
        "$node:$remote_layer/receipts/expert-$(printf '%03d' "$expert").json" \
        "$incoming/receipts/"
done
scp -q -F zima-ssh-config "$node:$remote_verification" "$incoming/VERIFICATION.json.tmp"
mv "$incoming/VERIFICATION.json.tmp" "$incoming/VERIFICATION.json"
layer_receipt="$incoming/LAYER_K2_RECEIPT.json"
verification_json=$(cat "$incoming/VERIFICATION.json")
jq -n --argjson layer "$layer" --argjson experts "$experts_json" \
    --argjson verification "$verification_json" \
    '{schema:"glm53-full-exl3-tp3.k275-layer-receipt.v1",passed:true,geometry_id:"rotating-uneven-768-640-640-v1",layer:$layer,bits:2,k2_experts:$experts,verified_experts:$verification.verified_experts,bytes:$verification.bytes,worst_relative_rmse:$verification.worst_relative_rmse}' \
    > "$layer_receipt"
write_progress checksumming_unas_incoming
(cd "$incoming" && find . -type f ! -name ARTIFACT_SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > ARTIFACT_SHA256SUMS)
(cd "$incoming" && sha256sum -c ARTIFACT_SHA256SUMS >/dev/null)
write_progress publishing_atomic_rename
mv "$incoming" "$target"

write_progress writing_completion_receipt
receipt_sha=$(sha256sum "$target/LAYER_K2_RECEIPT.json" | awk '{print $1}')
ledger_sha=$(sha256sum "$target/ARTIFACT_SHA256SUMS" | awk '{print $1}')
published_bytes=$(du -sb "$target" | awk '{print $1}')
ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
tmp="${result}.tmp.$$"
jq -n --arg run_stamp "$run_stamp" --argjson layer "$layer" --arg node "$node" \
    --arg lease_id "$lease_id" --arg started "$started" --arg ended "$ended" \
    --arg target "$target" --arg receipt_sha "$receipt_sha" --arg ledger_sha "$ledger_sha" \
    --argjson published_bytes "$published_bytes" --argjson experts "$experts_json" \
    '{schema:"glm53-full-exl3-tp3.k275-worker-result.v1",passed:true,exit_code:0,
      reason:"sealed_and_published",run_stamp:$run_stamp,layer:$layer,node:$node,
      lease_id:$lease_id,started_at:$started,ended_at:$ended,publish_path:$target,
      bits:2,k2_experts:$experts,layer_receipt_sha256:$receipt_sha,
      artifact_ledger_sha256:$ledger_sha,published_bytes:$published_bytes}' > "$tmp"
chmod 644 "$tmp"
mv "$tmp" "$result"
write_progress published
trap - EXIT HUP INT TERM
ssh -F zima-ssh-config "$node" "rm -rf -- '$remote_source'" || true
