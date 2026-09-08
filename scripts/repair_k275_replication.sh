#!/usr/bin/env bash
set -euo pipefail

# Repair only the two incoming replicas left by the first incremental copy.
# The original attempt used --append-verify, which cannot shrink a larger
# hardlinked K3 destination into a smaller K2 file.  This script uses ordinary
# rsync temp-file replacement, then verifies the complete changed-file ledger
# before publishing each final replica.

run_stamp=${1:?replication run stamp required}
target_basename=${2:?mixed replica basename required}
base_basename=${3:?sealed base basename required}

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
ssh_config=$project/zima-ssh-config
state=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/k275/replication/$run_stamp
logs=/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/k275-replication-$run_stamp
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
seed_ip=(unused 10.100.200.2 10.100.204.1)
dest_ip=(unused 10.100.200.1 10.100.204.2)
identity=/home/mj-kang/.ssh/id_ed25519_nvsync_cluster_assistant

[[ $run_stamp =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid run stamp" >&2; exit 2; }
[[ $target_basename == GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-* ]] || { echo "invalid target" >&2; exit 2; }
[[ $base_basename == GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-* ]] || { echo "invalid base" >&2; exit 2; }
mkdir -p "$logs"

write_status() {
    local phase="$1" details="${2-}"
    [[ -n "$details" ]] || details='{}'
    jq -n --arg phase "$phase" --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        --arg run_stamp "$run_stamp" --arg target "$target_basename" \
        --argjson details "$details" \
        '{schema:"glm53-full-exl3-tp3.k275-incremental-repair-status.v1",phase:$phase,updated_at:$at,run_stamp:$run_stamp,target_replica:$target,details:$details}' \
        > "$state/REPAIR_STATUS.json.tmp"
    mv "$state/REPAIR_STATUS.json.tmp" "$state/REPAIR_STATUS.json"
}

test -s "$state/files-from.txt"
test -s "$state/files.sha256"
test -s "$state/rank-0.json"
ssh -F "$ssh_config" mj-spark-1 "test -s '/home/mj-kang/Dev/models/$target_basename/ASSEMBLY_COMPLETE.json'"

write_status preflight
for rank in 1 2; do
    node=${nodes[$rank]}
    incoming="/home/mj-kang/Dev/models/$target_basename.incoming-$run_stamp"
    final="/home/mj-kang/Dev/models/$target_basename"
    ssh -F "$ssh_config" "$node" "test -d '$incoming' && test ! -e '$final'"
    ssh -F "$ssh_config" "$node" "docker ps --format '{{.Names}}' | grep -Eq '^(glm53-k3-|glm53-tp3-k3-uneven)'" && {
        echo "experimental container active on $node" >&2
        exit 2
    } || true
    ssh -F "$ssh_config" "$node" "mkdir -p '$state'"
    scp -q -F "$ssh_config" "$state/files.sha256" "$node:$state/files.sha256"
done

seed_final="/home/mj-kang/Dev/models/$target_basename"
write_status fabric_replacing
for rank in 1 2; do
    node=${nodes[$rank]}
    incoming="/home/mj-kang/Dev/models/$target_basename.incoming-$run_stamp"
    ssh -F "$ssh_config" mj-spark-1 \
        "rsync -a --partial --protect-args --files-from='$state/files-from.txt' \
          -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -i $identity -b ${seed_ip[$rank]}' \
          '$seed_final/' 'mj-kang@${dest_ip[$rank]}:$incoming/'" \
        > "$logs/repair-rank-$rank-rsync.log" 2>&1
done

for rank in 1 2; do
    node=${nodes[$rank]}
    incoming="/home/mj-kang/Dev/models/$target_basename.incoming-$run_stamp"
    ssh -F "$ssh_config" "$node" "cd '$incoming' && sha256sum -c '$state/files.sha256'" \
        > "$logs/repair-rank-$rank-verify.log" 2>&1
    ssh -F "$ssh_config" "$node" "test ! -e '/home/mj-kang/Dev/models/$target_basename' && mv '$incoming' '/home/mj-kang/Dev/models/$target_basename'"
    jq -n --arg node "$node" --argjson rank "$rank" --arg target "$target_basename" \
        '{node:$node,rank:$rank,target:$target,verified_changed_files:true,published:true}' \
        > "$state/rank-$rank.json"
done

jq -s --arg target "$target_basename" --arg run_stamp "$run_stamp" \
    '{schema:"glm53-full-exl3-tp3.k275-incremental-replication-summary.v1",passed:(length==3 and all(.[];.verified_changed_files and .published)),run_stamp:$run_stamp,target_replica:$target,ranks:.}' \
    "$state"/rank-{0,1,2}.json > "$state/SUMMARY.json"
jq -e '.passed == true' "$state/SUMMARY.json" >/dev/null
jq -n --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" --arg summary "$state/SUMMARY.json" \
    '{schema:"glm53-full-exl3-tp3.k275-incremental-replication-complete.v1",passed:true,completed_at:$at,summary:$summary}' \
    > "$state/COMPLETE.json"
write_status complete "$(jq -n --arg summary "$state/SUMMARY.json" '{summary:$summary}')"
printf '%s\n' "$state/SUMMARY.json"
