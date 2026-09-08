#!/usr/bin/env bash
set -euo pipefail

# Build local mixed replicas by hardlinking the already-present 3.0bpw base and
# transferring only the changed K2 expert files plus mixed-checkpoint metadata.
# Run this controller on Zima; it is safe to leave in a Zima tmux session.

source_dir=${1:?assembled 2.75bpw source directory required}
run_stamp=${2:?replication run stamp required}
base_basename=${3:?existing 3.0bpw replica basename required}
target_basename=${4:?new mixed replica basename required}

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
ssh_config=$project/zima-ssh-config
state_root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/k275/replication
state=$state_root/$run_stamp
logs=/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/k275-replication-$run_stamp
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
identity=/home/mj-kang/.ssh/id_ed25519_nvsync_cluster_assistant

[[ $run_stamp =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid run stamp" >&2; exit 2; }
[[ $source_dir == /mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-* ]] || {
    echo "source must be the assembled 2.75bpw UNAS checkpoint" >&2
    exit 2
}
[[ $base_basename == GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-* ]] || {
    echo "base must be the sealed 3.0bpw replica" >&2
    exit 2
}
[[ $target_basename == GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-* ]] || {
    echo "target must be the assembled 2.75bpw replica" >&2
    exit 2
}

mkdir -p "$state" "$logs"

write_status() {
    local phase="$1" details="${2-}"
    [[ -n "$details" ]] || details='{}'
    jq -n --arg phase "$phase" --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        --arg run_stamp "$run_stamp" --arg source "$source_dir" \
        --arg base "$base_basename" --arg target "$target_basename" \
        --argjson details "$details" \
        '{schema:"glm53-full-exl3-tp3.k275-incremental-replication-status.v1",
          phase:$phase,updated_at:$at,run_stamp:$run_stamp,source:$source,
          base_replica:$base,target_replica:$target,details:$details}' \
        > "$state/STATUS.json.tmp"
    mv "$state/STATUS.json.tmp" "$state/STATUS.json"
}

fail() {
    local code=$? line=$1
    trap - ERR
    jq -n --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" --argjson code "$code" \
        --argjson line "$line" --arg run_stamp "$run_stamp" \
        '{schema:"glm53-full-exl3-tp3.k275-incremental-replication-failure.v1",
          failed_at:$at,exit_code:$code,line:$line,run_stamp:$run_stamp,
          policy:"incoming replicas and the sealed base remain preserved"}' \
        > "$state/FAILED.json.tmp"
    mv "$state/FAILED.json.tmp" "$state/FAILED.json"
    write_status failed "$(jq -n --argjson code "$code" --argjson line "$line" '{exit_code:$code,line:$line}')"
    exit "$code"
}
trap 'fail "$LINENO"' ERR

test -s "$source_dir/ASSEMBLY_COMPLETE.json"
jq -e '.passed == true and .target_bpw == "2.75" and .layers == 76' \
    "$source_dir/ASSEMBLY_COMPLETE.json" >/dev/null
test -s "$source_dir/expert-bits.json"

write_status preparing_file_list
python3 - "$source_dir" "$state/files-from.txt" "$state/files.sha256" "$state/FILE_LIST.json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
files_path = pathlib.Path(sys.argv[2])
hashes_path = pathlib.Path(sys.argv[3])
receipt_path = pathlib.Path(sys.argv[4])
bits = json.loads((source / "expert-bits.json").read_text(encoding="utf-8"))
manifest = json.loads((source / "MANIFEST.json").read_text(encoding="utf-8"))
paths = set()
for layer, row in bits["bits"].items():
    for expert in row["k2"]:
        paths.add(f"k3-layer-{int(layer):03d}-expert-{int(expert):03d}.safetensors")
paths.update({
    "expert-bits.json",
    "quantization_config.json",
    "tier-map.json",
    "model.safetensors.index.json",
    "MANIFEST.json",
    "SHA256SUMS",
    "ASSEMBLY_COMPLETE.json",
})
missing = sorted(path for path in paths if not (source / path).is_file())
if missing:
    raise SystemExit(f"missing source files: {missing[:8]}")
rows = []
for path in sorted(paths):
    record = manifest.get("files", {}).get(path)
    if not record:
        target = source / path
        digest = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
        record = {"bytes": target.stat().st_size, "sha256": digest}
    if not record.get("sha256"):
        raise SystemExit(f"source manifest lacks {path}")
    rows.append({"path": path, "bytes": int(record["bytes"]), "sha256": record["sha256"]})
files_path.write_text("".join(f"{row['path']}\n" for row in rows), encoding="utf-8")
hashes_path.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8")
receipt_path.write_text(json.dumps({
    "schema": "glm53-full-exl3-tp3.k275-replication-file-list.v1",
    "files": rows,
    "file_count": len(rows),
    "changed_payload_bytes": sum(row["bytes"] for row in rows if row["path"].endswith(".safetensors")),
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

k2_bytes=$(jq -r '.changed_payload_bytes' "$state/FILE_LIST.json")
files_count=$(jq -r '.file_count' "$state/FILE_LIST.json")
disk_required=$((k2_bytes + 20 * 1024 * 1024 * 1024))
write_status preflight "$(jq -n --argjson files "$files_count" --argjson k2_bytes "$k2_bytes" \
    --argjson disk_required "$disk_required" \
    '{changed_file_count:$files,changed_payload_bytes:$k2_bytes,disk_required_bytes:$disk_required}')"

for rank in 0 1 2; do
    node=${nodes[$rank]}
    service=${services[$rank]}
    available=$(ssh -F "$ssh_config" "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    disk=$(ssh -F "$ssh_config" "$node" "df -B1 --output=avail /home/mj-kang/Dev | tail -n 1")
    running=$(ssh -F "$ssh_config" "$node" "docker inspect -f '{{.State.Running}}' '$service'")
    competing=$(ssh -F "$ssh_config" "$node" "docker ps --format '{{.Names}}' | grep -Ec '^(glm53-k3-|glm53-tp3-k3-uneven)' || true")
    ssh -F "$ssh_config" "$node" "test -s '/home/mj-kang/Dev/models/$base_basename/ASSEMBLY_COMPLETE.json'"
    [[ "$running" == false && "$competing" == 0 ]]
    (( available >= 12 * 1024 * 1024 * 1024 ))
    (( disk >= disk_required ))
    ssh -F "$ssh_config" "$node" "test ! -e '/home/mj-kang/Dev/models/$target_basename' && test ! -e '/home/mj-kang/Dev/models/$target_basename.incoming-$run_stamp'"
    jq -n --arg node "$node" --argjson rank "$rank" --argjson available "$available" \
        --argjson disk "$disk" --argjson k2_bytes "$k2_bytes" \
        '{node:$node,rank:$rank,mem_available_bytes:$available,disk_available_bytes:$disk,
          changed_payload_bytes:$k2_bytes,passed:true}' > "$state/rank-$rank-preflight.json"
done

clone_base() {
    local rank=$1 node=${nodes[$1]}
    local final="/home/mj-kang/Dev/models/$target_basename"
    local incoming="${final}.incoming-$run_stamp"
    ssh -F "$ssh_config" "$node" "cp -al '/home/mj-kang/Dev/models/$base_basename' '$incoming'"
}
write_status cloning_base
for rank in 0 1 2; do clone_base "$rank" & done
wait

seed_incoming="/home/mj-kang/Dev/models/$target_basename.incoming-$run_stamp"
ssh -F "$ssh_config" mj-spark-1 "mkdir -p '$state'"
scp -q -F "$ssh_config" "$state/files-from.txt" "mj-spark-1:$state/files-from.txt"
scp -q -F "$ssh_config" "$state/files.sha256" "mj-spark-1:$state/files.sha256"
write_status seed_copying "$(jq -n '{node:"mj-spark-1",source:"zima"}')"
rsync -a --partial --protect-args --files-from="$state/files-from.txt" \
    -e "ssh -F $ssh_config" "$source_dir/" "mj-spark-1:$seed_incoming/" \
    > "$logs/rank-0-rsync.log" 2>&1
ssh -F "$ssh_config" mj-spark-1 \
    "cd '$seed_incoming' && sha256sum -c '$state/files.sha256'" \
    > "$logs/rank-0-verify.log" 2>&1

seed_ip=(unused 10.100.200.2 10.100.204.1)
dest_ip=(unused 10.100.200.1 10.100.204.2)
write_status fabric_copying "$(jq -n '{source_node:"mj-spark-1",destinations:["mj-spark-2","mj-spark-3"]}')"
for rank in 1 2; do
    node=${nodes[$rank]}
    dest_incoming="/home/mj-kang/Dev/models/$target_basename.incoming-$run_stamp"
    ssh -F "$ssh_config" mj-spark-1 \
        "rsync -a --partial --protect-args --files-from='$state/files-from.txt' \
          -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -i $identity -b ${seed_ip[$rank]}' \
          '$seed_incoming/' 'mj-kang@${dest_ip[$rank]}:$dest_incoming/'" \
        > "$logs/rank-$rank-rsync.log" 2>&1 &
done
wait

for rank in 0 1 2; do
    node=${nodes[$rank]}
    ssh -F "$ssh_config" "$node" "mkdir -p '$state'"
    scp -q -F "$ssh_config" "$state/files.sha256" "$node:$state/files.sha256"
    incoming="/home/mj-kang/Dev/models/$target_basename.incoming-$run_stamp"
    ssh -F "$ssh_config" "$node" "cd '$incoming' && sha256sum -c '$state/files.sha256'" \
        > "$logs/rank-$rank-verify.log" 2>&1
    ssh -F "$ssh_config" "$node" "test ! -e '/home/mj-kang/Dev/models/$target_basename' && mv '$incoming' '/home/mj-kang/Dev/models/$target_basename'"
    jq -n --arg node "$node" --argjson rank "$rank" --arg target "$target_basename" \
        '{node:$node,rank:$rank,target:$target,verified_changed_files:true,published:true}' \
        > "$state/rank-$rank.json"
done

jq -s --arg source "$source_dir" --arg base "$base_basename" --arg target "$target_basename" \
    --arg run_stamp "$run_stamp" \
    '{schema:"glm53-full-exl3-tp3.k275-incremental-replication-summary.v1",passed:(length==3 and all(.[];.verified_changed_files and .published)),run_stamp:$run_stamp,source:$source,base_replica:$base,target_replica:$target,ranks:.}' \
    "$state"/rank-{0,1,2}.json > "$state/SUMMARY.json"
jq -e '.passed == true' "$state/SUMMARY.json" >/dev/null
jq -n --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" --arg summary "$state/SUMMARY.json" \
    '{schema:"glm53-full-exl3-tp3.k275-incremental-replication-complete.v1",passed:true,completed_at:$at,summary:$summary}' \
    > "$state/COMPLETE.json"
write_status complete "$(jq -n --arg summary "$state/SUMMARY.json" '{summary:$summary}')"
printf '%s\n' "$state/SUMMARY.json"
