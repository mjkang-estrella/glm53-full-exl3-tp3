#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
source_dir=${2:?sealed checkpoint path required}
base=$(basename "$source_dir")
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
cd "$project"
central="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/replication"
seal="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/checkpoint-seal.json"
logs="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp/replication"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
mkdir -p "$central" "$logs"

test -s "$source_dir/ASSEMBLY_COMPLETE.json"
test -s "$seal"
jq -e '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1"' "$source_dir/ASSEMBLY_COMPLETE.json" >/dev/null
expected_complete=$(jq -r '.assembly_complete_sha256' "$seal")
source_bytes=$(jq -r '.file_bytes' "$source_dir/MANIFEST.json")
reserve_bytes=$((12 * 1024 * 1024 * 1024))
disk_buffer_bytes=$((20 * 1024 * 1024 * 1024))

# Fail before the first byte is copied if a node cannot retain a bounded disk
# buffer, the 12 GiB host reserve, or the protected service-stop condition.
for rank in 0 1 2; do
    node=${nodes[$rank]}
    service=${services[$rank]}
    disk_available=$(ssh -F zima-ssh-config "$node" "df -B1 --output=avail /home/mj-kang/Dev | tail -n 1")
    memory_available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    running=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '$service'")
    competing=$(ssh -F zima-ssh-config "$node" "docker ps --format '{{.Names}}' | grep -Ec '^(glm53-k3-candidate|glm53-tp3-k3-uneven-L)' || true")
    target_exists=$(ssh -F zima-ssh-config "$node" "if test -e '/home/mj-kang/Dev/models/$base'; then echo true; else echo false; fi")
    passed=true
    (( disk_available >= source_bytes + disk_buffer_bytes )) || passed=false
    (( memory_available >= reserve_bytes )) || passed=false
    [[ "$running" == false && "$competing" == 0 && "$target_exists" == false ]] || passed=false
    jq -n --arg node "$node" --arg service "$service" --argjson rank "$rank" \
        --argjson disk_available "$disk_available" --argjson memory_available "$memory_available" \
        --argjson source_bytes "$source_bytes" --argjson disk_buffer_bytes "$disk_buffer_bytes" \
        --arg running "$running" --argjson competing "$competing" --arg target_exists "$target_exists" --arg passed "$passed" \
        '{schema:"glm53-full-exl3-tp3.replica-preflight.v1",rank:$rank,node:$node,protected_service:$service,
          protected_service_running:($running=="true"),competing_experimental_containers:$competing,final_target_exists:($target_exists=="true"),
          source_bytes:$source_bytes,disk_available_bytes:$disk_available,disk_buffer_required_bytes:$disk_buffer_bytes,
          projected_disk_available_bytes:($disk_available-$source_bytes),memory_available_bytes:$memory_available,
          host_reserve_required_bytes:12884901888,passed:($passed=="true")}' \
        > "$central/rank-$rank-preflight.json.tmp"
    mv "$central/rank-$rank-preflight.json.tmp" "$central/rank-$rank-preflight.json"
    [[ "$passed" == true ]] || { printf '%s\n' "replica preflight failed on $node" >&2; exit 2; }
done

copy_one() {
    local rank=$1 node=${nodes[$1]}
    local final="/home/mj-kang/Dev/models/$base"
    local incoming="${final}.incoming-$stamp"
    local remote_state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/replication/rank-$rank"
    ssh -F zima-ssh-config "$node" "mkdir -p '$incoming' '$remote_state'"
    rsync -a --partial --append-verify --info=progress2 "$source_dir/" \
        -e "ssh -F zima-ssh-config" "$node:$incoming/" > "$logs/rank-$rank-rsync.log" 2>&1
    ssh -F zima-ssh-config "$node" \
        "cd '$project' && python3 scripts/verify_replica.py --checkpoint '$incoming' --output '$remote_state/VERIFIED.json' --workers 4" \
        > "$logs/rank-$rank-verify.log" 2>&1
    scp -q -F zima-ssh-config "$node:$remote_state/VERIFIED.json" "$central/rank-$rank.json"
    jq -e --arg expected "$expected_complete" '.passed == true and .assembly_complete_sha256 == $expected' "$central/rank-$rank.json" >/dev/null
    if ssh -F zima-ssh-config "$node" "test -e '$final'"; then
        printf '%s\n' "refusing to replace existing final replica $node:$final" >&2
        return 2
    fi
    ssh -F zima-ssh-config "$node" "mv '$incoming' '$final'"
}

pids=()
for rank in 0 1 2; do
    copy_one "$rank" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
    jq -n --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        '{schema:"glm53-full-exl3-tp3.replication-summary.v1",passed:false,failed_at:$at}' \
        > "$central/SUMMARY.json.tmp"
    mv "$central/SUMMARY.json.tmp" "$central/SUMMARY.json"
    exit 2
fi
jq -s --arg source "$source_dir" --arg base "$base" --arg expected "$expected_complete" \
    '{schema:"glm53-full-exl3-tp3.replication-summary.v1",
      passed:(length==3 and all(.[];.passed and .assembly_complete_sha256==$expected) and ([.[].assembly_complete_sha256]|unique|length)==1 and ([.[].manifest_sha256]|unique|length)==1 and ([.[].sha256sums_sha256]|unique|length)==1),
      source:$source,replica_basename:$base,ranks:.}' \
    "$central/rank-0.json" "$central/rank-1.json" "$central/rank-2.json" > "$central/SUMMARY.json.tmp"
mv "$central/SUMMARY.json.tmp" "$central/SUMMARY.json"
jq -e '.passed == true' "$central/SUMMARY.json" >/dev/null
