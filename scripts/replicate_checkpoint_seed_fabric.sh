#!/usr/bin/env bash
set -Eeuo pipefail

stamp=${1:?real-test stamp required}
source_dir=${2:?sealed checkpoint path required}
handoff_stamp=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
base=$(basename "$source_dir")
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
central="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/replication"
seal="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/checkpoint-seal.json"
logs_root="/home/mj-kang/Dev/logs/glm53-full-exl3-tp3/real-test-$stamp"
fabric_state="$central/fabric-$handoff_stamp"
fabric_logs="$logs_root/replication-fabric-$handoff_stamp"
seed_node=mj-spark-1
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
services=(glm53-exl3-head glm53-exl3-worker minimax-h3-comfy)
dest_ips=(unused 10.100.200.1 10.100.204.2)
source_ips=(unused 10.100.200.2 10.100.204.1)
source_ifaces=(unused enp1s0f0np0 enp1s0f1np1)
expected_hosts=(unused mj-spark-2 mj-spark-3)
identity=/home/mj-kang/.ssh/id_ed25519_nvsync_cluster_assistant
reserve_bytes=$((12 * 1024 * 1024 * 1024))
disk_buffer_bytes=$((20 * 1024 * 1024 * 1024))
mkdir -p "$central" "$fabric_state" "$fabric_logs"
cd "$project"

atomic_status() {
    local phase=$1 details=${2:-}
    [[ -n "$details" ]] || details='{}'
    jq -n --arg phase "$phase" --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        --arg handoff_stamp "$handoff_stamp" --argjson details "$details" \
        '{schema:"glm53-full-exl3-tp3.fabric-replication-status.v1",handoff_stamp:$handoff_stamp,
          phase:$phase,updated_at:$at,details:$details}' > "$fabric_state/STATUS.json.tmp"
    mv "$fabric_state/STATUS.json.tmp" "$fabric_state/STATUS.json"
}

on_error() {
    local code=$1 line=$2
    trap - ERR
    jq -n --arg at "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" --argjson exit_code "$code" \
        --argjson line "$line" --arg handoff_stamp "$handoff_stamp" \
        '{schema:"glm53-full-exl3-tp3.fabric-replication-failure.v1",handoff_stamp:$handoff_stamp,
          failed_at:$at,exit_code:$exit_code,line:$line,
          policy:"partials and completed replicas preserved; Flash/H3 remain stopped"}' \
        > "$fabric_state/FAILED.json.tmp"
    mv "$fabric_state/FAILED.json.tmp" "$fabric_state/FAILED.json"
    atomic_status failed "$(jq -n --argjson exit_code "$code" --argjson line "$line" '{exit_code:$exit_code,line:$line}')"
    exit "$code"
}
trap 'on_error "$?" "$LINENO"' ERR

fail() {
    printf '%s\n' "$1" >&2
    on_error 2 "$LINENO"
}

test -s "$source_dir/ASSEMBLY_COMPLETE.json"
test -s "$seal"
jq -e '.passed == true and .geometry_id == "rotating-uneven-768-640-640-v1"' \
    "$source_dir/ASSEMBLY_COMPLETE.json" >/dev/null
expected_complete=$(jq -r '.assembly_complete_sha256' "$seal")
source_bytes=$(jq -r '.file_bytes' "$source_dir/MANIFEST.json")
seed_final="/home/mj-kang/Dev/models/$base"

# The first controller owns the only Zima-to-Spark stream. Wait for its rank-0
# hash verification and atomic rename; never race its rsync or verifier.
atomic_status waiting_for_seed "$(jq -n --arg node "$seed_node" '{seed_node:$node}')"
while :; do
    seed_ready=false
    if test -s "$central/rank-0.json" && \
       jq -e --arg expected "$expected_complete" \
          '.passed == true and .assembly_complete_sha256 == $expected' "$central/rank-0.json" >/dev/null && \
       ssh -F zima-ssh-config "$seed_node" "test -d '$seed_final'"; then
        seed_ready=true
    fi
    old_controller=false
    if pgrep -f "^bash scripts/replicate_checkpoint\.sh $stamp " >/dev/null; then
        old_controller=true
    fi
    if [[ "$seed_ready" == true && "$old_controller" == false ]]; then
        break
    fi
    incoming_bytes=$(ssh -F zima-ssh-config "$seed_node" \
        "du -sb '${seed_final}.incoming-$stamp' 2>/dev/null | awk '{print \$1}' || true")
    incoming_bytes=${incoming_bytes:-0}
    atomic_status waiting_for_seed \
        "$(jq -n --arg node "$seed_node" --argjson bytes "$incoming_bytes" \
            --argjson source_bytes "$source_bytes" --argjson old_controller "$old_controller" \
            '{seed_node:$node,apparent_bytes:$bytes,source_bytes:$source_bytes,old_controller_running:$old_controller}')"
    sleep 30
done

# The superseded controller intentionally reports failure for the two stopped
# LAN streams. Preserve that report before publishing the fabric result.
if test -s "$central/SUMMARY.json"; then
    cp --no-clobber "$central/SUMMARY.json" "$fabric_state/prior-zima-parallel-SUMMARY.json"
fi
jq -e --arg expected "$expected_complete" \
    '.passed == true and .assembly_complete_sha256 == $expected' "$central/rank-0.json" >/dev/null

# Refuse any competing checkpoint transfer before qualifying the direct links.
if ps -eo args | grep -F "rsync" | grep -F "$base" | grep -v grep >/dev/null; then
    fail "competing Zima checkpoint rsync remains after seed completion"
fi

for rank in 1 2; do
    node=${nodes[$rank]}
    service=${services[$rank]}
    dest_ip=${dest_ips[$rank]}
    source_ip=${source_ips[$rank]}
    source_iface=${source_ifaces[$rank]}
    expected_host=${expected_hosts[$rank]}
    final="/home/mj-kang/Dev/models/$base"
    incoming="${final}.incoming-$stamp"

    route=$(ssh -F zima-ssh-config "$seed_node" "ip route get '$dest_ip' | head -n 1")
    [[ "$route" == *" dev $source_iface "* && "$route" == *" src $source_ip "* ]] || {
        fail "direct route mismatch for rank $rank: $route"
    }
    configured_host=$(ssh -F zima-ssh-config "$seed_node" \
        "ssh -G '$node' 2>/dev/null | awk 'tolower(\$1)==\"hostname\" {print \$2; exit}'")
    [[ "$configured_host" == "$dest_ip" ]] || {
        fail "Spark 1 SSH alias $node resolves to $configured_host, not $dest_ip"
    }
    host_key=$(ssh -F zima-ssh-config "$seed_node" \
        "ssh-keygen -F '$dest_ip' -f ~/.ssh/known_hosts | ssh-keygen -lf - | head -n 1")
    test -n "$host_key"
    reached=$(ssh -F zima-ssh-config "$seed_node" \
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=5 -i '$identity' -b '$source_ip' 'mj-kang@$dest_ip' hostname")
    [[ "$reached" == "$expected_host" ]] || {
        fail "direct SSH reached $reached instead of $expected_host"
    }

    memory_available=$(ssh -F zima-ssh-config "$node" "awk '/MemAvailable:/ {print \$2*1024}' /proc/meminfo")
    disk_available=$(ssh -F zima-ssh-config "$node" "df -B1 --output=avail /home/mj-kang/Dev | tail -n 1")
    partial_allocated=$(ssh -F zima-ssh-config "$node" "du -sB1 '$incoming' 2>/dev/null | awk '{print \$1}' || true")
    partial_allocated=${partial_allocated:-0}
    remaining_bytes=$((source_bytes > partial_allocated ? source_bytes - partial_allocated : 0))
    projected_disk_available=$((disk_available - remaining_bytes))
    running=$(ssh -F zima-ssh-config "$node" "docker inspect -f '{{.State.Running}}' '$service'")
    competing=$(ssh -F zima-ssh-config "$node" \
        "docker ps --format '{{.Names}}' | grep -Ec '^(glm53-k3-candidate|glm53-tp3-k3-uneven-L)' || true")
    final_exists=$(ssh -F zima-ssh-config "$node" "test -e '$final' && echo true || echo false")
    passed=true
    (( memory_available >= reserve_bytes )) || passed=false
    (( projected_disk_available >= disk_buffer_bytes )) || passed=false
    [[ "$running" == false && "$competing" == 0 ]] || passed=false
    if [[ "$final_exists" == true ]]; then
        test -s "$central/rank-$rank.json" && \
            jq -e --arg expected "$expected_complete" \
                '.passed == true and .assembly_complete_sha256 == $expected' "$central/rank-$rank.json" >/dev/null || passed=false
    fi
    jq -n --arg node "$node" --arg service "$service" --argjson rank "$rank" \
        --arg dest_ip "$dest_ip" --arg source_ip "$source_ip" --arg source_iface "$source_iface" \
        --arg route "$route" --arg host_key "$host_key" --arg reached "$reached" \
        --argjson source_bytes "$source_bytes" \
        --argjson disk_available "$disk_available" --argjson partial_allocated "$partial_allocated" \
        --argjson remaining_bytes "$remaining_bytes" --argjson projected_disk_available "$projected_disk_available" \
        --argjson memory_available "$memory_available" --arg running "$running" \
        --argjson competing "$competing" --arg final_exists "$final_exists" --argjson passed "$passed" \
        '{schema:"glm53-full-exl3-tp3.fabric-preflight.v1",rank:$rank,node:$node,
          direct:{destination_ip:$dest_ip,source_ip:$source_ip,source_interface:$source_iface,
                  route:$route,known_host_fingerprint:$host_key,reached_hostname:$reached},
          protected_service:$service,protected_service_running:($running=="true"),
          competing_experimental_containers:$competing,final_target_exists:($final_exists=="true"),
          source_bytes:$source_bytes,disk_available_bytes:$disk_available,
          partial_allocated_bytes:$partial_allocated,remaining_bytes:$remaining_bytes,
          projected_disk_available_bytes:$projected_disk_available,disk_buffer_required_bytes:21474836480,
          memory_available_bytes:$memory_available,host_reserve_required_bytes:12884901888,
          passed:$passed}' > "$fabric_state/rank-$rank-preflight.json.tmp"
    mv "$fabric_state/rank-$rank-preflight.json.tmp" "$fabric_state/rank-$rank-preflight.json"
    [[ "$passed" == true ]] || fail "fabric preflight failed on $node"
done

copy_one() {
    local rank=$1 node=${nodes[$1]} dest_ip=${dest_ips[$1]} source_ip=${source_ips[$1]}
    local final="/home/mj-kang/Dev/models/$base"
    local incoming="${final}.incoming-$stamp"
    local remote_state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/replication/rank-$rank"
    local started completed
    if ssh -F zima-ssh-config "$node" "test -d '$final'"; then
        jq -e --arg expected "$expected_complete" \
            '.passed == true and .assembly_complete_sha256 == $expected' "$central/rank-$rank.json" >/dev/null
        return 0
    fi
    ssh -F zima-ssh-config "$node" "mkdir -p '$incoming' '$remote_state'"
    started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    ssh -F zima-ssh-config "$seed_node" bash -s -- \
        "$seed_final" "$incoming" "$dest_ip" "$source_ip" "$identity" <<'REMOTE' \
        > "$fabric_logs/rank-$rank-rsync.log" 2>&1
set -euo pipefail
source_dir=$1
incoming=$2
dest_ip=$3
source_ip=$4
identity=$5
rsync -a --partial --append-verify --protect-args --info=progress2 --stats "$source_dir/" \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -i $identity -b $source_ip" \
    "mj-kang@$dest_ip:$incoming/"
REMOTE
    completed=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    ssh -F zima-ssh-config "$node" \
        "cd '$project' && python3 scripts/verify_replica.py --checkpoint '$incoming' --output '$remote_state/VERIFIED.json' --workers 4" \
        > "$fabric_logs/rank-$rank-verify.log" 2>&1
    scp -q -F zima-ssh-config "$node:$remote_state/VERIFIED.json" "$central/rank-$rank.json.tmp"
    mv "$central/rank-$rank.json.tmp" "$central/rank-$rank.json"
    jq -e --arg expected "$expected_complete" \
        '.passed == true and .assembly_complete_sha256 == $expected' "$central/rank-$rank.json" >/dev/null
    ssh -F zima-ssh-config "$node" "test ! -e '$final' && mv '$incoming' '$final'"
    jq -n --argjson rank "$rank" --arg node "$node" --arg source_node "$seed_node" \
        --arg source_ip "$source_ip" --arg destination_ip "$dest_ip" \
        --arg started_at "$started" --arg completed_at "$completed" \
        --arg rsync_log_sha256 "$(sha256sum "$fabric_logs/rank-$rank-rsync.log" | awk '{print $1}')" \
        '{schema:"glm53-full-exl3-tp3.fabric-transfer.v1",rank:$rank,node:$node,
          source_node:$source_node,source_ip:$source_ip,destination_ip:$destination_ip,
          started_at:$started_at,rsync_completed_at:$completed_at,rsync_log_sha256:$rsync_log_sha256,
          verified_and_published:true}' > "$fabric_state/rank-$rank-transfer.json.tmp"
    mv "$fabric_state/rank-$rank-transfer.json.tmp" "$fabric_state/rank-$rank-transfer.json"
}

atomic_status fabric_copying \
    "$(jq -n '{source_node:"mj-spark-1",destinations:["mj-spark-2","mj-spark-3"],parallel:true}')"
pids=()
for rank in 1 2; do
    copy_one "$rank" &
    pids+=("$!")
done

while kill -0 "${pids[0]}" 2>/dev/null || kill -0 "${pids[1]}" 2>/dev/null; do
    rank1_bytes=$(ssh -F zima-ssh-config mj-spark-2 \
        "du -sb '/home/mj-kang/Dev/models/$base.incoming-$stamp' 2>/dev/null | awk '{print \$1}' || true")
    rank2_bytes=$(ssh -F zima-ssh-config mj-spark-3 \
        "du -sb '/home/mj-kang/Dev/models/$base.incoming-$stamp' 2>/dev/null | awk '{print \$1}' || true")
    rank1_bytes=${rank1_bytes:-$source_bytes}
    rank2_bytes=${rank2_bytes:-$source_bytes}
    atomic_status fabric_copying \
        "$(jq -n --argjson source_bytes "$source_bytes" --argjson rank1_bytes "$rank1_bytes" \
            --argjson rank2_bytes "$rank2_bytes" \
            '{source_node:"mj-spark-1",source_bytes:$source_bytes,
              destinations:[{rank:1,node:"mj-spark-2",apparent_bytes:$rank1_bytes},
                            {rank:2,node:"mj-spark-3",apparent_bytes:$rank2_bytes}],parallel:true}')"
    sleep 30
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || fail "one or more fabric copy/verify jobs failed"

jq -s --arg source "$source_dir" --arg base "$base" --arg expected "$expected_complete" \
    --arg strategy "zima-to-spark1-seed-then-explicit-direct-spark1-fanout" \
    '{schema:"glm53-full-exl3-tp3.replication-summary.v1",
      passed:(length==3 and all(.[];.passed and .assembly_complete_sha256==$expected) and
              ([.[].assembly_complete_sha256]|unique|length)==1 and
              ([.[].manifest_sha256]|unique|length)==1 and
              ([.[].sha256sums_sha256]|unique|length)==1),
      strategy:$strategy,source:$source,replica_basename:$base,ranks:.}' \
    "$central/rank-0.json" "$central/rank-1.json" "$central/rank-2.json" \
    > "$central/SUMMARY.json.tmp"
mv "$central/SUMMARY.json.tmp" "$central/SUMMARY.json"
jq -e '.passed == true' "$central/SUMMARY.json" >/dev/null
atomic_status complete \
    "$(jq -n --arg summary "$central/SUMMARY.json" '{summary:$summary,replicas_verified:3}')"
trap - ERR
