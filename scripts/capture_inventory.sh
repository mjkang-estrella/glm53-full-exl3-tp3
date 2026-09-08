#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
stamp=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
local_base="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/rollback/${stamp}/mj-zima"
mkdir -p "$local_base/spark-summaries"
chmod 700 "$local_base"
umask 077

date -u +%FT%TZ > "$local_base/captured-at.txt"
python3 scripts/kernel_audit.py --write-policy "$local_base/lifecycle-policy.json"
uname -a > "$local_base/uname.txt"
free -b > "$local_base/free-before.txt"
swapon --show --bytes > "$local_base/swap-before.txt" 2>&1 || true
df -B1 -T > "$local_base/filesystems-before.txt"
findmnt > "$local_base/mounts-before.txt"
ip -brief address > "$local_base/ip-address.txt" 2>&1 || true
ip route show table all > "$local_base/ip-routes.txt" 2>&1 || true
ss -lntup > "$local_base/listeners.txt" 2>&1 || true
ps -eo pid,ppid,user,state,rss,vsz,lstart,args --sort=-rss > "$local_base/processes.txt"
systemctl --user list-units --all --no-pager > "$local_base/user-units.txt" 2>&1 || true
systemctl --user list-unit-files --no-pager > "$local_base/user-unit-files.txt" 2>&1 || true
docker ps -a --no-trunc --size > "$local_base/docker-ps-all.txt" 2>&1 || true
docker images --digests --no-trunc > "$local_base/docker-images.txt" 2>&1 || true

for node in mj-spark-1 mj-spark-2 mj-spark-3; do
    scp -q -F zima-ssh-config scripts/remote_inventory.sh "$node:/home/mj-kang/Dev/experiment/remote_inventory_glm53_tp3.sh"
    ssh -F zima-ssh-config "$node" "bash /home/mj-kang/Dev/experiment/remote_inventory_glm53_tp3.sh '$stamp'" > "$local_base/${node}.remote-path.txt"
    remote_path=$(cat "$local_base/${node}.remote-path.txt")
    scp -q -F zima-ssh-config "$node:$remote_path/summary.json" "$local_base/spark-summaries/${node}.json"
    scp -q -F zima-ssh-config "$node:$remote_path/ARTIFACT_SHA256SUMS" "$local_base/spark-summaries/${node}.ARTIFACT_SHA256SUMS"
    scp -q -F zima-ssh-config "$node:$remote_path/rollback-commands.sh" "$local_base/spark-summaries/${node}.rollback-commands.sh"
done

for endpoint in \
    http://192.168.0.238:8888/health \
    http://192.168.0.234:8188/system_stats; do
    safe=$(printf '%s' "$endpoint" | tr '/:' '__')
    curl --connect-timeout 3 --max-time 10 -sS -D "$local_base/${safe}.headers" -o "$local_base/${safe}.body" -w '%{http_code}\n' "$endpoint" > "$local_base/${safe}.status" || true
done

find "$local_base" -type f ! -name ARTIFACT_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$local_base/ARTIFACT_SHA256SUMS"
chmod -R go-rwx "$local_base"
printf '%s\n' "$stamp"
