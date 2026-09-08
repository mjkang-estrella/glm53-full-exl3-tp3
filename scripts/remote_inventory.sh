#!/usr/bin/env bash
set -u

stamp=${1:?UTC inventory stamp required}
host=$(hostname -s)
base="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/rollback/${stamp}/${host}"
mkdir -p "$base/containers" "$base/images"
chmod 700 "$base" "$base/containers" "$base/images"
umask 077

date -u +%FT%TZ > "$base/captured-at.txt"
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
if [ -f "$project/scripts/kernel_audit.py" ]; then
    cd "$project"
    python3 scripts/kernel_audit.py --write-policy "$base/lifecycle-policy.json"
fi
uname -a > "$base/uname.txt"
free -b > "$base/free-before.txt"
swapon --show --bytes > "$base/swap-before.txt" 2>&1 || true
df -B1 -T > "$base/filesystems-before.txt"
findmnt > "$base/mounts-before.txt"
ip -brief address > "$base/ip-address.txt" 2>&1 || true
ip route show table all > "$base/ip-routes.txt" 2>&1 || true
ss -lntup > "$base/listeners.txt" 2>&1 || true
ps -eo pid,ppid,user,state,rss,vsz,lstart,args --sort=-rss > "$base/processes.txt"
systemctl --user list-units --all --no-pager > "$base/user-units.txt" 2>&1 || true
systemctl --user list-unit-files --no-pager > "$base/user-unit-files.txt" 2>&1 || true
docker ps -a --no-trunc --size > "$base/docker-ps-all.txt" 2>&1 || true
docker images --digests --no-trunc > "$base/docker-images.txt" 2>&1 || true
docker stats --no-stream > "$base/docker-stats.txt" 2>&1 || true
nvidia-smi -q > "$base/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi pmon -c 1 > "$base/nvidia-pmon.txt" 2>&1 || true
nvidia-smi --query-gpu=timestamp,name,uuid,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.total,memory.used,memory.free --format=csv,noheader > "$base/gpu-summary.csv" 2>&1 || true
journalctl -k --since '-2 hours' --no-pager > "$base/kernel-last-2h.txt" 2>&1 || true

running="$base/running-containers.txt"
docker ps --format '{{.Names}}' | sort > "$running" 2>/dev/null || :
: > "$base/rollback-commands.sh"
chmod 700 "$base/rollback-commands.sh"
while IFS= read -r name; do
    [ -n "$name" ] || continue
    docker inspect "$name" > "$base/containers/${name}.inspect.json"
    image=$(docker inspect -f '{{.Image}}' "$name")
    safe_image=$(printf '%s' "$image" | tr '/:@' '____')
    docker image inspect "$image" > "$base/images/${safe_image}.inspect.json" 2>/dev/null || true
    printf 'docker start %q\n' "$name" >> "$base/rollback-commands.sh"
done < "$running"

for tree in \
    /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3 \
    /home/mj-kang/Dev/experiment/glm53-exl3-2spark-c190db1 \
    /home/mj-kang/Dev/experiment/glm53-exl3-sm120-22-tp3 \
    /home/mj-kang/Dev/experiment/glm53-exl3-tp3 \
    /home/mj-kang/Dev/runtime/minimax-h3-comfy; do
    if [ -d "$tree" ]; then
        label=$(basename "$tree")
        find "$tree" -xdev -type f -maxdepth 4 -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum > "$base/${label}.sha256"
    fi
done

python3 - "$base" <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
containers = []
for path in sorted((base / "containers").glob("*.inspect.json")):
    doc = json.loads(path.read_text())[0]
    state = doc.get("State", {})
    host = doc.get("HostConfig", {})
    mounts = [
        {"source": m.get("Source"), "destination": m.get("Destination"), "mode": m.get("Mode"), "rw": m.get("RW")}
        for m in doc.get("Mounts", [])
    ]
    containers.append({
        "name": doc.get("Name", "").lstrip("/"),
        "id": doc.get("Id"),
        "image_id": doc.get("Image"),
        "image_ref": doc.get("Config", {}).get("Image"),
        "command": doc.get("Config", {}).get("Cmd"),
        "entrypoint": doc.get("Config", {}).get("Entrypoint"),
        "network_mode": host.get("NetworkMode"),
        "ipc_mode": host.get("IpcMode"),
        "shm_size": host.get("ShmSize"),
        "restart_policy": host.get("RestartPolicy"),
        "mounts": mounts,
        "running": state.get("Running"),
        "started_at": state.get("StartedAt"),
        "health": state.get("Health", {}).get("Status"),
    })
summary = {
    "schema": "glm53-full-exl3-tp3.spark-inventory.v1",
    "host": pathlib.Path("/etc/hostname").read_text().strip(),
    "captured_at": (base / "captured-at.txt").read_text().strip(),
    "secure_inventory_path": str(base),
    "containers": containers,
}
(base / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
PY

find "$base" -type f ! -name ARTIFACT_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$base/ARTIFACT_SHA256SUMS"
chmod -R go-rwx "$base"
printf '%s\n' "$base"
