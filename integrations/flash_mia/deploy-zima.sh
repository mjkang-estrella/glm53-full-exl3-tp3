#!/usr/bin/env bash
# Zima orchestrates an E3 layer build and TP2 launch. HF uploads are unrelated.
set -euo pipefail
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
recipe=/home/mj-kang/Dev/experiment/glm53-flash-mia-20260909
state=/home/mj-kang/Dev/state/glm53-flash-e3-20260909
cd "$project"
mkdir -p "$state"
exec 9>"$state/deploy.lock"
flock -n 9
trap 'printf "FAILED line=%s time=%s\n" "$LINENO" "$(date -u -Iseconds)" > "$state/PHASE"' ERR
date -u -Iseconds > "$state/started-at.txt"
printf 'BACKUP\n' > "$state/PHASE"
cp -p /srv/projects/librechat/librechat.yaml "$state/librechat.before.yaml"
cp -p /srv/projects/librechat/docker-compose.override.yml "$state/compose.before.yml"
for rank in 0 1 2; do
  ssh -F zima-ssh-config "mj-spark-$((rank+1))" "docker inspect glm53-k3-cand-speed-qualified-final-rank$rank" > "$state/k275-rank$rank.before.json"
done
printf 'STOP_K275\n' > "$state/PHASE"
bash scripts/stop_candidate_attempt.sh 20260907T224500Z speed-qualified-final > "$state/stop-k275.log" 2>&1
for node in mj-spark-1 mj-spark-2; do
  rows=$(ssh -F zima-ssh-config "$node" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader')
  test -z "$rows"
done
printf 'BUILD_E3_ON_SPARK2\n' > "$state/PHASE"
ssh -F zima-ssh-config mj-spark-2 "mkdir -p /home/mj-kang/Dev/logs/glm53-flash-e3-20260909; cd '$recipe'; docker build -f Dockerfile.e3-layer --build-arg BASE=glm53-exl3:e2-c190db1 --build-arg GLM53_RECIPE_STAMP=dc6936cea8fd7b2e7ee5b7a48a5aa193857ca489 -t glm53-flash-e3:dc6936c-20260909 . > /home/mj-kang/Dev/logs/glm53-flash-e3-20260909/build.log 2>&1"
printf 'SHIP_IMAGE_TO_SPARK1\n' > "$state/PHASE"
ssh -F zima-ssh-config mj-spark-1 'bash -o pipefail -c "ssh mj-spark-2 docker save --platform linux/arm64 glm53-flash-e3:dc6936c-20260909 | docker load"' > "$state/image-ship.log" 2>&1
printf 'LAUNCH_FLASH_E3\n' > "$state/PHASE"
for rank in 0 1; do
  node=mj-spark-$((rank+1))
  role=head
  [[ "$rank" == 0 ]] || role=worker
  name=glm53-flash-e3-$role-20260909
  ssh -F zima-ssh-config "$node" "tmux new-session -d -s flash-e3-guard-$rank \"while ! docker inspect -f '{{.State.Running}}' '$name' 2>/dev/null | grep -q true; do sleep 2; done; python3 '$project/scripts/watchdog.py' --container '$name' --state-dir '$state/watchdog-rank$rank' --reserve-gib 12\""
done
ssh -F zima-ssh-config mj-spark-1 "cd '$recipe'; bash start.sh" > "$state/launch.log" 2>&1
printf 'READY_FOR_VALIDATION\n' > "$state/PHASE"
