#!/usr/bin/env bash
# Restore the preserved containers and classify the encoder -> cold-load ->
# post-ready lifecycle. This script always attempts all three starts.
set -u

cd "$(dirname "$0")/.."
run_stamp=${1:?run stamp required}
candidate_start=${2:?candidate/encoder start timestamp required}
baseline=${3:?pre-stop service baseline JSON required}
evidence=${4:?restore evidence directory required}
mkdir -p "$evidence" "$evidence/timeline"
utc_now() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }
failed=0

if [ -s "$evidence/PASSED" ]; then
    exit 0
fi
candidate_end=$(utc_now)
printf '%s\n' "$candidate_end" > "$evidence/timeline/candidate-runtime-end.txt"
flash_start=$(utc_now)
printf '%s\n' "$flash_start" > "$evidence/timeline/flash-restore-start.txt"

ssh -F zima-ssh-config mj-spark-2 'docker start glm53-exl3-worker >/dev/null 2>&1 || true' &
p2=$!
ssh -F zima-ssh-config mj-spark-3 'docker start minimax-h3-comfy >/dev/null 2>&1 || true' &
p3=$!
wait "$p2" "$p3" || failed=1
ssh -F zima-ssh-config mj-spark-1 'docker start glm53-exl3-head >/dev/null 2>&1 || true' || failed=1

worker_death_observed=false
flash_ready=
head_code=0
h3_code=0
head_running=false
worker_running=false
for _ in $(seq 1 180); do
    head_code=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' \
        http://192.168.0.238:8888/health || true)
    h3_code=$(curl --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' \
        http://192.168.0.234:8188/system_stats || true)
    head_running=$(ssh -F zima-ssh-config mj-spark-1 \
        "docker inspect -f '{{.State.Running}}' glm53-exl3-head 2>/dev/null || echo false")
    worker_running=$(ssh -F zima-ssh-config mj-spark-2 \
        "docker inspect -f '{{.State.Running}}' glm53-exl3-worker 2>/dev/null || echo false")
    if [ "$head_running" != true ] || [ "$worker_running" != true ]; then
        worker_death_observed=true
        failed=1
        break
    fi
    if [ "$head_code" = 200 ] && [ "$h3_code" = 200 ]; then
        flash_ready=$(utc_now)
        printf '%s\n' "$flash_ready" > "$evidence/timeline/flash-ready.txt"
        break
    fi
    sleep 5
done
if [ -z "$flash_ready" ]; then failed=1; fi
printf '%s %s %s\n' "$head_code" "$worker_running" "$h3_code" > "$evidence/restored-health.txt"

generation="$evidence/restored-generation.json"
if [ "$head_code" = 200 ] && [ "$worker_running" = true ]; then
    remote_generation="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/bulk/$run_stamp/restored-generation.json"
    if ssh -F zima-ssh-config mj-spark-1 \
        "bash /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/scripts/verify_flash_generation.sh '$remote_generation'" \
        > "$evidence/restored-generation-summary.json"; then
        scp -q -F zima-ssh-config "mj-spark-1:$remote_generation" "$generation" || failed=1
    else
        failed=1
    fi
fi
if [ ! -s "$generation" ]; then
    jq -n '{schema:"glm53-full-exl3-tp3.missing-generation.v1",choices:[]}' > "$generation"
fi

sleep 15
post_end=$(utc_now)
printf '%s\n' "$post_end" > "$evidence/timeline/post-ready-end.txt"

head_after=$(ssh -F zima-ssh-config mj-spark-1 \
    "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' glm53-exl3-head 2>/dev/null || echo 'false true -1'")
worker_after=$(ssh -F zima-ssh-config mj-spark-2 \
    "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' glm53-exl3-worker 2>/dev/null || echo 'false true -1'")
h3_after=$(ssh -F zima-ssh-config mj-spark-3 \
    "docker inspect -f '{{.State.Running}} {{.State.OOMKilled}} {{.RestartCount}}' minimax-h3-comfy 2>/dev/null || echo 'false true -1'")
read -r head_running head_oom head_restart_after <<< "$head_after"
read -r worker_running worker_oom worker_restart_after <<< "$worker_after"
read -r h3_running h3_oom h3_restart_after <<< "$h3_after"

head_restart_before=$(jq -r '.containers["mj-spark-1:glm53-exl3-head"].restart_count' "$baseline")
worker_restart_before=$(jq -r '.containers["mj-spark-2:glm53-exl3-worker"].restart_count' "$baseline")
h3_restart_before=$(jq -r '.containers["mj-spark-3:minimax-h3-comfy"].restart_count' "$baseline")

jq -n --arg candidate_start "$candidate_start" --arg candidate_end "$candidate_end" \
    --arg flash_start "$flash_start" --arg flash_ready "$flash_ready" --arg post_end "$post_end" \
    '{schema:"glm53-full-exl3-tp3.lifecycle-timeline.v1",
      candidate_runtime:{start:$candidate_start,end:$candidate_end},
      protected_flash_rollback:{start:$flash_start,
        ready:(if $flash_ready == "" then null else $flash_ready end),timeout_seconds:900},
      post_ready:{start:(if $flash_ready == "" then null else $flash_ready end),end:$post_end}}' \
    > "$evidence/timeline.json"

jq -n \
    --arg head_running "$head_running" --arg head_oom "$head_oom" \
    --argjson head_before "$head_restart_before" --argjson head_after "$head_restart_after" \
    --arg worker_running "$worker_running" --arg worker_oom "$worker_oom" \
    --argjson worker_before "$worker_restart_before" --argjson worker_after "$worker_restart_after" \
    --arg h3_running "$h3_running" --arg h3_oom "$h3_oom" \
    --argjson h3_before "$h3_restart_before" --argjson h3_after "$h3_restart_after" \
    --arg death "$worker_death_observed" --arg flash_http "$head_code" --arg h3_http "$h3_code" \
    '{schema:"glm53-full-exl3-tp3.restored-service-state.v1",containers:{
      "mj-spark-1:glm53-exl3-head":{running:($head_running=="true"),oom_killed:($head_oom=="true"),restart_count_before:$head_before,restart_count_after:$head_after},
      "mj-spark-2:glm53-exl3-worker":{running:($worker_running=="true"),oom_killed:($worker_oom=="true"),restart_count_before:$worker_before,restart_count_after:$worker_after},
      "mj-spark-3:minimax-h3-comfy":{running:($h3_running=="true"),oom_killed:($h3_oom=="true"),restart_count_before:$h3_before,restart_count_after:$h3_after}},
      worker_death_observed:($death=="true"),health:{flash_http:($flash_http|tonumber),h3_http:($h3_http|tonumber)}}' \
    > "$evidence/service-state.json"

if ! python3 scripts/kernel_audit.py --timeline "$evidence/timeline.json" \
    --service-state "$evidence/service-state.json" --generation "$generation" \
    --output "$evidence/lifecycle-audit.json"; then
    failed=1
fi
find "$evidence" -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$evidence/SHA256SUMS"
if (( failed == 0 )); then
    date -u +%FT%TZ > "$evidence/PASSED"
    exit 0
fi
date -u +%FT%TZ > "$evidence/FAILED"
exit 1
