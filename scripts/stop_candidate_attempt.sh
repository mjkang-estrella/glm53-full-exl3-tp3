#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?real-test stamp required}
attempt=${2:?candidate attempt required}
[[ "$stamp" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || { echo "invalid stamp" >&2; exit 2; }
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || { echo "invalid attempt" >&2; exit 2; }

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
state="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt"
prefix="glm53-k3-cand-$attempt"
nodes=(mj-spark-1 mj-spark-2 mj-spark-3)
test -s "$state/READY.json"
test ! -e "$state/STOPPED.json"
mkdir -p "$state/ranks"

# Stop only Zima guards that name this exact attempt. A guard left running
# would interpret the intentional container stop as a candidate fault.
while IFS=$'\t' read -r session command; do
    if [[ "$command" == *"guard_candidate_cluster.sh $stamp"*"$attempt"* ]]; then
        tmux kill-session -t "$session" 2>/dev/null || true
    fi
done < <(tmux list-panes -a -F '#{session_name}\t#{pane_start_command}' 2>/dev/null || true)

stop_requested_at=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
for rank in 0 1 2; do
    node=${nodes[$rank]}
    ssh -F zima-ssh-config "$node" \
        "docker stop --time 60 '$prefix-rank$rank' >/dev/null 2>&1 || true; \
         tmux kill-session -t '$prefix-watchdog-$rank' 2>/dev/null || true; \
         tmux kill-session -t '$prefix-log-$rank' 2>/dev/null || true" &
done
wait
stopped_at=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)

for rank in 0 1 2; do
    node=${nodes[$rank]}
    remote_watchdog="$state/rank-$rank/watchdog"
    ssh -F zima-ssh-config "$node" "docker inspect '$prefix-rank$rank'" \
        > "$state/ranks/rank-$rank.final-state.json"
    ssh -F zima-ssh-config "$node" "docker logs '$prefix-rank$rank'" \
        > "$state/ranks/rank-$rank.full.log" 2>&1
    scp -q -F zima-ssh-config "$node:$remote_watchdog/metrics.csv" \
        "$state/ranks/rank-$rank.metrics.csv"
done

started=$(cat "$state/started-at.txt")
kernel_clean=true
python3 "$project/scripts/kernel_audit.py" --since "$started" --until "$stopped_at" \
    --output "$state/final-kernel-audit.json" || kernel_clean=false
memory_clean=true
python3 "$project/scripts/summarize_watchdog_metrics.py" --attempt "$attempt" \
    --attempt-dir "$state" --output "$state/memory-summary.json" || memory_clean=false

all_clean=true
for rank in 0 1 2; do
    jq -e '.[0].State.Running == false and .[0].State.OOMKilled == false' \
        "$state/ranks/rank-$rank.final-state.json" >/dev/null || all_clean=false
done
[[ "$kernel_clean" == true ]] || all_clean=false
[[ "$memory_clean" == true ]] || all_clean=false

jq -n --arg attempt "$attempt" --arg stop_requested_at "$stop_requested_at" \
    --arg stopped_at "$stopped_at" \
    --argjson clean "$all_clean" \
    '{schema:"glm53-full-exl3-tp3.candidate-intentional-stop.v1",attempt:$attempt,stop_requested_at:$stop_requested_at,stopped_at:$stopped_at,all_ranks_stopped:true,oom_killed:false,kernel_and_memory_clean:$clean,automatic_service_restore_attempted:false}' \
    > "$state/STOPPED.json.tmp"
mv "$state/STOPPED.json.tmp" "$state/STOPPED.json"
[[ "$all_clean" == true ]]
