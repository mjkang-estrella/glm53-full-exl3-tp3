#!/usr/bin/env bash
# Zima one-shot continuation, not a recurring/scheduled task.
set -euo pipefail
root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/speed/20260908
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
deadline=$((SECONDS+10800))
while :; do
  phase=$(jq -r '.phase' "$root/STATUS.json")
  [[ "$phase" != needs_attention ]] || { echo 'primary campaign needs attention; no follow-up started'; exit 2; }
  if [[ -f "$root/RESULTS.json" ]] && [[ $(jq -r '.phase' "$root/RESULTS.json") == complete ]]; then break; fi
  (( SECONDS < deadline )) || { echo 'primary campaign timeout'; exit 2; }
  sleep 10
done
# Atomic replacement leaves any old file descriptors intact. Primary work
# has finished; no running startup script is modified in place.
for name in start_candidate_cluster_attempt.sh speed_campaign.py speed_graph_followup.py; do
  target="$project/scripts/$name"
  if [[ -e "$target" && ! -e "$target.pre-speed-followup-20260908" ]]; then cp "$target" "$target.pre-speed-followup-20260908"; fi
  cp "$root/next/$name" "$target.speed-next"
  mv "$target.speed-next" "$target"
done
target="$project/runtime/run_candidate_node.sh"
if [[ ! -e "$target.pre-speed-followup-20260908" ]]; then cp "$target" "$target.pre-speed-followup-20260908"; fi
cp "$root/next/run_candidate_node.sh" "$target.speed-next"
mv "$target.speed-next" "$target"
echo 'PRIMARY_COMPLETE_STARTING_GRAPH_AND_SPIN_FOLLOWUP'
exec /home/mj-kang/Dev/cache/glm53-full-exl3-tp3/eval-venv-numpy2.3.3/bin/python \
  "$project/scripts/speed_graph_followup.py"
