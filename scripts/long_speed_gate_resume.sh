#!/usr/bin/env bash
# Zima: qualify the smaller KV pool between benchmark and the next profile.
set -euo pipefail
controller_pid=${1:?paused controller pid}
[[ "$controller_pid" =~ ^[0-9]+$ ]]
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
root=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/speed/20260908
py=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/eval-venv-numpy2.3.3/bin/python
expected="$py $project/scripts/speed_graph_followup.py"
cd "$project"
deadline=$((SECONDS+600))
result=/home/mj-kang/Dev/benchmark/llama-benchy/results/k275-speed-20260908/mtp4-kv1g/result.json
until ssh -F zima-ssh-config mj-spark-1 "test -s '$result'"; do
  (( SECONDS < deadline )) || { echo 'benchmark did not finish; controller remains paused'; exit 2; }
  sleep 5
done
ssh -F zima-ssh-config mj-spark-1 "jq -e '(.benchmarks[0].tg_throughput.values|length)==3' '$result' >/dev/null"
started=$(date -u -Iseconds)
"$py" scripts/long_context_probe.py --endpoint http://192.168.0.238:8893 \
  --output "$root/long30k-kv1g-mtp4.json" --target-tokens 30000 \
  --mode raw-native --max-tokens 64 --timeout-seconds 600
jq -e '.passed == true' "$root/long30k-kv1g-mtp4.json" >/dev/null
python3 scripts/kernel_audit.py --since "$started" --until "$(date -u -Iseconds)" \
  --ssh-config zima-ssh-config --output "$root/long30k-kv1g-mtp4-kernel.json"
ps -p "$controller_pid" -o args= | grep -Fx "$expected" >/dev/null
kill -CONT "$controller_pid"
echo 'LONG_CONTEXT_PASSED_CONTROLLER_RESUMED'
