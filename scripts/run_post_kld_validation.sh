#!/usr/bin/env bash
set -euo pipefail

attempt=${1:?candidate attempt required}
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || {
    echo "invalid candidate attempt" >&2
    exit 2
}

stamp=20260906T034145Z
project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
base="/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt"
validation="$base/validation"
endpoint=http://192.168.0.238:8893
python=python3
mkdir -p "$validation"

jq -e '.phase == "complete" and (.completed_windows | length) == 4 and .error == null' \
    "$validation/kld/STATUS.json" >/dev/null
jq -e '.passed == true and .prediction_positions == 8188' \
    "$validation/kld/SCORE.json" >/dev/null
test ! -e "$base/guard/STOP.json"
curl -fsS --max-time 10 "$endpoint/health" >/dev/null

atomic_status() {
    local phase=$1 stage=${2:-} error=${3:-}
    jq -n --arg phase "$phase" --arg stage "$stage" --arg error "$error" \
        --arg attempt "$attempt" --arg updated "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
        '{schema:"glm53-full-exl3-tp3.post-kld-validation-status.v1",attempt:$attempt,phase:$phase,current_stage:(if $stage=="" then null else $stage end),updated_at:$updated,error:(if $error=="" then null else $error end)}' \
        > "$validation/POST_KLD_STATUS.json.tmp"
    mv "$validation/POST_KLD_STATUS.json.tmp" "$validation/POST_KLD_STATUS.json"
}

run_stage() {
    local stage=$1
    shift
    local started ended rc
    test ! -e "$base/guard/STOP.json"
    curl -fsS --max-time 10 "$endpoint/health" >/dev/null
    atomic_status running "$stage"
    started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    printf '%s\n' "$started" > "$validation/$stage.started-at"
    set +e
    "$@" > "$validation/$stage.controller.log" 2>&1
    rc=$?
    set -e
    ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    printf '%s\n' "$ended" > "$validation/$stage.ended-at"
    printf '%s\n' "$rc" > "$validation/$stage.controller.exit"
    if ! "$python" "$project/scripts/kernel_audit.py" --since "$started" --until "$ended" \
        --output "$validation/$stage.kernel-audit.json"; then
        atomic_status failed "$stage" "strict kernel audit failed"
        return 2
    fi
    if (( rc != 0 )); then
        atomic_status failed "$stage" "functional command exited $rc"
        return "$rc"
    fi
}

# The first immutable extended-probes.json used 128-token limits. Its JSON and
# both tool-call probes passed, while two semantically correct responses ended
# at the cap. Retry only those affected probes under distinct v2 receipts.
jq -e '.checks.json_schema == true and .checks.tool_auto == true and .checks.tool_forced == true' \
    "$validation/extended-probes.json" >/dev/null
if ! jq -e '.passed == true' "$validation/extended-probes-v2.json" >/dev/null 2>&1 \
    || ! jq -e '.passed == true' "$validation/extended-probes-v2.kernel-audit.json" >/dev/null 2>&1; then
    run_stage extended-probes-v2 "$python" "$project/scripts/extended_candidate_probes.py" \
        --endpoint "$endpoint" --output "$validation/extended-probes-v2.json" \
        --probes reasoning,code_repair
fi
run_stage strict-code-repair "$python" "$project/scripts/quick_candidate_probes.py" \
    --endpoint "$endpoint" --output "$validation/strict-code-repair.json" \
    --probes code_repair
run_stage long-context-v3 "$python" "$project/scripts/long_context_probe.py" \
    --endpoint "$endpoint" --output "$validation/long-context-v3.json" \
    --target-tokens 27000 --timeout-seconds 10800 --max-tokens 96 --mode raw-native
run_stage benchmark "$python" "$project/scripts/benchmark_candidate.py" \
    --endpoint "$endpoint" --output "$validation/benchmark.json"
run_stage sustained-one-hour "$python" "$project/scripts/sustained_workload.py" \
    --endpoint "$endpoint" --duration-seconds 3600 --state-dir "$validation/sustained-one-hour"

atomic_status complete
