#!/usr/bin/env bash
set -euo pipefail

stamp=${1:?candidate test stamp required}
attempt=${2:?candidate attempt required}
start_index=${3:-0}
evidence=${4:?UNAS evidence root required}
expected_rows=${5:-2055}

[[ "$stamp" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo "invalid stamp" >&2; exit 2; }
[[ "$attempt" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || { echo "invalid attempt" >&2; exit 2; }
[[ "$start_index" =~ ^[0-3]$ ]] || { echo "start index must be 0..3" >&2; exit 2; }
[[ "$expected_rows" =~ ^[0-9]+$ ]] && (( expected_rows >= 2049 && expected_rows <= 32768 )) || { echo "invalid expected rows" >&2; exit 2; }
[[ "$evidence" == /mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-* ]] || {
    echo "evidence must be on the K275 UNAS tree" >&2
    exit 2
}

project=/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
reference=/mnt/unas-models/ZAI/GLM-5.3-BF16-full-logits-bounded-reference-427368f1
manifest=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260906T034145Z/teacher-reference-manifests/reference-full-panel/aggregate-manifest.json
python=/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/eval-venv-numpy2.3.3/bin/python
state=/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/$stamp/candidate/attempts/$attempt/validation/kld
mkdir -p "$state" "$evidence"

completed_json() {
    find "$evidence" -mindepth 1 -maxdepth 1 -type d -name 'confirmation-????' -printf '%f\n' \
        | sort | jq -Rsc 'split("\n")[:-1]'
}

write_status() {
    local phase="$1" current="${2-}" error="${3-}"
    local completed
    completed=$(completed_json)
    jq -n --arg phase "$phase" --arg current "$current" --arg error "$error" \
        --arg attempt "$attempt" --arg stamp "$stamp" \
        --arg evidence "$evidence" --argjson completed "$completed" \
        '{schema:"glm53-full-exl3-tp3.k275-kld-campaign-status.v1",stamp:$stamp,attempt:$attempt,phase:$phase,current_window:(if $current=="" then null else $current end),completed_windows:$completed,updated_at:(now|todateiso8601),evidence_root:$evidence,error:(if $error=="" then null else $error end)}' \
        > "$state/STATUS.json.tmp"
    mv "$state/STATUS.json.tmp" "$state/STATUS.json"
}

for index in $(seq "$start_index" 3); do
    window=$(printf 'confirmation-%04d' "$index")
    if [[ -s "$evidence/$window/CAPTURE_COMPLETE.json" ]]; then
        continue
    fi
    tokens="$reference/reference-full-panel/calibration/panel-v1/arrays/$window.tokens.npy"
    started=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    write_status capturing "$window"
    if ! "$python" "$project/scripts/capture_matched_logits.py" \
        --stamp "$stamp" --attempt "$attempt" --window-id "$window" \
        --tokens "$tokens" --output-root "$evidence" --timeout 7200 \
        --endpoint http://192.168.0.238:8893 --ssh-config "$project/zima-ssh-config" \
        --node mj-spark-1 --scheduler-singletons --expected-rows "$expected_rows" \
        > "$state/$window.controller.log" 2>&1; then
        write_status failed "$window" "capture controller failed; inspect $state/$window.controller.log"
        exit 2
    fi
    ended=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
    python3 "$project/scripts/kernel_audit.py" --since "$started" --until "$ended" \
        --ssh-config "$project/zima-ssh-config" --output "$state/$window.kernel-audit.json"
    jq -e '.passed == true and .expected_rows == 2047 and .expected_vocab == 154880' \
        "$evidence/$window/CAPTURE_COMPLETE.json" >/dev/null
done

write_status scoring
"$python" "$project/scripts/score_matched_logits.py" \
    --reference-root "$reference" --capture-root "$evidence" --manifest "$manifest" \
    --output "$state/SCORE.json" > "$state/SCORE.controller.log" 2>&1
write_status complete
