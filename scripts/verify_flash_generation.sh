#!/usr/bin/env bash
set -euo pipefail

evidence=${1:?evidence JSON path required}
container=${2:-glm53-exl3-head}
endpoint=${3:-http://127.0.0.1:8888/v1/chat/completions}

env_lines=$(docker inspect "$container" --format '{{range .Config.Env}}{{println .}}{{end}}')
api_key=$(printf '%s\n' "$env_lines" | sed -n 's/^VLLM_API_KEY=//p' | head -1)
model=$(printf '%s\n' "$env_lines" | sed -n 's/^SERVED_MODEL_NAME=//p' | head -1)
test -n "$model"
mkdir -p "$(dirname "$evidence")"
temporary="${evidence}.tmp.$$"
trap 'rm -f "$temporary"' EXIT

payload=$(jq -nc --arg model "$model" \
    '{model:$model,messages:[{role:"user",content:"Reply with exactly: RESTORED"}],temperature:0,max_tokens:64,chat_template_kwargs:{enable_thinking:false}}')
curl_args=(--fail --silent --show-error --max-time 180 -H 'Content-Type: application/json')
if [ -n "$api_key" ]; then
    curl_args+=(-H "Authorization: Bearer $api_key")
fi
curl "${curl_args[@]}" -d "$payload" "$endpoint" > "$temporary"
jq -e '
    .choices[0].finish_reason == "stop"
    and (.choices[0].message.content | type == "string" and length > 0)
' "$temporary" >/dev/null
chmod 600 "$temporary"
mv "$temporary" "$evidence"
trap - EXIT
jq '{id,model,finish_reason:.choices[0].finish_reason,content_length:(.choices[0].message.content|length),usage}' "$evidence"
