# Authenticated LibreChat route

The operator authorized this cutover on September 8, after the speed campaign and private wrap-up. The public source repository is `https://github.com/mjkang-estrella/glm53-full-exl3-tp3`.

## Route and model identity

`https://ai.mj-kang.com` → existing Cloudflare tunnel → authenticated LibreChat on Zima → internal reasoning/attachment adapter → Spark 1 port 8893 → qualified TP3/DCP3 K275.

The former adapter target, Docker gateway port 18888, forwarded to the stopped Flash service. The adapter now targets `192.168.0.238:8893` directly over the existing LAN. The old SSH tunnel was left unchanged for other consumers. No new public port, unauthenticated inference endpoint, login bypass or account permission was introduced.

- Model-spec name: `GLM-5.3-K275-EXL3`.
- UI label: `GLM-5.3 K275 (32K)`.
- Actual upstream/API model ID: `GLM-5.3-K3-TP3-CANDIDATE`. This legacy identifier mounts K275, not K3.
- Model/prompt context limit in LibreChat: 32,768, replacing the stale 1,000,000 setting.
- Title generation uses the same actual upstream model ID.
- `addParams.chat_template_kwargs.enable_thinking: true` preserves the expected reasoning parser behavior. A preflight with this false exposed reasoning text in the final-content field; it is not used for this route.
- Adapter `TEXT_ONLY_MODEL=true`: retain document extraction and explicitly omit unsupported image/media input. Existing global writing policy, reasoning-field translation, tool normalization and MCP search configuration remain intact.

The GPU containers, runtime image, quantized weights, MTP, resident allocation, KV budget and watchdogs are unchanged. Only LibreChat's API container and the adapter were recreated.

## Configuration changes

In the existing adapter Compose service, retain all secrets, mounts, health checks and unrelated settings. Change only:

```yaml
environment:
  UPSTREAM_HOST: "192.168.0.238"
  UPSTREAM_PORT: "8893"
  TEXT_ONLY_MODEL: "true"
```

In the existing `librechat.yaml`, update the model spec, `models.default`, `titleModel`, display label and token-config key to the identities above. Set both `preset.maxContextTokens` and `tokenConfig.<actual-model-id>.context` to 32768. Keep the adapter base URL `http://deepseek-reasoning-adapter:8080/v1`, existing authentication configuration and MCP entries. Add to that custom endpoint:

```yaml
addParams:
  chat_template_kwargs:
    enable_thinking: true
```

Actual live configuration files and secrets are not distributed here.

## Tool-call compatibility patch

The tested backend emitted valid tool-call arguments but `finish_reason=stop`. `tool-finish-reason.patch` changes this to `tool_calls` only when calls exist, and ensures buffered streamed calls precede the terminal event. It preserves plain-text stops and does not relabel truncated `length` responses as successful.

Patch preimage SHA256: `0376411c6d26e45008f8dd6e8bfbef2472fc910ef1ab721507c1be4aa5a82ea2`.
Patched adapter SHA256: `af50bf2d5f7c87230920e828ec196dc118187bfa680dda0803bba7ae33434941`.

Apply only to a checked copy of the current adapter, not a stale local adapter or arbitrary upstream version:

```bash
patch --dry-run -p1 < /path/to/tool-finish-reason.patch
patch -p1 < /path/to/tool-finish-reason.patch
python3 /path/to/test_tool_finish.py local-adapters/deepseek_reasoning_adapter.py
```

The six new CPU contract tests and four existing attachment tests passed before deployment. Retain existing Unslop and attachment behavior; do not replace the whole adapter with this repository's old historical helper copies.

## Verification and rollback

Require Compose validation, adapter health, discovery from inside the LibreChat API container, exact-marker chat, streaming text, tool calls and streamed tool calls through the adapter. Confirm public HTTPS home/login and protected API authentication separately. The browser session used for verification was logged out, so it confirmed the login page, not a signed-in chat transcript.

The deployment backup is `/srv/projects/librechat/backups/glm53-k275-public-20260908` on Zima. It retains pre-cutover `librechat.yaml`, `docker-compose.override.yml`, and `deepseek_reasoning_adapter.py`. Restore those exact files and recreate only the affected API/adapter containers to undo application changes. The old upstream Flash service is still stopped, so reverting this configuration alone does not provide a working Flash model; do not restore GPU services automatically.

For a browser that cached the old model list, refresh and start a new conversation with `GLM-5.3 K275 (32K)`. Existing conversations may retain the old Flash model selection and need an explicit model switch.
