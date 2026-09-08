# Historical agent rules, superseded

Do not execute these instructions. Current AGENTS.md and later user decisions take precedence.

# GLM-5.3 full K3 TP3 execution rules

- Read `PLAN.md` and `TASK_PROMPT.md` before acting.
- Zima is the always-on control node. It has no GPU and only about 35 GB of
  local free storage. Never copy model weights to Zima's root filesystem.
- The immutable BF16 source is
  `/mnt/unas-models/ZAI/GLM-5.3-BF16` at revision
  `304b8051cfb2b260b61ce0cbe330e02a98e73639`.
- Use the UNAS for source and sealed artifacts. Use each Spark's local NVMe for
  active encoding, caches, logs, and serving.
- Do not modify, rename, move, or delete BF16 source files.
- The equal-width 768/768/768 K3 layout is rejected by measured capacity. Keep
  its verified layer immutable as evidence and never resume its bulk encode.
- The active design uses rotating uneven 768/640/640 expert slices with no
  expert-channel padding. Rotate the 768-channel owner across normal layers and
  assign the layer-78 MTP wide slice to rank 2.
- Do not begin the bulk encode until all three uneven rotations pass the
  synthetic TP3 runtime gate and one complete real uneven-layer K3
  qualification and capacity projection pass.
- Preserve and verify GLM-5.3-Flash on Sparks 1 and 2 and H3 on Spark 3 before
  stopping them. Restore them after a bounded failed test.
- Never run a large encoder beside an active large GPU service on the same
  Spark.
- Keep at least 12 GiB host memory available. Use a separate watchdog and stop
  the encoder on sustained swap, NVRM, Xid, kernel OOM, or filesystem errors.
- The operator authorized known recovered `NV_ERR_NO_MEMORY` retries only while
  the protected Flash rollback is cold-loading. Allow them only if both ranks
  survive, neither container is OOM-killed, startup finishes within the bounded
  timeout, health and a real generation pass, and no NVRM event occurs after
  readiness. Any Xid, death, timeout, failed canary, persistent retry loop, or
  post-ready NVRM remains a hard stop. Encoder and candidate-runtime windows
  must be NVRM-clean.
- Do not use `sudo` workarounds. If a required step needs sudo, write the exact
  command and stop at that boundary.
- Keep new code under `/home/mj-kang/Dev/experiment`, caches under
  `/home/mj-kang/Dev/cache`, logs under `/home/mj-kang/Dev/logs`, state under
  `/home/mj-kang/Dev/state`, and checkpoint-only artifacts under
  `/home/mj-kang/Dev/models`.
- Do not change Codex Router, ZCode, LibreChat, the Zima adapter, or the public
  model route during encoder development.
- A load message or allocated KV cache is not success. Require real generation,
  numerical gates, kernel health, and rollback verification.
