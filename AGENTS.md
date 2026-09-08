# GLM-5.3 agent handoff

- Read README.md, docs/REPRODUCE.md, docs/DEPENDENCIES.md, and SPEED_RESULTS.md before operational work.
- This is the completed K3/K275 and speed campaign. PLAN.md, TASK_PROMPT.md, STATUS.md, old lifecycle tests, and docs/history are historical evidence, not instructions to resume encoding or restore Flash/H3.
- Control Zima directly over SSH; use Zima tmux for authorized long-running work. Do not start or message a separate Codex task unless explicitly requested.
- Default to read-only inspection. Keep the qualified original K275 service running on all three Sparks. No automatic Flash/H3 restoration, public-route changes, benchmark reruns, reencoding, or cleanup.
- Last qualified attempt: speed-qualified-final, port 8893, legacy alias GLM-5.3-K3-TP3-CANDIDATE. Verify mounts and live state; actual checkpoint is K275.
- Preserve all 256 experts, official top-8 routing, sparse index policy, and sealed checkpoint bytes. Original K275 beat the tested fixed-budget quality changes. No claimed new KLD score from the speed campaign.
- Winner: resident UVA, TP3/DCP3, MTP3/draft TP1, eager, prefill128, NCCL4/1MiB, FP8 KV 1GiB/rank, 32K, one sequence, stock spin, 12GiB host reserve and independent watchdogs.
- Fail closed on memory-floor violations, model-container swap, NVRM/Xid, kernel OOM or filesystem errors. Never weaken a guard to complete a run. Old Flash cold-load exceptions do not authorize candidate faults.
- A load/health response is not qualification. Require normal real generation, matched throughput runs, long-context retrieval, kernel audit and memory evidence. The historical gate used low-reasoning native text for exact retrieval, not a high-reasoning intelligence evaluation.
- Keep failed measurements. Selection requires accepted AND eligible_for_selection; do not promote prefill256/MTP or tested MTP3/graph cases from partial or short results. A renewed experiment needs a changed hypothesis, new label/output directory, and bounded guards.
- A fixed pp2048 benchmark may use the 120-second no-progress guard. Do not apply it to valid long-context prefill. Do not SIGSTOP controllers while child workloads continue.
- A restart is disruptive: inspect the exact active attempt, intentionally stop only it when authorized, then use start_speed_winner.sh with a new unique label. Do not rerun speed_campaign.py as a restart mechanism.
- Read each node's /home/mj-kang/Dev/README.md before adding files. Checkpoints belong only in Dev/models; logs/state/cache/benchmark code belong in their canonical Dev subdirectories. Never stage model weights on Zima root storage.
- BF16 and sealed K3/K275 archives are immutable. Metadata updates must replace files atomically, never write through hardlinks. Cleanup requires exact targets, verified retained copies or explicit acceptance of loss, and receipts.
- Preserve SSH host-key checks and the Sync-managed ring. No sudo workarounds; stop for an exact operator command when needed.
- Commit only scoped source, docs and curated results. No credentials, raw JSONL/captures, images, weights or compiler caches. Source rsync must exclude .git and local artifact directories. No public push without explicit authorization.
- Retrospective commits reconstruct the source progression. Do not claim every historical commit independently passed the final gates.
