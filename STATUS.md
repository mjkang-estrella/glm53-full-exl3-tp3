> Historical snapshot, superseded September 8, 2026. Do not execute lifecycle, restoration, or encoding instructions here. Start with README.md and AGENTS.md.

# Current status: equal-width K3 stopped at capacity gate

Timestamp: 2026-09-04T13:51:08Z

## Outcome

Phase 0 and the corrected TP3 runtime gate passed. The corrected real BF16
layer-3 K3 qualification also completed and verified, but the measured
equal-width `768/768/768` capacity projection failed the required 12 GiB
per-Spark host-memory reserve. Bulk K3 encoding has not started.

## Passing evidence

- BF16 source: 282 shards, 1,506,667,387,408 bytes, pinned revision verified.
- Tensor plan: 58,368 routed matrices and 1,217 byte-exact carry tensors.
- Corrected synthetic K3 RMSE: about 0.132 to 0.133.
- Corrected actual TP3 loader, padding, fused MoE, and NCCL gate: passed.
- Real layer 3: 256 of 256 experts, 2,304 slice encodes, 9,216 tensors.
- Verified layer bytes: 4,109,847,248.
- Worst expert RMSE: 0.1322658401.
- Real-layer GPU peak: 84 C. Host availability stayed above about 114.6 GiB.
- Real-layer kernel window: no NVRM, Xid, OOM, or I/O faults.

## Capacity blocker

The measured layer projects to:

```text
Checkpoint:                 350,129,495,488 bytes, 326.0835 GiB
Optimistic rank requirement: 117,662,823,083 bytes
Optimistic reserve:            9,061,571,925 bytes, 8.4392 GiB
Required reserve:             12,884,901,888 bytes, 12 GiB
```

This lower bound already fails while assuming perfect passthrough and KV
sharding and zero engine, CUDA, NCCL, allocator, graph, activation, and indexer
overhead. Lowering context cannot recover enough memory because the 32K KV
lower bound is less than 1 GiB per rank.

## Live state

- No Codex or K3 encoder worker is active.
- The Codex resume turn ended after reaching its account usage limit.
- Flash head and worker are healthy and a real generation returned
  `CURRENT_FLASH_OK`.
- H3 was found stopped after the detached layer completed and was restored. Its
  `/system_stats` endpoint now returns HTTP 200.
- No K4 work or client/public-route change occurred.

## Authorized next direction

The operator authorized this redesign on 2026-09-04. Do not resume equal-width
bulk K3. Preserve the verified layer and its receipts
as evidence. Redesign K3 around an unpadded uneven physical layout such as
`768/640/640`, rotating which physical rank owns the 768-channel slice across
layers so the full-model weight total stays balanced per Spark. A fixed wide
rank would retain the current worst-rank capacity failure. Then qualify the
rank-specific loader, buffers, kernels, graph plan, NCCL reduction, and one real
layer before any bulk run. Removing the 12.5-percent expert padding should
recover about 32.06 GiB cluster-wide and roughly 10.69 GiB per Spark after
rotation. The resulting optimistic reserve is about 19 GiB per Spark before
unmodeled runtime overhead, enough to justify a bounded prototype but not a fit
claim. Resume the same Zima Codex thread with GPT-5.6 Sol at xhigh when its
account usage becomes available. The prior error reported a reset on 2026-09-07
at 4:00 PM local time.
