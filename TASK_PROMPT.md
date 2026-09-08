> Historical snapshot, superseded September 8, 2026. Do not execute lifecycle, restoration, or encoding instructions here. Start with README.md and AGENTS.md.

# Task

Build and start the K3-only TP3 quantization project described in `PLAN.md`.
You are running persistently on `mj-zima` and must orchestrate the three DGX
Sparks over SSH. Work autonomously within the safeguards below and leave
durable logs and resumable state.

## Current objective

The equal-width Phase 1 experiment is complete and rejected by capacity. Build
and qualify the rotating uneven 768/640/640 K3 layout. If and only if every new
Phase 1 gate passes, preserve the live services and dispatch resumable bulk K3
encoding across all three Sparks. Do not perform K4 promotion and do not build
the 3.25 bpw checkpoint in this run.

## Known state

- BF16 source is complete at
  `/mnt/unas-models/ZAI/GLM-5.3-BF16`.
- Pinned source revision:
  `304b8051cfb2b260b61ce0cbe330e02a98e73639`.
- Expected inventory: 282 safetensor shards and 1,506,667,387,408 weight bytes.
- The published R10 source bundle passed its hash verifier on Spark 3.
- Corrected equal-width K3 synthetic and three-Spark runtime gates passed.
- Corrected real layer 3 completed and verified all 256 experts, 2,304 slice
  encodes, and 9,216 tensors at worst RMSE 0.1322658401.
- Equal-width layer bytes were 4,109,847,248. Its optimistic 32K capacity
  projection left only 8.4392 GiB reserve per Spark and failed the 12 GiB gate.
- The equal-width payload is evidence only and cannot be reused by the uneven
  checkpoint.
- Current GLM-5.3-Flash runs on Sparks 1 and 2. H3 runs on Spark 3.

## Required work

1. Revalidate source identity and finish the remaining Phase 0 manifests.
2. Inventory live containers, images, environments, endpoints, GPU state,
   routes, and rollback commands before stopping anything.
3. Implement deterministic TP3 K3 encoding from BF16 using an unpadded rotating
   768/640/640 schedule. Normal layers rotate the wide rank every layer; layer
   78 assigns it to rank 2. Preserve all 256 experts, top-8 routing, index top-k
   2048, index frequency 4, and the MTP MoE layer. Carry all non-routed tensors
   byte-exact.
4. Add unit and synthetic tests for every width, offset, and rotation,
   identity-H encoding, deterministic seeds, pack/unpack equality, exact
   semantic channel coverage, and runtime reconstruction.
5. Before bulk encoding, run a uniform-K3 synthetic routed-expert layer through
   the actual `GlmMoeDsaForCausalLM` TP3 loader and execution path on all three
   Sparks. Verify all three uneven rotations, rank-specific geometry and
   buffers, NCCL reduction, reference/fused parity, and kernel health.
6. Encode one complete real BF16 MoE layer at its uneven geometry on Spark 3.
   Record every expert's K3 error, elapsed time, peak memory, temperature,
   power, clocks, source bytes, output bytes, and error logs. Do not perform K4
   promotion.
7. Project full checkpoint size and resident memory. Stop before bulk work if
   the initial 32K serving profile cannot retain at least 12 GiB host-memory
   reserve per Spark or if the runtime gate fails.
8. If all gates pass, create dated rollback artifacts, stop the relevant GPU
   services, and start resumable layer-queued K3 workers on all three Sparks.
   A fixed modulo assignment is acceptable only if it can recover work from a
   failed or slow node. Prefer a central lease queue with atomic receipts.
9. Keep progress, logs, manifests, and exact resume commands under the canonical
   `~/Dev` paths. Publish only sealed layer artifacts to the UNAS.

## Safety boundaries

- Never overwrite or delete the BF16 source, existing checkpoints, current
  Flash deployment, H3 configuration, or rollback artifacts.
- Do not copy the 1.4 TiB source onto Zima's local disk.
- Never run an encoder beside a large GPU service on the same Spark.
- Use bounded local staging, atomic output renames, and checksummed receipts.
- Maintain a 12 GiB host-memory reserve and stop on sustained swap thrashing.
- Stop and restore services on NVRM, Xid, kernel OOM, I/O failure, corrupt
  output, parity failure, or missing rollback state.
- Exception authorized on 2026-09-04: recovered `NV_ERR_NO_MEMORY` retries may
  occur only during the preserved Flash rollback's cold model load, profiling,
  and graph capture. Accept that rollback only if both ranks survive, neither
  container is OOM-killed, startup completes within its timeout, `/health` and
  a real generation pass, and a post-ready kernel audit is clean. Any Xid,
  worker death, persistent retry loop, timeout, failed canary, post-ready NVRM,
  or NVRM during encoder or candidate-runtime execution is still a hard stop.
- Do not use sudo workarounds. Stop and report the exact required command.
- Do not alter any client or public route during quantization.

## Completion behavior

Continue until Phase 1 has a measured pass or a concrete blocker. Start bulk K3
only after the gate passes. If blocked, leave Flash and H3 restored, write
`state/BLOCKED.md` with evidence and the exact next action, and exit cleanly.
If bulk K3 starts, write `state/STATUS.md` with worker sessions, progress paths,
resume commands, expected completion estimate, and rollback commands before
ending the Codex turn.
