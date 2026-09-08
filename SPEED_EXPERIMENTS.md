# K275 decode-speed experiments, September 8

Keep the original K275 checkpoint, all experts/routing, TP3/DCP3, FP8 KV, 32K context and a 12 GiB host reserve. Do not change public routes or restore Flash/H3. Leave the fastest accepted profile running.

Use Spark 1's existing llama-benchy 0.4.0, pp2048/tg256, exact requested generation, three measured runs after warmup, concurrency 1, no prefix cache, generation latency mode. Record prompt throughput and first-response latency separately from decode throughput. The tool counts observed content/reasoning token IDs, which can exclude suppressed formatting tokens even with 256 requested tokens. Preserve raw request counts and the pinned tool commit; do not silently change the benchmark between profiles.

The current baseline is `k275-post-reencode-v1`, eager, NCCL1/512 KiB, 128-token prefill chunks and MTP off.

The previous K3 graph failure came from `LazyExpertStore.observe_ids`, specifically dynamic GPU boolean indexing followed by `.cpu().tolist()`. Resident execution has no eviction and does not need LFU bookkeeping. Skip this observer only for fully resident execution; preserve it for actual caching. Make DCP padding diagnostics one-shot instead of printing every layer/token. Neither change modifies routing or tensor arithmetic.

Test the optimized eager profile, then CUDA graphs with that synchronization removed. Compare NCCL2/256 KiB, NCCL4/1 MiB and NCCL8/2 MiB using the best accepted execution mode. Test MTP with draft TP1 and one, two and three speculative tokens if compatible and safe. If graphs plus MTP cannot fit, test eager MTP without lowering the memory reserve. Compare 256-token prefill chunks on the final promising profile. DCP remains enabled throughout.

Reject failed generations, coherence failures, CUDA/kernel faults, OOM, host-reserve violations and materially slower profiles. Run real native output probes before and after benchmarking. Capture speculative acceptance counters when MTP is enabled. Repeat the selected winner before leaving it loaded. Unsupported paths are recorded as failures, not assigned a throughput value.

After the primary matrix, test the installed `FULL_DECODE_ONLY` graph mode, which omits prefill graph capture. Test it without MTP and with the best accepted MTP profile. Also check the NCCL4/MTP interaction if the single-factor NCCL sweep preferred another channel count. These follow-up configurations retain the same memory reserve and target-model settings. The default graph mode's first measured allocation cost was 0.88 GiB.

The engine reported 99,070 KV tokens with MTP and the original 1.6875 GiB allocation per rank, despite a 32K/one-sequence limit. MTP1's minimum available memory was 12.044 GiB on the head rank. Test 1 GiB KV per rank without changing the 32K limit before graph/MTP follow-ups. Require reported capacity and a long-context generation check; do not lower context or the host reserve to qualify a profile. Also test the image's validated 16 ms spin-wait patch on the promising configuration.

The measured MTP ladder rose from 10.52 to 12.20 to 12.99 tok/s for one, two and three draft tokens. Add the launcher's supported four-token case after KV right-sizing passes, then compare the graph interaction using the best accepted draft length.

The original 1.6875 GiB KV / 256-token prefill / MTP3 case passed its short generation probes but hit the memory watchdog during llama-benchy: available memory was 12,827,885,568 bytes versus the 12,884,901,888-byte reserve. The benchmark connection reset after the protective stop. No speed is accepted for this failed case. Its kernel audit passed. Retry 256-token prefill only after the 1 GiB KV case passes; this removes unused allocation while retaining the 32K limit.

The 1 GiB / MTP3 / 256-token prefill retry passed at 13.573 decode tok/s and 197.03 prompt tok/s, with minimum head-rank availability 12.534 GiB.

The MTP4 / 1 GiB / 256-token prefill trial stalled during its second measured request. Worker progress stopped around 06:54:23 UTC; EngineCore repeatedly reported unavailable shared-memory broadcast blocks. All GPUs showed roughly 96% utilization at only 18 W, and no memory-stop marker appeared. The Zima controller had been paused to prevent a transition while planning a separate long-context check; its SSH benchmark child remained running. Resuming the controller did not restore model progress. The long-context helper was canceled before it ran, and the MTP4 containers were manually stopped. The complete kernel audit passed. This trial is excluded, and its root cause is not isolated. Do not call its partial output a valid benchmark or claim that four-token MTP universally fails.

Decode-only graphs without MTP passed at 7.511 tok/s and used 0.68 GiB for capture. Combining them with MTP3 and prefill256 required a reported 1.91 GiB capture allocation and failed the 12 GiB readiness gate; head-rank available memory reached 11.872 GiB. Test one smaller-prefill graph/MTP configuration at 128, whose eager footprint was lower, while retaining the same context, KV precision, 1 GiB pool and 12 GiB reserve. Do not relax the floor or promote a failed boot.

The later `final-confirm` trial of MTP3/prefill256/1 GiB KV also stalled, this time during the third measured request and without any controller pause. It was stopped manually. Mark larger-prefill MTP profiles ineligible for final selection, retaining all their measured values and identifying the repeat failure. The exact mechanism is unresolved. Recovery uses 128-token prefill and adds a fixed-workload benchmark no-progress timeout of 120 seconds. This does not govern long-context prefill.
