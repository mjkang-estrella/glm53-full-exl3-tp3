# Progress and decisions

Dates below identify recorded artifacts, not the later Git import dates.

| Stage | Evidence / outcome | Decision |
|---|---|---|
| September 3–4: K3 geometry | Equal padded 768/768/768 failed capacity. Rotating uneven 768/640/640 passed the required gates. | Preserve all channels and all 256 experts; rotate the wide owner, MTP layer 78 wide owner rank 2. |
| September 6: sealed K3 | Four-window KLD 0.0339743205 nats; top-1 agreement 93.6004%. | Keep immutable K3 on NAS. |
| September 7: K275 | 64 K2 / 192 K3 experts per routed layer; KLD 0.0485539331 nats, top-1 92.6844% in the recorded DCP3 evaluation. | Keep original mixed checkpoint. These are calibration-window scores, not a broad intelligence benchmark. |
| September 8: quality trials | Ten scale adjustments produced no reliable accepted improvement. Layer-33 selection-only error +1.024%, reencode-only +1.640%, combined +57.017% on all-256 summed-output validation. | Reject the pilot changes. Layer errors are not whole-model KLD. Training-only selection did not use evaluation errors. |
| September 8: archive integrity | Hardlinked checksum ledger was overwritten during assembly; restored exactly from untouched manifest and checked against the original seal. 39,266 ledger entries passed the NAS audit. | Atomic metadata replacement and base-seal checks added; retain both archives. |
| September 8: resident hot path | Combined telemetry/logging cleanup: 5.121 → 6.731 decode tok/s. | Skip useless LFU host synchronization only when fully resident; log DCP padding once. Changes were not benchmarked independently. |
| September 8: NCCL | 2/4/8-channel tests: 6.708 / 6.747 / 6.698 tok/s. | Retain 4 channels / 1 MiB, but the small spread is not evidence of a large NCCL-specific gain. |
| September 8: MTP | One/two/three draft tokens: 10.520 / 12.203 / 12.991 tok/s. | Three-token MTP is the main measured additional speed gain. |
| September 8: graph/prefill trials | Decode-only graphs without MTP reached 7.511. Tested MTP3 graph cases failed memory reserve. MTP/prefill256 stalled on repeat, despite an earlier 13.573 result. | Keep eager, prefill128, stock spin; exclude failed configurations from selection. Exact stall mechanism unresolved. |
| September 8: qualification | Identical 1 GiB-KV profile: 12.950 and 13.174 tok/s, six-run mean 13.062 ± 0.801. Real 30,039-token prompt passed. | Leave qualified original K275 loaded on all three Sparks. |

## What the result does and does not establish

The improvement is a matched llama-benchy decode comparison, not prefill-plus-decode wall time. Winning prompt throughput averaged 142.64 tok/s; TTFR averaged 14.74 s. Sampling text, MTP acceptance, concurrency, and context affect throughput; code/agent workloads may differ.

The 32K limit is configured and a 30,039-token prompt plus 11 generated tokens was exercised. This is not a 1M-context qualification or proof of maximum possible context. No new KLD/top-1 score is claimed for the final speed profile. Earlier K3/K275 quality scores should not be compared with unrelated GGUF tests as if datasets and runtime protocols matched.

The qualification window retained 12.762 / 14.451 / 14.579 GiB minimum available host memory. Model-container swap was zero; host swap-in counters were nonzero, so do not claim the entire host had zero swap activity. Peak temperatures were 72 / 77 / 71 °C; the kernel audit passed with zero faults.

## Retained failures and cleanup

Keep [speed exclusions](../SPEED_RESULTS.md), [quality trials](../KLD_EXPERIMENT_RESULTS.md), [reencoding results](../REENCODE_RESULTS.md), and [archive receipts](../ARCHIVE_AND_REENCODE.md) with the accepted result. A briefly paused controller confounded the MTP4 stall, but the later prefill256/MTP3 stall occurred without a pause. Neither partial run is accepted.

Prior authorized cleanup removed 32 invalid raw parts (2.364 GiB) and three failed containers, retaining receipts and error evidence. Those invalid raw parts are not recoverable. A separate 45.2 GiB duplicate-capture cleanup had verified NAS copies. This wrap-up performs no model or evidence deletion.
