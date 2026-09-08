# K275 llama-benchy speed results

Best qualified profile: **13.06 decode tok/s** across 6 measured runs, versus 5.12 before tuning. Speedup: **2.55x**.
This is the original K275 model. Higher short-run results that failed repeatability or memory checks are not eligible.

Campaign phase: complete.
All cases use original K275 weights, all experts/routing, TP3/DCP3, 32K context, FP8 KV and a 12 GiB host-reserve guard.
Client: Spark 1, llama-benchy 0.4.0 at commit 446dd42fde2ebbaa1d68a0dfe9dc1e5b833f95ad, clean checkout.
Workload: pp2048, requested tg256 with exact-tg, three measured warm runs, concurrency 1, no cache, generation latency mode.
Decode throughput excludes prefill. The standard tool counts observed content/reasoning token IDs and can exclude suppressed formatting tokens; requested length and observed length are not conflated.

| Case | NCCL channels / KiB | Graphs | MTP tokens | Prefill chunk | KV GiB/rank | Decode tok/s | Change | Prompt tok/s | TTFR seconds | Outcome |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 / 512 | off | 0 | 128 | 1.688 | 5.121 ± 0.007 | +0.0% | 88.19 | 23.601 | passed |
| hotpath-eager | 1 / 512 | off | 0 | 128 | 1.688 | 6.731 ± 0.055 | +31.4% | 136.39 | 15.649 | passed |
| hotpath-graphs | 1 / 512 | default | 0 | 128 | 1.688 | 5.935 ± 0.579 | +15.9% | 116.41 | 18.582 | passed |
| nccl2 | 2 / 256 | off | 0 | 128 | 1.688 | 6.708 ± 0.064 | +31.0% | 143.03 | 14.673 | passed |
| nccl4 | 4 / 1024 | off | 0 | 128 | 1.688 | 6.747 ± 0.027 | +31.8% | 147.84 | 14.197 | passed |
| nccl8 | 8 / 2048 | off | 0 | 128 | 1.688 | 6.698 ± 0.044 | +30.8% | 146.67 | 14.309 | passed |
| mtp1 | 4 / 1024 | off | 1 | 128 | 1.688 | 10.520 ± 0.291 | +105.4% | 145.90 | 14.396 | passed |
| mtp2 | 4 / 1024 | off | 2 | 128 | 1.688 | 12.203 ± 0.935 | +138.3% | 144.83 | 14.511 | passed |
| mtp3 | 4 / 1024 | off | 3 | 128 | 1.688 | 12.991 ± 0.453 | +153.7% | 143.62 | 14.646 | passed |
| prefill256 | 4 / 1024 | off | 3 | 256 | 1.688 | n/a | n/a | n/a | n/a | rejected; see launch/probe logs |
| winner-confirm | 4 / 1024 | off | 3 | 128 | 1.688 | 12.377 ± 0.663 | +141.7% | 143.77 | 14.630 | passed |
| kv1g | 4 / 1024 | off | 3 | 128 | 1.000 | 12.950 ± 0.682 | +152.9% | 142.59 | 14.753 | passed |
| prefill256-kv1g | 4 / 1024 | off | 3 | 256 | 1.000 | 13.573 ± 0.681 | +165.0% | 197.03 | 10.774 | measured; disqualified on repeat |
| mtp4-kv1g | 4 / 1024 | off | 4 | 256 | 1.000 | n/a | n/a | n/a | n/a | rejected; see launch/probe logs |
| decode-only-graphs | 4 / 1024 | FULL_DECODE_ONLY | 0 | 128 | 1.000 | 7.511 ± 0.024 | +46.7% | 145.51 | 14.390 | passed |
| mtp-decode-only-graphs | 4 / 1024 | FULL_DECODE_ONLY | 3 | 256 | 1.000 | n/a | n/a | n/a | n/a | rejected; see launch/probe logs |
| spinwait16, 16 ms spin | 4 / 1024 | off | 3 | 256 | 1.000 | 12.710 ± 0.549 | +148.2% | 195.87 | 10.840 | measured; disqualified on repeat |
| final-confirm | 4 / 1024 | off | 3 | 256 | 1.000 | n/a | n/a | n/a | n/a | rejected; see launch/probe logs |
| mtp3-graphs-p128 | 4 / 1024 | FULL_DECODE_ONLY | 3 | 128 | 1.000 | n/a | n/a | n/a | n/a | rejected; see launch/probe logs |
| qualified-final | 4 / 1024 | off | 3 | 128 | 1.000 | 13.174 ± 0.890 | +157.3% | 142.70 | 14.730 | passed |

## Speculative acceptance

Counter deltas cover the benchmark, its warmup/latency requests and the post-benchmark generation probes. They are not per-run or code-workload acceptance guarantees.

| Case | Accepted draft tokens | Proposed draft tokens | Acceptance |
|---|---:|---:|---:|
| mtp1 | 476 | 588 | 81.0% |
| mtp2 | 624 | 936 | 66.7% |
| mtp3 | 688 | 1155 | 59.6% |
| winner-confirm | 681 | 1146 | 59.4% |
| kv1g | 716 | 1191 | 60.1% |
| prefill256-kv1g | 708 | 1104 | 64.1% |
| spinwait16 | 683 | 1191 | 57.3% |
| qualified-final | 699 | 1158 | 60.4% |

## Runtime changes

Resident execution no longer runs LFU eviction bookkeeping. Previously every layer synchronized routing IDs to CPU despite having all experts resident and no evictions. Non-resident caching retains its original accounting.
DCP padding diagnostics now print once rather than once per layer/token. Padding and attention arithmetic are unchanged.
The historical CUDA-graph failure stack points to dynamic routing-index filtering in observe_ids. The resident-only no-op removes that capture blocker. The combined hot-path case changes bookkeeping and logging, so its gain is not attributed to either change in isolation.

All benchmarks use the same tool/protocol. MTP acceptance and speed depend on text and can differ on code or agent workloads. A throughput winner is not a universal workload guarantee.
No public/client route was changed, and Flash/H3 remain stopped. Sealed K3 and K275 checkpoint files were not modified.

## Excluded trials

The original approximately 1.69 GiB KV / MTP3 / prefill256 run hit the host-memory guard, about 54 MiB below the 12 GiB floor. Its benchmark was interrupted, and no throughput is accepted. The 1 GiB KV retry is a separately measured configuration.

The four-token MTP trial stalled during its second measured request and was manually stopped. A controller pause had occurred during planning of a separate long-context check; its benchmark child remained active, and resuming the controller did not recover model progress. The long-context helper never ran. Kernel audit was clean. Root cause is unresolved, and partial output is not a valid speed result.

A later confirmation of MTP3 with prefill256 also stalled, without a controller pause. Larger-prefill MTP configurations are therefore ineligible for the final selection, including earlier short runs that completed. Their measurements remain visible for audit. The exact stall mechanism is unresolved.
Recovery benchmarks add a 120-second no-progress watchdog for the fixed pp2048 workload. This watchdog is not applied to legitimate long-context prefill.

The tested MTP3/decode-only-graph combinations failed the memory reserve at both 256- and 128-token prefill settings with 1 GiB KV. This is a constraint on those tested configurations, not a claim that all MTP/graph combinations are impossible.

Selected profile: `kv1g`.
Loaded confirmation attempt: `speed-qualified-final`.

## Final qualification

A real request used 30,039 prompt tokens and returned exactly `K3-NEEDLE-74291` with normal stopping in 186.95 seconds. Passed: True.
The model limit remains 32,768 tokens, TP3/DCP3, one sequence, FP8 KV at 1 GiB per rank. The winning profile uses MTP3, draft TP1 setting, NCCL4/1 MiB, 128-token prefill, eager execution and stock spin-wait.
No checkpoint weights, experts or routing were changed. No new KLD/top-1 score is claimed for this speed sweep.

Lowest observed host availability: 12.762 GiB. Model-container swap stayed at zero. Kernel audit passed: True.
