# K275 KLD experiments

## Outcome

No changed setting passed the repeatability gate. The original scales were restored and the candidate was left running.
Ten changes were tested against a fresh baseline: six global scale changes, three activation-clamp changes, and one layer-specific activation refit.

All cases keep 2.75 routed-expert bpw, all 256 experts per routed layer and the existing top-8 router.
The revised implementation changes the FP16 down-projection scale vectors directly. Trellis bits stay fixed.

## Development sweep

The two development windows were selection-0000 and selection-0001.

| Case | K2 gain | K3 gain | SwiGLU limit | KLD | Top-1 |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | 1 | 10.0 | 0.06814969 | 90.7670% |
| k2-095 | 0.95 | 1 | 10.0 | 0.07414471 | 89.6434% |
| k2-0975 | 0.975 | 1 | 10.0 | 0.06949309 | 89.9121% |
| k2-1025 | 1.025 | 1 | 10.0 | 0.06846037 | 90.0830% |
| k2-105 | 1.05 | 1 | 10.0 | 0.06708764 | 90.4250% |
| k3-099 | 1 | 0.99 | 10.0 | 0.06899287 | 90.5471% |
| k3-101 | 1 | 1.01 | 10.0 | 0.06778925 | 90.3517% |
| clamp20 | 1 | 1 | 20.0 | 0.06937088 | 90.2785% |
| clamp40 | 1 | 1 | 40.0 | 0.07037882 | 90.0098% |
| unclamped | 1 | 1 | unclamped | 0.07009114 | 90.2296% |
| refit-layer33 | 1 | 1 | 10.0 | 0.06790612 | 90.1808% |

## Confirmation

The selected setting was chosen using development KLD before confirmation scoring.
The four confirmation windows contain 8,188 prediction positions and all 154,880 vocabulary entries.
These are public confirmation data previously used for reporting, not a private final test set.

| Case | KLD | Top-1 |
|---|---:|---:|
| baseline | 0.04811580 | 92.7455% |
| k2-105 | 0.04828429 | 92.7088% |

KLD change: +0.00016849 nats, +0.350%.
Top-1 change: -0.0366 percentage points.

| Window | Baseline KLD | Candidate KLD | Change |
|---|---:|---:|---:|
| confirmation-0000 | 0.03402826 | 0.03167727 | -0.00235099 |
| confirmation-0001 | 0.10229647 | 0.10657198 | +0.00427551 |
| confirmation-0002 | 0.02715023 | 0.02662150 | -0.00052874 |
| confirmation-0003 | 0.02898825 | 0.02826642 | -0.00072183 |

First confirmation KLD/top-1 gate passed: False.

## Repeatability check

The same frozen settings were captured and scored a second time.

| Case | First KLD | Repeat KLD | First top-1 | Repeat top-1 |
|---|---:|---:|---:|---:|
| baseline | 0.04811580 | 0.04888221 | 92.7455% | 92.7455% |
| k2-105 | 0.04828429 | 0.04627999 | 92.7088% | 92.5867% |

| Case | Mean KLD across two runs | Mean top-1 | KLD spread between runs |
|---|---:|---:|---:|
| baseline | 0.04849901 | 92.7455% | 0.00076640 |
| k2-105 | 0.04728214 | 92.6478% | 0.00200430 |

Repeated improvement gate: False.
Loaded after the checks: baseline.
Runtime attempt: k275-quality-v2.
Native generation probes passed: 3/3.

The candidate's two-run mean KLD is lower, but its first confirmation regressed and its mean top-1 agreement is lower.
The gate required lower KLD on both paired comparisons and a mean top-1 loss no larger than 0.1 percentage point.
The baseline's aggregate top-1 percentage was identical, but teacher-agreement outcomes changed at 192/8188 individual positions.
The cause of this forward-run variation is unresolved. It is not a scorer arithmetic discrepancy, and these two repetitions do not establish a statistically reliable gain.

## Activation refit

Layer 33 used 4,096 input rows from separate code and reasoning windows, selection-0002 and selection-0003.
54 of 64 K2 experts had enough routed rows. Per-expert output gains were fitted against BF16 source weights using FP32 matrix arithmetic.
Every fifth input row was set aside to choose shrinkage. That local check error fell 2.958%, and the full-model refit case is listed above.
Because those check rows selected shrinkage, this is not an unbiased final estimate. Only one layer was refitted; ten K2 experts lacked enough routed calibration rows.
No local clipping effect was measured at this layer. Local reconstruction error is not full-model KLD.

## Next experiments

1. Resolve evaluation repeatability with identical-input baseline captures before accepting small KLD gains.
2. Pilot activation/Hessian-aware re-encoding from BF16, retaining the exact packed budget. This batch changed scales, not trellis codes.
3. Revisit which 64 experts per layer use K2: compare measured K2-versus-K3 activation error, rather than ranking only existing K3 weight error. Keep each rank's byte budget and all experts unchanged.

These are proposed follow-ups, not results from this batch. A full-model re-encode was not started.

## Runtime and evidence

Attempt k275-quality-v2 uses TP3/DCP3, interleave 1, 32K context, 128-token prefill chunks, FP8 KV, eager execution and MTP disabled.
Backbone expert weights use resident UVA across the three Sparks, without demand-loading experts from NVMe. The 12 GiB host-memory reserve and hardware watchdogs remained enabled.
BF16, K3 and K275 checkpoint artifacts and public/client routes were not overwritten. Flash/H3 remain stopped.
The initial output-multiplier experiment at 256-token chunks stalled and was stopped. It has no valid changed-scale KLD result.
Implementation and prefill chunk size both changed before the successful v2 run; the cause of the first stall was not isolated.
The scorer was checked against the original scorer on the prior four-window capture; mean KLD differed by only 6.1e-13 nats and top-1 was identical.

Receipts: /home/mj-kang/Dev/state/glm53-full-exl3-tp3/quality/20260908-scale-v2
Raw captures: /mnt/unas-models/ZAI/GLM-5.3-EXL3-K275-quality-experiments-20260908-scale-v2

Removed duplicate Spark-1 raw logit parts for 38 captures (45.23 GiB).
Both NAS and local file hashes were checked before removal. Local completion receipts remain; all raw data are recoverable from the verified NAS copies.

Final kernel audit passed: True (2026-09-08T02:40:00+00:00 through 2026-09-08T04:06:16+00:00).
Fault counts: mj-spark-1=0, mj-spark-2=0, mj-spark-3=0.
