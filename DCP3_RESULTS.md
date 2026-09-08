# K275 DCP3 qualification

Date: 2026-09-08

The sealed 2.75 bpw K2/K3 checkpoint was tested in an isolated three-Spark
candidate. The production route and the sealed replicas were not changed.

## Qualified runtime path

- Attempt: `k275-dcp3-i1-cg-v1`
- Checkpoint: `GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z`
- TP3 / DCP3 / PP1, one sequence, 32,768-token model limit
- `max_num_batched_tokens=256`, eager execution, MTP off, prefix cache off
- FP8 KV with fixed `1,812,613,120` bytes per rank
- resident-UVA rank-local packs, 256 resident experts per layer
- NCCL1 with 524,288-byte buffers
- `cp_kv_cache_interleave_size=1`

All 76 layers loaded on every rank. `/health` returned 200 and the startup
kernel audit was strict-zero-fault. Host available memory at readiness was
14,405,447,680 / 15,548,387,328 / 15,854,678,016 bytes (ranks 0/1/2).

## Compatibility fixes exercised

The image's SM120 sparse MLA implementation did not advertise decode LSE,
its metadata lacked the optional `decode` field, its decode dispatch has no
66-head specialization, and the generic DCP reducer passed a three-rank
dimension directly to Triton's power-of-two `arange`. The runtime adapter
therefore:

1. enables SM120 LSE and filters sparse indices per DCP KV shard;
2. supplies `decode=None` for the older sparse metadata;
3. executes gathered GLM heads as 64 + 2 padded to 8, then concatenates;
4. pads the gathered LSE rank dimension 3 -> 4 with `-inf`.

Each change is runtime-only and has dated rollback copies on Zima.

## Quality and long context

The four-window matched BF16-logit campaign completed all 8,188 positions:

| Metric | DCP3 K275 |
|---|---:|
| Mean `KL(BF16 || K275)` | 0.0485539331 nats |
| Argmax agreement | 92.6844% |
| Mean top-5 overlap | 83.5515% |

This is consistent with the prior DCP1 K275 score (`0.0485110415` nats,
`92.6111%` top-1); DCP did not add a material fidelity loss. It remains worse
than the sealed 3.0 bpw K3 reference (`0.0339743205` nats, `93.6004%` top-1),
so K275 is not promoted.

Raw-native 24,024-token retrieval passed exactly with
`K3-NEEDLE-74291` and `finish_reason=stop` in 133.65 seconds. The chat-template
variant at the same length failed only its exact-output gate because it spent
the 32-token budget explaining the task and stopped at the length limit.

The bounded short task suite was coherent and matched earlier K275 behavior:
code, code-complex, Korean, reasoning, and automatic tool-call checks passed;
strict exact-output, arithmetic, and stream-finish assertions failed due the
model's explanation/reasoning formatting under the pinned template.

## Throughput sample

Three warm single-stream samples under this DCP3 configuration measured a
median `6.1387` generated tok/s and median TTFT `0.4106` seconds. The reusable
benchmark JSON retains stale DCP1/1024 labels in its generic configuration
block; the authoritative runtime settings are the READY record above.

The candidate remains experimental and isolated. Keep the sealed K3 reference
as the quality baseline; do not alter the public route based on this result.
