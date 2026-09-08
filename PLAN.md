> Historical snapshot, superseded September 8, 2026. Do not execute lifecycle, restoration, or encoding instructions here. Start with README.md and AGENTS.md.

# GLM-5.3 full EXL3 TP3 quantization plan

Date: 2026-09-03

## Goal

Build and evaluate one TP3-native uniform-K3 EXL3 checkpoint of the full
`zai-org/GLM-5.3-BF16` model on the three DGX Sparks. Quantize every routed
expert at 3.0 bpw and preserve the complete per-expert error ledger required
for a possible later K4 promotion.

The current run stops after the K3 checkpoint is assembled, served, and
evaluated. A mixed K3/K4 3.25 bpw checkpoint is a deferred follow-up decision.
Do not begin K4 encoding automatically.

The equal-width padded TP3 experiment completed one real layer and failed its
optimistic capacity gate. The active hypothesis is now that an unpadded,
rotating uneven TP3 K3 layout can fit safely while retaining enough of full
GLM-5.3's coding and reasoning advantage to justify its lower context and
speed. Fitting the three Sparks is necessary, but it is not sufficient.

The current GLM-5.3-Flash service remains the production rollback target until
the K3 checkpoint passes every gate.

## Decisions already made

- Quantize from the BF16 source. Do not quantize the official FP8 checkpoint
  and do not repartition the released TP4 EXL3 checkpoint.
- Retain all 256 routed experts and official top-8 routing.
- Preserve the official sparse-index policy, including `index_topk=2048` and
  `index_topk_freq=4`.
- Keep attention, indexer, router, dense layers, shared experts, embeddings,
  norms, and `lm_head` in their source precision for the first checkpoint.
- Quantize the normal MoE layers and the MTP MoE layer.
- Use FP8 KV for serving tests.
- Validate the target model with speculative decoding disabled before enabling
  the in-checkpoint MTP3 path.
- Build, freeze, and evaluate K3 before deciding whether to produce a mixed
  K3/K4 checkpoint.
- Use local DGX Spark NVMe for active encoding and serving. Use the UNAS as the
  authoritative source and artifact library.
- Use all three Sparks for the bulk encode after a one-layer qualification.
- Perform K3 quantization, BF16-teacher KLD scoring, and serving qualification
  in the homelab.
- Defer new official-FP8 and NVFP4 layer-streaming reference implementations.
  They are not prerequisites for the K3 development decision.
- Do not rent cloud GPUs. A cloud reference run requires a separate future
  decision and explicit approval.

## Homelab roles

| System | Role |
|---|---|
| MacBook | Control surface, orchestration, reports, and final artifact review |
| `mj-zima` | Always-on Hugging Face download and UNAS control node |
| UNAS Pro 4 | Authoritative BF16 source, completed checkpoint archive, and receipts |
| `mj-spark-1` | Encoder worker 1, later TP3 serving head |
| `mj-spark-2` | Encoder worker 2, later TP3 serving worker |
| `mj-spark-3` | Initial encoder qualification, encoder worker 3, later TP3 serving worker |

The BF16 source completed downloading through Zima to:

```text
/mnt/unas-models/ZAI/GLM-5.3-BF16
```

Pin the source to revision:

```text
304b8051cfb2b260b61ce0cbe330e02a98e73639
```

Verified BF16 source inventory:

```text
282 safetensor shards
1,506,667,387,408 weight bytes
about 1.403 TiB
```

Do not serve or perform random-access quantization directly from NFS. Copy the
source shards needed by the current work unit to local NVMe, write the output
locally, verify it, and then publish the completed layer artifact to the UNAS.

## Model structure

| Field | Value |
|---|---:|
| Architecture | `GlmMoeDsaForCausalLM` |
| Hidden size | 6,144 |
| Normal transformer layers | 78 |
| Dense layers | 0 through 2 |
| Normal MoE layers | 3 through 77 |
| MTP MoE layer | 78 |
| Routed experts per MoE layer | 256 |
| Selected experts per token | 8 |
| Shared experts per layer | 1 |
| Semantic expert intermediate size | 2,048 |
| Attention heads | 64 |
| KV latent rank | 512 |
| Native context limit | 1,048,576 |
| Vocabulary size | 154,880 |

The quantized scope contains 76 MoE layers when the MTP layer is included.
Each routed expert has `gate_proj`, `up_proj`, and `down_proj` matrices:

```text
76 layers * 256 experts * 3 projections = 58,368 semantic matrices
```

## Why a new TP3 encoding is required

The released `davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw` checkpoint is encoded for
TP4. Every routed-expert projection has four independently encoded trellis
slices named `rank0` through `rank3`. Its qualified loader selects the
serialized slice matching the live tensor-parallel rank.

A TP3 launch would consume ranks 0 through 2 and omit rank 3. Changing metadata
would therefore remove one quarter of every routed-expert projection. The
output might remain fluent while being numerically wrong.

The new checkpoint must encode the BF16 source directly for the physical TP3
layout.

## TP3 physical geometry

The semantic expert intermediate size does not divide by three:

```text
2048 / 3 = 682.666...
```

EXL3 trellis slices require 128-channel alignment. The rejected equal-width
layout used 768 channels on every rank:

```text
640 * 3 = 1920
768 * 3 = 2304
```

That layout produced a valid layer but added 12.5 percent physical padding and
failed capacity. Do not use it for bulk encoding. Preserve its layer-3 artifact
and receipts as rejected design evidence.

Use an unpadded rotating uneven layout:

```text
Semantic expert width: 2048
Physical expert width: 2048
Per-layer rank widths: one 768-channel slice and two 640-channel slices
Physical padding: 0 channels
```

For normal MoE layers 3 through 77, rotate the wide slice by layer:

```text
layer 3: rank widths [768, 640, 640], offsets [0, 768, 1408]
layer 4: rank widths [640, 768, 640], offsets [0, 640, 1408]
layer 5: rank widths [640, 640, 768], offsets [0, 640, 1280]
repeat every three layers
```

Layers 3 through 77 contain exactly 75 layers, so each rank owns the wide slice
25 times. Assign the extra MTP layer 78 wide slice to rank 2, keeping the serving
head rank from receiving the small unavoidable imbalance.

Projection slices are rank-specific:

| Projection | Semantic shape | Rank-local shape |
|---|---:|---:|
| `gate_proj` | `[2048, 6144]` | `[width_for_rank, 6144]` |
| `up_proj` | `[2048, 6144]` | `[width_for_rank, 6144]` |
| `down_proj` | `[6144, 2048]` | `[6144, width_for_rank]` |

All widths and offsets are multiples of 128. No quantized padding payload or
runtime expert-channel mask is permitted. The loader, buffers, kernels, graph
plans, and checkpoint metadata must read each layer's explicit width and offset
for every rank. A fixed `768/640/640` assignment is insufficient because its
wide rank would retain the equal-width worst-rank memory failure.

The verified equal-width layer measured 4,109,847,248 bytes. Scaling its routed
payload by 2048/2304 and retaining 37,781,104,640 bytes of BF16 passthrough gives
an initial uneven-checkpoint projection of about 293.76 GiB. Rotating the wide
slice projects roughly 19.19 GiB optimistic reserve per Spark at 32K, before
unmodeled runtime overhead. These are design estimates, not fit evidence.

Other TP3 boundaries also require physical padding:

```text
Attention heads: 64 semantic to 66 physical, 22 per rank
Vocabulary: 154,880 semantic to 154,944 physical
```

Keep the public model configuration semantic. Record physical padding in
checkpoint and runtime metadata rather than changing the visible architecture.

### Deferred alternatives

Do not use these alternatives in the first build:

- `704 * 3 = 2112` reduces padding but violates EXL3's 128-channel slice
  requirement.
- Equal `[768, 768, 768]` slices have already failed the optimistic capacity
  gate and must not be resumed for bulk encoding.
- A 64-channel EXL3 block would create a new format and require new quantizer,
  kernel, and KLD qualification work.
- Expert parallelism could preserve width 2,048 but would require an unverified
  EXL3 all-to-all runtime over the three-node RoCE ring.

Revisit a 64-channel format or expert parallelism only if the rotating uneven
layout fails its runtime or capacity gates.

## Quantization method

Use the EXL3 R10 numerical path derived from ExLlamaV3 v0.0.43 with the MCG
codebook. The first build follows the published GLM-5.3 3.25 bpw method:

- Data-free encoding with identity Hessian.
- Deterministic seeds derived from layer, expert, projection, and TP rank.
- Hadamard-style rotations to spread outliers.
- Trellis search with error feedback.
- Reconstruction after every encode.
- Per-expert relative round-trip error aggregated across all projections and
  TP ranks.

### K3 stage

Encode all routed experts at K3:

```text
76 * 256 * 3 projections * 3 ranks = 175,104 slice encodes
```

For every slice, save:

- `trellis`
- `suh`
- `svh`
- `mcg` marker
- reconstruction statistics
- source tensor identity
- encoder identity and seed
- completion receipt

Assemble and freeze the uniform K3 checkpoint before starting K4 promotion.

Proposed artifact name:

```text
GLM-5.3-EXL3-TR3-3.0bpw-TP3
```

### Deferred K4 promotion stage

Do not execute this stage during the current K3 run. If K3 passes serving but
needs more fidelity, make a new decision about equal versus uneven TP geometry
before continuing. For an unchanged geometry, promotion would proceed as
follows for each MoE layer:

1. Aggregate the K3 reconstruction error for each expert across `gate_proj`,
   `up_proj`, `down_proj`, and all three TP ranks.
2. Rank all 256 experts by relative round-trip error.
3. Select the 64 highest-error experts.
4. Re-encode only those experts at K4.
5. Replace their payloads in a copy-on-write checkpoint assembly.
6. Keep the other 192 K3 expert payloads byte-identical to the frozen K3
   checkpoint.

Additional K4 work:

```text
76 * 64 * 3 projections * 3 ranks = 43,776 slice encodes
```

The resulting average routed-expert bitrate is:

```text
(192 * 3 + 64 * 4) / 256 = 3.25 bpw
```

Proposed artifact name:

```text
GLM-5.3-EXL3-TR3-3.25bpw-TP3
```

Never overwrite the K3 checkpoint during promotion.

## Work distribution

Quantization is task-parallel, not tensor-parallel. Each Spark processes
different layers independently. RoCE is not on the critical path during the
encode.

Use a deterministic layer assignment such as:

```text
mj-spark-1: layers where layer_index mod 3 = 0
mj-spark-2: layers where layer_index mod 3 = 1
mj-spark-3: layers where layer_index mod 3 = 2
```

Balance the final MTP layer explicitly if measured layer times differ. Every
work unit must be independently resumable.

## Execution phases

### Phase 0: source completion

- Completed: the Zima download exited successfully.
- Completed: all 282 shards are present with zero incomplete files.
- Completed: `model.safetensors.index.json` references 282 existing shards with
  zero missing shards.
- Completed: source revision and total weight bytes match the pinned inventory.
- Remaining before encode: generate the full source SHA-256 manifest, audit
  unindexed tensors, and record tokenizer and chat-template identities.
- Do not start bulk encoding from a partial source.

### Phase 1: rotating uneven-layout qualification

- Preserve the H3 container identity and restart path.
- Stop H3 only for the bounded qualification.
- Verify the R10 source bundle hashes.
- Run the synthetic EXL3 smoke test.
- Implement and unit-test the per-layer width, offset, and wide-rank rotation
  schedule, including the MTP layer-78 override.
- Encode one real expert at each of the three rotation geometries.
- Verify packed-index and reconstruction parity.
- Run a synthetic uniform-K3 routed-expert layer through the actual
  `GlmMoeDsaForCausalLM` TP3 loader and execution kernel on all three Sparks.
- Verify the rank-specific 768/640/640 physical shapes, offsets, buffer sizes,
  and NCCL reduction for all three rotations.
- Compare reference EXL3 execution with the fused path before bulk encoding.
- Encode one complete real uneven layer and record its K3 error ranking. Do not
  perform K4 promotion in the current run. The rejected equal-width layer is not
  reusable checkpoint payload.
- Record elapsed time, peak memory, temperature, clock, power, local I/O, and
  output size.
- Restore H3 if the bulk run is not starting immediately.

For this K3-only run, the complete real-layer qualification records K3 error
ranking but does not perform K4 promotion. The corrected equal-width layer
completed 256 experts in 3,768.46 aggregate expert seconds, about 62.8 minutes,
and is preserved as numerical and timing evidence only. The uneven real-layer
result replaces all capacity and schedule projections before bulk work.

### Phase 2: bulk K3 encoding

- Preserve the live GLM-5.3-Flash and H3 container configurations.
- Stop GPU workloads before starting encoders on their nodes.
- Split the remaining layers across all three Sparks.
- Stage only the source shards needed by each active work unit.
- Write local temporary outputs and atomically seal completed layer artifacts.
- Publish sealed layers and receipts to the UNAS.
- Monitor progress, memory, temperatures, clocks, power, and kernel logs.
- Resume from receipts after interruption. Never restart completed matrices.

### Phase 3: K3 checkpoint assembly

- Assemble all K3 layers with the untouched BF16 tensors.
- Generate the model index, per-layer TP3 rank-width and offset metadata,
  attention and vocabulary padding metadata, tier map, source manifest, and
  output manifest.
- Verify that every output tensor is indexed once.
- Verify that every indexed tensor exists.
- Verify byte identity for every carried BF16 tensor.
- Freeze the K3 checkpoint and its manifest.

### Phase 4: deferred K4 promotion and 3.25 bpw assembly

- Do not run this phase automatically.
- Preserve the K3 expert-error ledger and immutable checkpoint.
- Revisit physical geometry and measured K3 memory before authorizing K4.
- If geometry stays unchanged, promote exactly 64 experts per layer and reuse
  the other 192 K3 payloads.
- If geometry changes, re-encode K3 under the new geometry before promotion.

### Phase 5: local checkpoint replication

- Copy the selected candidate checkpoint from the UNAS to local NVMe on every
  Spark using resumable transfers.
- Verify complete file counts and hashes on all three nodes.
- Launch only from local NVMe.
- Keep the UNAS copy as the authoritative archive.

### Phase 6: runtime bring-up

Start with the smallest safe runtime envelope:

```text
TP=3
DCP=1
PP=1
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=1024
KV_CACHE_DTYPE=fp8
CUDA graphs disabled
MTP disabled
prefix caching disabled
```

Use the proven three-node fabric rules:

- `NCCL_IB_SUBNET_AWARE_ROUTING=1`
- `NCCL_NET_PLUGIN=none`
- `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`
- `NCCL_IB_GID_INDEX=3`

Require a real completion before enabling CUDA graphs, MTP, larger batched
tokens, concurrency, or longer context.

### Phase 7: validation and comparison

Evaluate K3 against the sealed BF16 teacher logits and the current
GLM-5.3-Flash service. Do not block the K3 decision on a new official-FP8 or
NVFP4 layer-streaming evaluator.

### Phase 8: controlled routing decision

Do not change Codex Router, ZCode, LibreChat, the Zima adapter, or the public
route until the candidate passes every checkpoint, runtime, quality, and
stability gate.

If accepted, update clients and routes together, refresh model discovery, and
run a real agent task through every path. Keep the previous Flash containers,
checkpoint, image, environment, and route configuration available for rollback.

## Safety controls

DGX Spark uses unified CPU and GPU memory. An uncontrolled GPU allocation can
starve Linux and make the machine appear frozen. Apply these controls:

- Never run the encoder alongside an active large GPU service on the same node.
- Process bounded expert batches rather than entire layers in memory at once.
- Avoid RAM-backed temporary directories.
- Limit CPU worker count and prefetch depth.
- Keep a host-memory reserve and stop the encoder before available memory falls
  below the chosen threshold.
- Watch swap activity and abort on sustained thrashing.
- Use local NVMe for work files and compiler caches.
- Write logs and state under `~/Dev/logs` and `~/Dev/state`, not the model
  directory.
- Keep model directories checkpoint-only.
- Use atomic output renames and checksummed receipts.
- Check NVRM, Xid, kernel OOM, filesystem I/O, and network errors continuously.
- Treat the protected GLM-5.3-Flash cold-start window separately from encoder
  and candidate-runtime windows. Recovered `NV_ERR_NO_MEMORY` allocation
  retries are permitted during Flash model loading, profiling, and CUDA-graph
  capture only when both ranks remain alive, neither container is OOM-killed,
  startup completes within its bounded timeout, `/health` passes, a real
  generation finishes normally, and the kernel log is clean after readiness.
- Continue to fail immediately on any Xid, worker death, `OOMKilled` state,
  persistent allocation loop, startup timeout, failed canary, or NVRM event
  after Flash readiness. No NVRM event is permitted during K3 encoding or the
  candidate TP3 execution window.
- Do not overclock the Sparks.
- Confirm normal clocks, power, and temperature before accepting timing data.
- Use the supplied power adapters and unobstructed cooling.
- Do not use a successful weight load or KV allocation as proof of serving
  success.
- Do not make serving depend on the UNAS remaining online.

No `sudo` workaround is allowed. If a required system change needs `sudo`, stop
and provide one exact command for the user to run.

## Completion and quality gates

### Source integrity

- 282 of 282 BF16 shards present.
- Source revision matches the pinned revision.
- Index references resolve with zero missing shards or tensors.
- Tokenizer and chat template identities recorded.

### K3 structural integrity

- 76 quantized MoE layers.
- 256 K3 experts per layer.
- Three TP payloads per projection.
- Expected projection and payload counts match exactly.
- No NaN or infinite values.
- Pack and unpack equality passes.
- Runtime reconstruction matches the encoder reference.
- Per-layer rank widths and offsets cover exactly 2,048 semantic channels with
  zero gaps, overlap, or expert-channel padding.
- Every carried BF16 tensor is byte-identical to the source.

### Mixed K3/K4 structural integrity

- Exactly 192 K3 and 64 K4 experts per layer.
- Promotion map derived from that layer's TP3 K3 error results.
- Promoted payloads pass reconstruction checks.
- Non-promoted payloads match the frozen K3 checkpoint byte for byte.
- The K3 checkpoint remains independently loadable.

### Logit fidelity

Use the same sealed BF16 teacher-logit windows for both candidates. Measure
full-vocabulary teacher-forced `KL(teacher || student)` at every position.

#### Full GLM-5.3 reference checkpoints

The comparison must distinguish weight precision from KV-cache precision.
`FP8` or `NVFP4` can describe model weights, the KV cache, or both. Record both
columns for every measurement.

| Reference | Pinned revision | Weight treatment | Safetensor size | Same-panel KLD vs BF16 |
|---|---|---|---:|---:|
| BF16 teacher, `zai-org/GLM-5.3-BF16` | `304b8051cfb2b260b61ce0cbe330e02a98e73639` | Source BF16 | about 1.403 TiB | `0.0` by definition |
| Official FP8, `zai-org/GLM-5.3` | `187fb9fff6319062325ff825627ef6db084d9bc6` | Block FP8, dynamic activations, 128 by 128 weight blocks | 703.74 GiB | Not publicly measured on this panel |
| IncoAI NVFP4, `incoai/GLM-5.3-NVFP4` | `54e52520606f96b3d9fc84088ad22882a61648ac` | Routed-expert NVFP4 weights and static NVFP4 activations, FP8 KV declared | 432.90 GiB | Not publicly measured on this panel |
| RadixArk NVFP4, `RadixArk/GLM-5.3-NVFP4` | `363e8f086905afd83db356a620f9aa401c23800a` | Routed-expert NVFP4 W4A4, non-expert and MTP tensors BF16, FP8 KV declared | 432.90 GiB | Not publicly measured on this panel |
| Ressl NVFP4, `ressl/GLM-5.3-NVFP4` | `3281937dd50fda90266fd66527d664d1658aa1a0` | Routed-expert weight-only NVFP4, non-expert and MTP tensors BF16 | 432.90 GiB | No benchmark or same-panel KLD published |

Source pages:

- [Official GLM-5.3 FP8](https://huggingface.co/zai-org/GLM-5.3)
- [IncoAI GLM-5.3 NVFP4](https://huggingface.co/incoai/GLM-5.3-NVFP4)
- [RadixArk GLM-5.3 NVFP4](https://huggingface.co/RadixArk/GLM-5.3-NVFP4)
- [Ressl GLM-5.3 NVFP4](https://huggingface.co/ressl/GLM-5.3-NVFP4)

Do not substitute one NVFP4 result for another. IncoAI and RadixArk use W4A4
expert execution, while the Ressl export describes a weight-only path. Their
runtime behavior and KLD may differ even though their checkpoint sizes are
almost identical.

The absence of a published number is a data gap, not evidence that FP8 or
NVFP4 matches BF16. Keep these cells marked as unmeasured until a result exists
on the exact teacher tokens and scoring path used for the EXL3 candidates.

#### Published full-model EXL3 references

These values use the sealed full GLM-5.3 BF16 teacher-logit panel with four
2,047-position confirmation windows and the complete 154,880-token vocabulary:

| Checkpoint | KV mode | Author result | Independent result | Comparability note |
|---|---|---:|---:|---|
| Published uniform K3 | BF16 replay | `0.03754` | Not reported | Layers 3 through 77 use K3 and the MTP layer uses K5, so this is contextual rather than an exact match for our all-K3 target |
| Published 3.25 bpw | FP8 | `0.026103` | `0.026776` | Current primary full-model EXL3 reference |
| Published 3.42 bpw | FP8 | `0.024105` | `0.023966` | Higher-bitrate quality reference |
| Published 3.25 bpw | NVFP4 KV | `0.035741` | `0.036661` | Same EXL3 weights with a lower-precision KV path |
| Published 3.42 bpw | NVFP4 KV | `0.037757` | `0.037060` | Shows that KV error can outweigh a small weight-bitrate improvement |

Sources:

- [Full GLM-5.3 uniform K3 measurement](https://huggingface.co/brandonmusic/GLM-5.3-EXL3-TR3-3bpw)
- [Full GLM-5.3 3.25 bpw measurement and reproduction kit](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw)
- [Full GLM-5.3 3.42 bpw and KV-mode comparison](https://huggingface.co/davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw)

The published uniform-K3 result used clean BF16 replay, while the mixed-bit
results above used a serving path with FP8 or NVFP4 KV. Compare the exact
conditions, not only the headline mean.

#### Required reference measurements

Produce or import the following rows on the same confirmation tokens before
making a final fidelity claim:

| Row | Weights | KV | Purpose |
|---|---|---|---|
| A | BF16 | BF16 | Teacher self-check and scorer floor |
| B | Official FP8 | FP8 | Official weight reference |
| C | Selected NVFP4 checkpoint | FP8 | NVFP4 weight reference without changing KV format |
| D | TP3 K3 | FP8 | Our capacity candidate |
| E | TP3 3.25 bpw | FP8 | Our quality candidate |
| F | TP3 3.25 bpw | NVFP4, optional | KV sensitivity only |

Row B, the full-model official FP8 result, is deferred for the current K3 run.
A missing Row B prevents claims that K3 is close to official FP8, but it does
not prevent a preliminary K3 serving decision based on BF16-teacher KLD,
matched task quality, memory, and stability.

Run every row in eager mode with speculation and sampling disabled. Use the
same token arrays, full-vocabulary FP32 log-softmax, reduction order, and final
normalization and `lm_head` path. Save per-token values, not only the mean.

The full official FP8 and public NVFP4 checkpoints do not fit in the current
three-Spark memory envelope for straightforward serving. The official FP8
safetensors alone occupy 703.74 GiB, while the three Sparks provide about 363
GiB of usable unified memory before runtime overhead.

The following layer-streaming design is deferred follow-up work and is not part
of the current K3 schedule:

1. Store the complete reference checkpoint on the UNAS and pin its revision and
   index.
2. Create byte-preserving TP3 reference slices from the source weights. Apply
   the candidate's rotating 768/640/640 expert partition plus only the physical
   attention-head and vocabulary padding required by TP3. Do not requantize
   weights or alter scales.
3. Place each rank's bounded workspace and layer cache on that Spark's local
   NVMe. Record a manifest proving that the three rank slices reconstruct the
   source tensor without gaps or overlap and that remaining head/vocabulary
   padding is zero.
4. Batch the four 2,048-token confirmation windows so each model layer is read
   once per reference pass.
5. On every layer, have all three Sparks load their local rank slice, execute
   the layer concurrently, perform the required RoCE collective, retain the
   resulting hidden and mHC states, release the layer weights, and continue
   through layer 78.
6. Run final normalization and stream `lm_head` over bounded vocabulary chunks.
   Distribute the vocabulary work across all three Sparks when numerical
   equivalence with a single-rank reduction has passed.
7. Accumulate full-vocabulary FP32 KLD against the sealed BF16 teacher logits
   without retaining the complete student-logit tensor in memory.
8. Save per-token KLD, top-1 agreement, top-k overlap, margin-conditioned
   disagreement, timings, per-rank memory peaks, source hashes, slice
   reconstruction proofs, collective settings, and evaluator identity.

The official FP8 and NVFP4 reference formats are not pre-encoded as four fixed
EXL3 trellis ranks. Their packed tensors and scales can therefore be partitioned
for TP3 at load or staging time, subject to block alignment and exact
reconstruction checks. Use the same semantic 64-to-66 attention-head padding
and rotating 768/640/640 expert partition as the candidate runtime.

This is distributed layer streaming, not data-parallel window splitting:

```text
For each layer:
  Spark 1 loads rank-0 slice
  Spark 2 loads rank-1 slice
  Spark 3 loads rank-2 slice
  all three compute concurrently
  RoCE combines the TP partials
  all three release the layer before the next one loads
```

This design reads and stores roughly one third of the active reference weights
per node. It also keeps the reference and candidate TP3 geometry aligned.

The layer-streaming evaluator performs one complete model forward. It does not
calculate independent layer KLD values and add them. MoE routing, attention,
residual connections, normalization, and later layers make layer-local KLD
non-additive.

Qualify the evaluator before trusting its numbers:

- Compare streamed and resident execution on a smaller compatible model.
- Compare one GLM layer's streamed output with the same layer held resident.
- Verify exact routing ids and weights.
- Verify final norm and `lm_head` chunking against an unchunked small case.
- Verify that vocabulary chunking changes KLD by no more than the declared
  numerical tolerance.
- Run the BF16 teacher self-check and require zero KLD apart from the measured
  scorer floor.
- Use the official FP8 dynamic-activation and KV behavior for the serving-path
  row. Also record a clean weight-only row when the runtime permits it.

The first four-window official FP8 distributed-streaming pass is expected to
take about two to five hours after evaluator qualification. Replace that
estimate with the first measured full-layer TP3 pass. The NVFP4 reference
reuses the same evaluator, TP3 schedule, and teacher tokens.

A matching public capture may be used as an independent cross-check only after
its token hashes, checkpoint revision, runtime, reduction path, and scorer
identity verify against this plan. Cloud execution is outside the current
plan.

If Row C remains unavailable, mark only the NVFP4-relative gate as unscored.
Do not use GLM-5.3-Flash KLD values as substitutes for either full-model row.

#### GLM-5.3-Flash context only

The following values are useful for interpreting scale but must not become
full GLM-5.3 acceptance gates. GLM-5.3-Flash has a different 45-layer hybrid
architecture, 288 routed experts, and a different BF16 teacher.

| Flash reference | Mean KLD vs Flash BF16 | Measurement note |
|---|---:|---|
| Official FP8 | `0.020615` | Cross-stack sealed 25-window result |
| Official FP8 | `0.024629` | Same-stack reference result |
| Official FP8 | `0.028104` | Larger 5,120-context fidelity suite |
| NVFP4 | `0.060535` | Same-stack reference result |

The spread among the official FP8 rows shows why the scoring panel, runtime,
and reduction path must accompany every KLD number.

Context source:

- [GLM-5.3-Flash FP8, EXL3, and NVFP4 KLD table](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks#quality-kld)
- [GLM-5.3-Flash large-panel official FP8 result](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1)

Primary provisional gate:

```text
Mean KLD with FP8 KV <= 0.030
Published TP4 3.25 bpw reference = 0.026103
Independent TP4 reproduction = 0.026776
```

Only after full-model official FP8 and NVFP4 same-panel rows are available,
apply these relative gates:

```text
Near-official-FP8 gate:
  EXL3_3.25_KLD <= Official_FP8_KLD + 0.005 nats
  EXL3_3.25_KLD / Official_FP8_KLD <= 1.25

NVFP4-superiority gate:
  EXL3_3.25_KLD < NVFP4_KLD
  EXL3_3.25_KLD / NVFP4_KLD <= 0.80
  paired 95% confidence interval for EXL3 minus NVFP4 remains below 0
```

The first pair defines "very close to official FP8" as no more than 0.005
nats of absolute drift and no more than 25 percent relative drift on the same
tokens. The second requires at least a 20 percent mean-KLD advantage over
NVFP4, supported by a paired per-context comparison rather than a point
estimate alone.

If the official FP8 full-model reference remains unavailable, report the
near-FP8 gate as unscored and do not make a final near-FP8 claim or select a
production winner on that basis. If the NVFP4 reference remains unavailable,
report the NVFP4-superiority gate as unscored. Do not infer either pass from
the GLM-5.3-Flash numbers. The absolute `0.030` gate and the task-level
comparison still apply as preliminary development gates.

Also report:

- KLD per evaluation window.
- Top-1 token agreement.
- Top-k overlap.
- Margin-conditioned disagreement.
- First divergence position.
- Legal, dialogue, prose, reasoning, code, and Korean subsets where available.

Do not translate KLD directly into a percentage of information retained.

### Kernel and distributed correctness

- Reference EXL3 execution matches the fused kernel within the declared
  numerical tolerance.
- Eager and graphed execution agree.
- Every rank reports the intended physical and semantic geometry.
- NCCL initializes over the two-leg RoCE ring.
- No rank omits or duplicates a weight slice.
- MTP-off target behavior passes before MTP3 is enabled.
- MTP3 changes throughput but not target correctness.

### Task quality

Run the same prompt set and sampling configuration on K3, 3.25 bpw, and the
current Flash model:

- Exact-output instructions.
- JSON schema and structured output.
- Forced and automatic tool calls.
- Reasoning and visible-answer separation.
- Code generation, code repair, and code review.
- Math and science reasoning.
- Korean and mixed Unicode, with zero replacement characters.
- Long-context retrieval at 55K, then 128K, then larger validated sizes.
- The Codex Router agent check.
- A real repository task with verifiable output.

The full model must show a meaningful coding or long-horizon reasoning gain to
justify replacing Flash.

### Performance and stability

Record:

- Cold and warm prefill throughput.
- Single-stream decode throughput.
- Aggregate C2 and C4 throughput where memory permits.
- Time to first token.
- KV pool size and safe request context.
- Resident memory and host available memory on every Spark.
- MTP acceptance by draft position.
- Queue, preemption, and request-failure counts.
- GPU clocks, temperature, and power.

Run a sustained mixed workload for at least one hour. Accept only if there are
no post-ready NVRM, Xid, kernel OOM, engine-fatal, filesystem, or queue-stall
events.

## Comparison and selection policy

Consider building a 3.25 bpw model later when all of these are true:

- Its KLD is lower than K3 by a meaningful amount.
- It improves difficult-window and margin-conditioned results.
- It improves or preserves actual task success.
- It fits with a useful context limit and stable memory reserve.
- Its speed cost is acceptable.

Choose K3 when:

- K4 promotion gives little task-level benefit.
- K3 provides materially better context or memory headroom.
- K3 is more stable on the three-Spark runtime.

Restore the current Flash service when neither full-model checkpoint passes
the quality, capacity, or stability gates.

## Time estimate

The measured Spark 3 encoder smoke and the expected TP3 workload imply:

| Execution mode | Estimated duration |
|---|---:|
| One Spark only | 3.5 to 4.5 days |
| All three Sparks | 1.5 to 2.5 days |
| Hybrid schedule preserving Flash part-time | 2.5 to 4 days |

The revised estimate uses the measured 3,768.46 expert-seconds for the rejected
equal-width layer and adds staging, assembly, hashes, initial KLD, and runtime
qualification. It excludes K4 promotion and the deferred streamed FP8 and
NVFP4 evaluator. Replace it after the first complete uneven real-layer
measurement.

## Rollback and preserved state

Before bulk encoding or candidate serving, record and preserve:

- Current GLM-5.3-Flash image digest and container definitions on Sparks 1 and
  2.
- Current Flash environment and model checkpoint revision.
- H3 container definition, mounts, image, and restart policy on Spark 3.
- Zima adapter and tunnel configuration.
- Codex Router and ZCode model configuration.
- LibreChat and public-route model identity.

A failed encode resumes from receipts. A failed candidate boot stops the TP3
candidate and restores the previous Flash and H3 services. Client routing must
remain on Flash until the TP3 candidate passes the final end-to-end gate.

On 2026-09-04, the operator explicitly authorized resuming after the corrected
TP3 runtime gate passed and the preserved Flash rollback recovered from
startup-only `NV_ERR_NO_MEMORY` retries. The resumed audit must classify events
by lifecycle window and enforce the conditions above instead of rejecting every
startup-only retry unconditionally.

## Final deliverables

- Immutable TP3 K3 checkpoint.
- Complete per-expert K3 error ledger suitable for a later K4 decision.
- Source and output SHA-256 manifests.
- Layer and expert completion receipts.
- Per-layer K3 expert ranking suitable for a later K4 promotion decision.
- Encoder source and environment lock.
- KLD results for the K3 checkpoint.
- Kernel and TP3 correctness report.
- Matched task-quality comparison against GLM-5.3-Flash.
- Performance, context, memory, and stability report.
- Reproducible start, stop, status, smoke, benchmark, and rollback commands.
- Final recommendation with the measured reasons for keeping K3, pursuing a
  later 3.25 bpw build, or restoring the existing Flash model.
