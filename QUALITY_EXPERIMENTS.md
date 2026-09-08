# K275 KLD experiments, September 8

These experiments run under Zima tmux and use all three Sparks for TP3/DCP3
inference. The starting model is the sealed 2.75 routed-expert bpw checkpoint
assembled on September 7. All experts, routing and sparse attention remain.

First measure scalar recalibration of routed K2/K3 outputs. These are changes
to the dequantized expert outputs and could be folded into the existing
down-projection FP16 scales; they are not output-logit temperature changes.
The packed trellis tensors and K2 expert selection stay fixed for this test.

The first output-multiplier trial at a 256-token prefill chunk stalled after
1,285 captured rows. Its kernel and memory audits remained clean. Preserve it
as a liveness failure, with no KLD result. The second implementation writes
the actual FP16 down.svh vectors in resident buffers and snapshots their
original values for exact restoration. It uses 128-token chunks and a new
attempt and evidence directory.

Development uses previously unused `selection-0000` and `selection-0001` from
the pinned public BF16 full-logit dataset revision
`427368f12a4bdc21668bc4171ce0dc54f8990200`. Downloaded token and logit files must
match the aggregate manifest hashes. The four `confirmation-0000..0003`
windows are reserved from parameter selection in this sweep. They are public
confirmation data previously used for baseline reporting, not a private final
test set.

Predeclared development cases:

| Case | K2 output gain | K3 output gain |
|---|---:|---:|
| Identity baseline | 1.000 | 1.000 |
| K2 -5% | 0.950 | 1.000 |
| K2 -2.5% | 0.975 | 1.000 |
| K2 +2.5% | 1.025 | 1.000 |
| K2 +5% | 1.050 | 1.000 |
| K3 -1% | 1.000 | 0.990 |
| K3 +1% | 1.000 | 1.010 |

Additional predeclared probes compare SwiGLU limits 20, 40 and unbounded
against the existing limit 10, and the separately fitted layer-33 expert
scales. The unquantized model code uses ordinary SiLU multiplication and
does not supply a SwiGLU clamp. The captured layer-33 sample found no clamp
effect; the whole-model probes establish whether other layers differ.

Choose the lowest mean development KLD, then compare it with a fresh identity
control on all four confirmation windows. Report all regressions. A useful
candidate must improve confirmation KLD without losing more than 0.1
percentage point top-1 agreement. The endpoint returns to identity gains when
the sweep finishes until any winning scales are saved and revalidated.

The second experiment can use captured inputs at layers 4, 5 and 33 for
activation-based scale refitting or a small Hessian encode. Those layers have
local BF16 staging available. Calibration text must differ from the scoring
windows; fit and validation rows must be split before refitting. A large
encoder must run only after all serving ranks stop and memory recovers.

Use the existing 32K profile, 128-token prefill chunks, 1.69 GiB FP8 KV per
rank, NCCL1, resident UVA and the 12 GiB host reserve. Changes use an opt-in
quality control file and preserve the existing stopped candidate for rollback.

Controller results are under
`/home/mj-kang/Dev/state/glm53-full-exl3-tp3/quality/20260908-scale-v2` on Zima.
Raw captures are under
`/mnt/unas-models/ZAI/GLM-5.3-EXL3-K275-quality-experiments-20260908-scale-v2`.
