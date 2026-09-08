# Archive and re-encoding experiment, September 8

## Protected checkpoints

The authoritative NAS directory is `/mnt/unas-models/ZAI`. Preserve these independently loadable checkpoints without changing their contents:

- `GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z`
- `GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z`

Preserve the BF16 source, original sealed K3 layers/error ledgers, K2 subsets, tokenizer, manifests, runtime source and latest quality evidence. Notes and new audit receipts live alongside the checkpoints in `GLM-5.3-EXL3-archive-notes-20260908`, not inside their sealed directories.

The previous scale campaign tried ten changes. K2 scales multiplied by 1.05 gave two-run mean KLD 0.04728214 versus baseline 0.04849901, but the first comparison regressed and mean top-1 fell from 92.7455% to 92.6478%. No changed setting was accepted. The original scales remain the baseline. Baseline teacher-agreement outcomes changed at 192 of 8,188 positions across repeats, so small full-model KLD gains are not yet reliable.

## Checksum-ledger hardlink incident

The new preservation audit found that the K275 assembler hardlinked the K3 `SHA256SUMS` file, then rewrote it in place. NAS K3 and K275 shared inode 71624. K3's sealed marker expected `1ea4a0c11d6046060cb713f140a3605b8c50cb7260cea038114ceda6de4b1898`, but its ledger contained K275's `b3311519773b35a8ede1fe10975df5e97096e1870fc994c51626a1da03ea2457`.

The original K3 manifest was intact. Reconstructing the ledger from it reproduced the original expected hash exactly. Restore those exact bytes using atomic replacement, which breaks the shared link without modifying K275. Save the bad ledger and repair receipt. The same issue affected Spark 2 and Spark 3 replicas; Spark 1's ledger was already correct. All three local ledgers were checked, and affected copies were repaired. No weight tensor was changed by this repair. Full NAS payload hashing follows separately.

The assembler now checks the base manifest/ledger before assembly and writes its new ledger atomically. A CPU regression test proves that replacing a hardlinked candidate ledger leaves the base bytes and inode intact.

## Cleanup scope and failure memo

Only delete the raw `.f32` parts of the following two invalid captures, after preserving their existing receipts, API responses, error records and a file/hash inventory. They cannot be used for KLD because raw rows did not align with prompt positions:

- `GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-kld-evidence-prefill256-20260907T224500Z`: capture `confirmation-0000-1788824319`. The scheduler emitted singleton sampling rows interleaved with 256-token prompt chunks; expected counts were wrong. Row/API log-probability differences reached 23.46 nats.
- `GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-kld-evidence-prefill256-retry2-20260907T224500Z`: capture `confirmation-0000-1788824597`. Normalized prompt row count still differed; row/API differences reached 23.89 nats.

Retain the third retry, DCP3 captures and scale-v2 raw evidence. Retain scale-v1 calibration inputs because this pilot uses them. Scale-v1's output-multiplier trial stalled after 1,285 captured rows; no changed-scale KLD was valid. Its cause was not isolated because both implementation and chunk size changed before v2 succeeded.

Stopped `k275-quality-v1` containers may be removed only after their inspect records and complete logs are saved to NAS. Keep current/baseline containers and all model replicas. The old equal-width layer remains immutable rejected capacity evidence.

Deletion receipts distinguish irrecoverable invalid raw parts from preserved diagnostics. Do not call the removed invalid data an archived full capture.

## Bounded layer-33 pilot

Run on Spark 2 under a Zima-controlled tmux job, after stopping all TP3 serving ranks. Preserve a runnable baseline; do not restore Flash/H3 or alter public routes.

Use the staged BF16 layer 33 and the already captured 4,096 input rows. Fit on the first 2,048 code-window rows and evaluate on the separate 2,048 reasoning-window rows. Neither window is a private final test set. Choose pilot experts using routing coverage only, not measured reconstruction loss.

Compare 8 existing K2 and 8 existing K3 experts with adequate coverage. For each expert:

1. Measure frozen K3 output error against BF16 weights using FP32 matrix arithmetic.
2. Encode/reconstruct identity-H K2 using the original deterministic rotations and normalization. For existing K2 experts, require exact payload equality with the frozen checkpoint.
3. Re-encode K2 using training activation covariance and the same rotations/scales, fixed regularization and unchanged TP3 geometry. Gate/up covariance uses layer inputs; down covariance uses that candidate's intermediate activations. Do not tune on evaluation rows.
4. Check packed size, pack/unpack equality, finite reconstruction and repeated local output determinism.

Choose 8 of the 16 pilot experts for K2 using training-only incremental K2-versus-K3 weighted output error. Keep every non-pilot assignment unchanged, yielding exactly 64 K2 and 192 K3 experts in the layer. Evaluate the frozen selection on the reasoning rows. Compare selection-only and activation-aware re-encoding separately.

This is a local pilot, not a full-model KLD result or an automatic promotion. Preserve outputs separately. Expand coverage only after numerical, byte-budget, held-out output and hardware gates pass. Maintain at least 12 GiB host reserve, a dedicated watchdog and strict zero GPU/kernel/OOM/I/O faults. No large encoder may overlap serving.

## Measured outcome

The full NAS checksum audit completed at 2026-09-08 04:51:52 UTC. All 39,266 ledger entries passed, with shared physical files read once and their expected hashes checked for every alias. Both assembled checkpoints are preserved. The invalid-capture cleanup removed 32 raw parts totaling 2,538,792,960 bytes and three stopped quality-v1 containers; diagnostics remain, but those invalid raw parts were intentionally discarded.

The corrected pilot encoded all 16 experts in 542.76 seconds, with 2.574 GiB peak PyTorch CUDA allocation. Identity-H replay exactly reproduced all eight frozen K2 experts. The first software attempt stopped on a CPU/GPU device mismatch in a reconstruction assertion; its logs and design remain preserved.

The frozen selection-only map promoted experts 154 and 174 to K3 and demoted 79 and 192 to K2. It retained exactly 64 K2 experts. A stronger check then summed all 256 routed experts using the same captured routing weights. Evaluation error rose 1.024% for selection only, 1.640% for activation-aware re-encoding with the original selection, and 57.017% for re-encoding plus selection. All three variants reduced training error. The shared scalar scores matched exactly across two full-layer passes.

Reject all variants. Do not expand this recipe into bulk encoding or publish a full-model derivative. These are local routed-layer output errors, not KLD or top-1 measurements. The inputs came from the original K275 model, not BF16 hidden-state replay. The calibration was small and split by domain, so the observed pattern supports investigating overfitting/domain sensitivity, not concluding that activation-aware quantization never helps.

The original K275 serving profile was restarted as `k275-post-reencode-v1`, retaining 32K context, TP3/DCP3, resident UVA, FP8 KV, 128-token prefill chunks, eager execution and MTP off. Flash/H3 and public routes remain unchanged. Restore verification receipts are stored beside this memo.
