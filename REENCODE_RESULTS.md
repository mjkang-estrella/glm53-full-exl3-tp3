# K3/K275 preservation and re-encoding results

The bounded layer-33 pilot found no accepted improvement. Original checkpoint weights were not replaced.

## NAS preservation

Full NAS checksum audit passed all 39,266 ledger entries.
Both assembled checkpoints remain under `/mnt/unas-models/ZAI`; exact directory names and recovery details are in `ARCHIVE_AND_REENCODE.md`.
The audit caught a shared-checksum-ledger bug in the earlier K275 assembler. The original K3 ledger was restored to its sealed SHA-256, breaking that hardlink. A regression-tested atomic writer prevents recurrence. No tensor weights were edited by the repair.

Removed 32 invalid raw parts, 2.364 GiB, and 3 stopped failed-test containers.
The invalid raw parts are not recoverable. Their hashes, ordering failures and API receipts remain. Container configuration and logs remain on NAS. No checkpoint was deleted.

## Experimental design

Sixteen layer-33 experts were chosen by routing coverage, eight from each existing tier. All original experts/routing and the 2.75-bpw budget were retained.
Fit used 2,048 code-window input rows; evaluation used 2,048 separate reasoning-window rows. Inputs came from the original K275 model. This was not BF16 hidden-state replay or a private final test.
K2 re-encoding used real activation covariance, fixed sigma 0.025 and the original rotations/scales. K2 assignment used only training incremental K2-versus-K3 error. Evaluation rows did not select the assignments or regularization.
All eight existing K2 experts replayed byte-exactly with identity H. All packed-size, pack/unpack, finite reconstruction and local repeat checks passed.
Encoding took 9.05 minutes, with 2.574 GiB peak PyTorch CUDA allocation.
The first attempt stopped on a CPU/GPU device mismatch in the reconstruction assertion. Its diagnostics were preserved; the corrected v2 attempt completed.

## Summed routed-expert output check

This check evaluated all 256 experts with the captured routes, retaining cross-expert error terms. It is not full-model KLD, task accuracy or native fused-kernel validation.

| Variant | Training relative MSE | Evaluation relative MSE | Evaluation error change |
|---|---:|---:|---:|
| Original K275 | 0.03721741 | 0.03827372 | baseline |
| Selection only | 0.03615288 | 0.03866583 | +1.024% |
| Re-encode only, original selection | 0.03487163 | 0.03890131 | +1.640% |
| Re-encode plus selection | 0.03269505 | 0.06009637 | +57.017% |

All changed cases reduced training error but increased evaluation error. This pattern is consistent with overfitting or domain sensitivity in this small calibration set; it does not prove that activation-aware encoding generally fails.
The shared cases produced exactly matching scalar scores in two full-layer passes. Local packed decoding and output checks were deterministic. The separate full-model repeatability issue remains unresolved.

## Decision

Reject these candidates. No full-model derivative or bulk re-encode was launched, and no new full-model KLD/top-1 result is claimed.
A follow-up would need broader, balanced calibration, stronger covariance regularization assessed on training-only splits, and a fresh evaluation set. The current results do not justify scaling this recipe to the whole model.

NAS evidence: `/mnt/unas-models/ZAI/GLM-5.3-EXL3-archive-notes-20260908`

## Restored serving verification

Attempt `k275-post-reencode-v1` passed 3/3 real generation checks.
Original K275 weights remain loaded at 32K with TP3/DCP3. No quality-control file was present on any rank.
Two unchanged-weight captures used only public window confirmation-0000, 2,047 positions. These are not the four-window KLD headline.

| Repeat | KLD | Top-1 |
|---|---:|---:|
| 1 | 0.03396670 | 91.3532% |
| 2 | 0.03356422 | 91.8906% |

Observed KLD spread: 0.00040248 nats. The cause remains unresolved; unchanged weights do not make this serving path numerically repeatable.
