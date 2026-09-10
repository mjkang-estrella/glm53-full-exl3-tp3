# GLM-5.3 on three DGX Sparks

The qualified original **K275 model measured 13.06 decode tok/s**, up from 5.12 before tuning: **2.55× faster**. Six measured llama-benchy runs, pp2048 / requested tg256, concurrency 1. Decode excludes prefill. These are recorded benchmark results, not live service status.

In the qualified run, all routed experts stayed resident in RAM across three Sparks. The profile used TP3/DCP3, MTP3, eager execution, 128-token prefill chunks, FP8 KV at 1 GiB per rank, and a 32,768-token model limit. A real 30,039-token prompt passed retrieval with normal stopping. No quantized weights or routing were changed for the speed improvement.

## Published checkpoints

**Both uploads are complete and verified.** Each public repository contains an `UPLOAD_COMPLETE.json` marker.

These are EXL3/TR3 quantized versions of GLM-5.3 BF16, not additional fine-tunes. Both Hugging Face cards explicitly set `base_model_relation: quantized`.

| Model | Download | Payload size | Payload files |
|---|---|---:|---:|
| Original K3, 3.0 routed-expert bpw | [Hugging Face](https://huggingface.co/mj-kang/GLM-5.3-EXL3-3.0bpw-TP3) | 315.87 GB | 19,559 |
| Original K2.75, mixed K2/K3 | [Hugging Face](https://huggingface.co/mj-kang/GLM-5.3-EXL3-2.75bpw-TP3) | 292.92 GB | 19,712 |

Sizes use decimal GB and include sealed metadata. Bit rates describe routed experts, not the whole-checkpoint average. Download a pinned completed revision and restore the original layout before loading. These are custom-runtime checkpoints, not drop-in Transformers or llama.cpp models. See [download and restore instructions](docs/DOWNLOADS.md) and [publication receipts](results/model-publication.json).

## Project guide

| Start here | Purpose |
|---|---|
| [Progress](docs/PROGRESS.md) | What changed, what failed, and what was retained |
| [Speed results](SPEED_RESULTS.md) | Complete measurements, exclusions, and qualification |
| [Reproduction runbook](docs/REPRODUCE.md) | Safe existing-homelab setup, benchmark, and rollback |
| [Dependencies](docs/DEPENDENCIES.md) | Exact image, checkpoints, encoder, and external evidence |
| [Download the models](docs/DOWNLOADS.md) | Verified public revisions and lossless restoration |
| [Commit map](docs/COMMITS.md) | Separate retrospective commits for each improvement |
| [Validation](docs/VALIDATION.md) | Packaging checks and isolated CPU test results |
| [Public source scope](docs/PUBLICATION.md) | Publication boundaries and history audit |
| [Hugging Face upload and restore](integrations/huggingface/README.md) | Zima-hosted resumable publication of both sealed checkpoints |
| [Agent instructions](AGENTS.md) | Current operational boundaries |
| [Machine-readable results](results/speed-results.json) | Per-run measurements and selection eligibility |

The recorded K275 qualification used attempt `speed-qualified-final` at `http://192.168.0.238:8893`. Its legacy API alias `GLM-5.3-K3-TP3-CANDIDATE` identified **K275**, not K3. The [LibreChat integration](integrations/librechat/README.md) records that cutover; subsequent [Flash replacement work](integrations/flash_mia/README.md) is separate. Model publication does not mean a checkpoint is currently serving. Inspect live state before operational work.

This is a reproduction bundle for the existing provisioned homelab, not a clean-OS installer. Images, checkpoints, rank packs, credentials, and raw benchmark/capture output are external. See the dependency inventory before launching anything.

Git history was created retrospectively on September 8, 2026. It is not the original experiment timeline. Earlier commits are historical snapshots, not independently qualified deployments. Public publication and the subsequent web-route cutover were separately authorized and recorded after the original ten-commit wrap-up.
