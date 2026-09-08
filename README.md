# GLM-5.3 on three DGX Sparks

The qualified original **K275 model generates 13.06 decode tok/s**, up from 5.12 before tuning: **2.55× faster**. Six measured llama-benchy runs, pp2048 / requested tg256, concurrency 1. Decode excludes prefill.

All routed experts remain resident in RAM across three Sparks. The selected profile is TP3/DCP3, MTP3, eager execution, 128-token prefill chunks, FP8 KV at 1 GiB per rank, and a 32,768-token model limit. A real 30,039-token prompt passed retrieval with normal stopping. No quantized weights or routing were changed for the speed improvement.

| Start here | Purpose |
|---|---|
| [Progress](docs/PROGRESS.md) | What changed, what failed, and what was retained |
| [Speed results](SPEED_RESULTS.md) | Complete measurements, exclusions, and qualification |
| [Reproduction runbook](docs/REPRODUCE.md) | Safe existing-homelab setup, benchmark, and rollback |
| [Dependencies](docs/DEPENDENCIES.md) | Exact image, checkpoints, encoder, and external evidence |
| [Commit map](docs/COMMITS.md) | Separate retrospective commits for each improvement |
| [Validation](docs/VALIDATION.md) | Packaging checks and isolated CPU test results |
| [Agent instructions](AGENTS.md) | Current operational boundaries |
| [Machine-readable results](results/speed-results.json) | Per-run measurements and selection eligibility |

The last qualified attempt is `speed-qualified-final`; its endpoint is `http://192.168.0.238:8893`. The legacy API alias `GLM-5.3-K3-TP3-CANDIDATE` serves **K275**, not K3. Recheck live state before acting. Public routes were not promoted and Flash/H3 remain stopped.

This is a reproduction bundle for the existing provisioned homelab, not a clean-OS installer. Images, checkpoints, rank packs, credentials, and raw benchmark/capture output are external. See the dependency inventory before launching anything.

Git history was created retrospectively on September 8, 2026. It is not the original experiment timeline. Earlier commits are historical snapshots, not independently qualified deployments. No public push is required.
