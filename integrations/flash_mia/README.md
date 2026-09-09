# GLM-5.3 Flash return, September 9

Upstream source: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks

Pinned commit: `dc6936cea8fd7b2e7ee5b7a48a5aa193857ca489`.

This directory provides the local deployment changes. Obtain the complete corresponding source by cloning that upstream revision and applying `launcher.patch` with `git apply`. Keep upstream LICENSE and LICENSE.MIT. The upstream runtime source is AGPL-3.0; this patch and derived deployment scripts are provided under the same license. Model and DFlash2 licenses remain separate and unchanged. No model weights or credentials are included.

The additive E3 image is built with upstream `Dockerfile.e3-layer` on `glm53-exl3:e2-c190db1`, exact base ID `sha256:b7ff496564001ee02655ce796370bf823a233a963f04c3d334a549bb5f0628e5`. The E3 result is `glm53-flash-e3:dc6936c-20260909`, ID `sha256:e8a146bc54d4c0cf8a9f04198e72fd65c5ab5f908f57d9ca80c540619340f003`. The upstream additive kernel is built without fast-math; the original extension and quantized weights are retained.

Profile: TP2 on Spark 1/2, DFlash2 k=7/draft TP2, FP8 target KV, E3 grouped prefill, MNBT7168, four sequences, indexer rightsizing, spinwait16. Dense-FP8 and adaptive verification are off, abliteration is off, and vision is enabled. Initial candidate: 500,000 context with utilization 0.84, following upstream's conservative receipt. Independent per-node watchdogs retain a 12GiB host reserve. Configured context is not a passed long-context test; inspect validation receipts before claiming qualification.

`deployment.env.example` has the exact non-secret settings. Port 8888 is the native service. Port 8600 was already occupied by a compatibility adapter, which was preserved. The launcher patch keeps worker artifacts under Dev/state, supports the existing canonical HF cache, and refuses to delete existing containers. Old Flash and K275 containers are retained stopped for rollback. Spark 3 is left free, without restarting H3. Zima's Hugging Face uploads are independent and remain running.

`deploy-zima.sh` records the initial stop/build/ship attempt. Its first launch stopped at the port-8600 preflight; the configuration was corrected to 8888 before retrying launch. Do not rerun that controller against the now-stopped K275 attempt. For later launches, inspect existing container names and use fresh names rather than deleting rollback containers.

CPU checks: 41 passed; one upstream stale default assertion failed in `test_indexer_workspace.py`, which expects `stock` despite upstream now defaulting to `rightsize`. The E3 build static check and GPU overlay self-check passed. The default GPU check skips real-checkpoint parity when the HF cache is not mounted. Live validation is recorded separately and does not replace a full KLD evaluation.

Private deployment receipts are under `/home/mj-kang/Dev/state/glm53-flash-e3-20260909` on Zima. Build/serve logs and per-node caches use canonical Dev/logs and Dev/cache paths. Backups include K275 inspect records and the pre-cutover LibreChat/Compose files. Never resume K275 alongside Flash on the same GPUs.

## Startup blocker

Neither E3 startup was promoted. The first attempt, using the upstream expandable-segment allocator, emitted NVIDIA `NV_ERR_NO_MEMORY` allocation errors on Spark 2 during model initialization. The 12GiB/strict-kernel watchdog stopped it. A second attempt using `backend:cudaMallocAsync` emitted the same class of errors on Spark 1 and was also stopped. Both peers were intentionally stopped; neither attempt was reported as host-OOMKilled. After shutdown, both nodes were responsive with about 117GiB available and no remaining GPU compute processes.

Buddy allocator statistics showed very few highest-order free blocks despite abundant total free memory, consistent with fragmentation but not definitive proof of the cause. Host memory compaction or a clean reboot requires operator authorization/privilege. The web configuration has not been switched: it still targets the stopped K275 endpoint and is temporarily unavailable. The prepared E3 image, cached weights, all rollback containers and Zima upload jobs are preserved. Do not weaken the kernel guard or describe Flash as deployed before a successful boot and real generation checks.
