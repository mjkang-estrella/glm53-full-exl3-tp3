# Current serving profile, September 8

Public-route update after the original qualification: authenticated `https://ai.mj-kang.com` now uses this K275 service, labeled `GLM-5.3 K275 (32K)`. See `integrations/librechat/README.md` for the mapping, compatibility fix and rollback. The earlier qualification notes below describe the pre-cutover state.

The original K275 checkpoint is running on all three Sparks as attempt `speed-qualified-final`, endpoint `http://192.168.0.238:8893`. The API name remains the legacy `GLM-5.3-K3-TP3-CANDIDATE`; the mounted weights are K275, not K3.

- TP3/DCP3, one sequence, 32,768-token limit, interleave 1.
- Original mixed 64 K2 / 192 K3 experts per routed layer; all experts and routing retained.
- Resident UVA, 256 experts per layer, no recurrent NVMe expert loading.
- Three MTP draft tokens, `draft_tensor_parallel_size=1` setting.
- NCCL four channels, 1 MiB buffers.
- Eager execution; CUDA graphs off. Stock spin-wait.
- 128-token prefill chunks; FP8 KV, 1 GiB per rank.
- 12 GiB host reserve and per-node watchdogs remain active.

The same configuration measured 12.950 and 13.174 decode tok/s in two three-run llama-benchy sets. Six-run average is about 13.06 tok/s, compared with 5.121 tok/s before tuning. This is pp2048, requested tg256, concurrency 1, no cache, generation latency mode; it excludes prefill from decode throughput.

A real request used 30,039 prompt tokens and returned exactly `K3-NEEDLE-74291`, stopping normally in 186.95 seconds. Minimum host availability through qualification was 12.762 / 14.451 / 14.579 GiB across the three ranks, with zero model-container swap.

The nominally faster MTP/prefill256 cases were excluded after repeat stalls. Graph/MTP3 combinations failed the memory gate. See `SPEED_RESULTS.md` and `SPEED_EXPERIMENTS.md` for the complete measured matrix and failure details. No public/client route was changed; Flash/H3 remain stopped. Sealed K3 and K275 archives were not modified.

## Stop and restart

Run on Zima from `/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3`:

```bash
bash scripts/stop_candidate_attempt.sh 20260907T224500Z speed-qualified-final
bash scripts/start_speed_winner.sh NEW_UNIQUE_ATTEMPT_LABEL
```

The restart helper refuses an overlapping GPU workload. Its new attempt label must be unique. Do not rerun the whole speed campaign to restart the model. Require health and a real generation after any restart.
