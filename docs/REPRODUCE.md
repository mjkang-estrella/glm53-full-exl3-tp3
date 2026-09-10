# Reproduce the qualified profile

This runbook targets the existing three-Spark homelab. The current service should remain running unless the operator authorizes a restart or experiment. A laptop disconnect must not terminate a long-running controller: use SSH to Zima and Zima tmux, not a separate Codex session.

## 1. Audit source and dependencies without changing serving

On the Mac, in this repository:

```bash
git status --short
python3 scripts/verify_reproduction.py
```

Check [dependencies](DEPENDENCIES.md). Hydrate the external encoder only if needed. The source audit checks syntax, safety regressions and recorded numerical invariants; it does not simulate CUDA or remeasure throughput.

To obtain either sealed checkpoint without access to the private NAS, use the verified public revisions in [DOWNLOADS.md](DOWNLOADS.md) and restore their original layout first. No re-encoding is needed. Public model availability does not supply the pinned runtime image or replace the existing-lab preflight requirements.

On Zima, from `/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3`:

```bash
bash scripts/inspect_speed_winner.sh speed-qualified-final
```

This checks exact container/image/model/rank-pack/profile identity, watchdog sessions and HTTP health. It does not allocate KV, generate tokens, or prove recent kernel cleanliness. Inspect current watchdog metrics and kernel logs before any workload. Preserve at least 12 GiB available host memory per node and zero model-container swap.

A fresh source clone needs the private SSH config (use `zima-ssh-config.example` as a template), trusted host keys, pinned image on every Spark, sealed replicas, rank packs, and the existing real-test replication receipt. Do not invent a new receipt stamp. Keep source at the canonical project path expected by launchers, and use the same Sync-managed RoCE ring. Missing dependencies block a launch.

## 2. Restart only when explicitly authorized

First discover the actual active attempt; the name below is the historical qualified one. Stop only that attempt. The stop helper preserves final state, logs, watchdog metrics and audits; it does not restore Flash/H3. A nonzero exit requires inspection before proceeding.

```bash
bash scripts/stop_candidate_attempt.sh 20260907T224500Z speed-qualified-final
tmux new-session -s glm53-reproduce
```

Inside that Zima tmux session, choose a genuinely new label, then:

```bash
bash scripts/start_speed_winner.sh YOUR_UNIQUE_ATTEMPT_LABEL
bash scripts/inspect_speed_winner.sh YOUR_UNIQUE_ATTEMPT_LABEL
```

The launcher refuses overlapping GPU workloads and reused state labels. It uses original K275, TP3/DCP3, resident UVA, MTP3/draft TP1, eager, prefill128, NCCL4/1MiB, FP8 KV 1GiB/rank, 32K/one sequence, stock spin, and 12GiB guards. Failure stops candidate ranks without automatic Flash/H3 restoration. Do not use the archived `speed_campaign.py` or `long_speed_gate_resume.sh` as restart helpers; they contain historical transition labels and controller assumptions.

## 3. Prove generation, then benchmark

Record a new output directory under `/home/mj-kang/Dev/state/glm53-full-exl3-tp3/reproduction/` and the Git commit, image ID, profile, start time, and actual attempt label. Do not reuse old result files. Use a short native completion before the full test; require coherent exact output and `finish_reason=stop`. The existing `raw_native_completion` in `scripts/long_context_probe.py` implements the template used by the campaign.

For the matched llama-benchy protocol, run from Spark 1, using the preserved environment and clean pinned checkout:

```bash
cd /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3
bash scripts/run_speed_bench.sh YOUR_UNIQUE_BENCHMARK_LABEL
```

That helper runs pp2048 / requested tg256 / exact-tg / depth0 / three measured runs / c1 / no cache / generation latency mode. Results go under `/home/mj-kang/Dev/benchmark/llama-benchy/results/k275-speed-20260908/<label>` on Spark 1. The date is a historical namespace, not the new run date. It refuses an existing result directory. Verify `benchy-commit.txt` and a clean diff before accepting a comparison. TTFR in the raw result is milliseconds.

**Do not leave the standalone helper unattended.** It does not include the campaign's no-progress controller. For unattended experiments, adapt `speed_campaign.py:benchmark` into a new, reviewed controller with unique state: its 120-second progress-file guard applies only to pp2048, stops that exact candidate on a stall, and preserves failure evidence. Never rerun the old complete campaign blindly. Record failed/partial output as rejected, not zero or a successful score.

Repeat the identical selected configuration for a second three-run set. Compare all six values with the recorded 13.062 ± 0.801 tok/s, not just the best sample. MTP acceptance is workload dependent. Prefill/TTFR and decode are separate measurements.

## 4. Qualify long context and memory

Run the following on Zima inside tmux, using a new output path whose parent exists:

```bash
/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/eval-venv-numpy2.3.3/bin/python \
  scripts/long_context_probe.py --endpoint http://192.168.0.238:8893 \
  --output /home/mj-kang/Dev/state/glm53-full-exl3-tp3/reproduction/YOUR_RUN/long30k.json \
  --target-tokens 30000 --mode raw-native --max-tokens 64 --timeout-seconds 900
```

Do not apply the 120-second benchmark-progress timeout to this prefill. Require `.passed == true`, exact retrieval and normal stop, and inspect actual `usage.prompt_tokens`. Historical actual usage was 30,039, not the prompt-builder estimate 30,025. This is a targeted native retrieval gate with low reasoning, not a comprehensive reasoning evaluation.

Audit the whole new run window with `scripts/kernel_audit.py --since <UTC_START> --until <UTC_END> --ssh-config zima-ssh-config --output <NEW_FILE>`. Collect each rank's watchdog `metrics.csv` into the attempt's `ranks/rank-N.metrics.csv`, then run `scripts/summarize_watchdog_metrics.py --attempt <LABEL> --attempt-dir <ATTEMPT_DIR> --output <NEW_FILE>`.

Acceptance requires normal generations, complete benchmark sets, long retrieval, available host memory never below 12GiB, no model-container swap, no watchdog stop, no OOMKilled ranks, and a zero-fault candidate kernel window. Leave the qualified service running afterward unless the operator requests otherwise. Do not promote a public route automatically.

## 5. Preserve evidence and rollback

Archive source commit/profile, raw benchy results/progress, generation checks, memory series, kernel audit, image identity and failure notes on NAS. Curate numerical summaries into Git; keep raw captures and credentials outside it. Mark rows `eligible_for_selection=false` when later checks invalidate an earlier short pass.

On failure, stop only the exact candidate, retain logs and inspect the cause. With authorization, restart the unchanged sealed K275 using the winner helper and a new label; repeat the gates. Do not weaken memory/kernel checks or switch automatically to Flash/H3. Graphs+MTP3 and larger-prefill MTP are recorded failures under the tested settings, not the next default experiment.

## CPU test environment

Full CPU unit tests need PyTorch, safetensors and pytest. The Mac's default Python may lack them. Use a separate source snapshot in the pinned Spark image with `--runtime runc --network none`, no GPU/device/host-network flags, bounded CPU/memory, a read-only source mount, and `NVIDIA_VISIBLE_DEVICES=void`. Never run tests by changing the serving container. Set `PYTHONPATH=/snapshot` and run `python3 -m pytest -q -p no:cacheprovider /snapshot/tests`, then `python3 /snapshot/scripts/test_quality_scale.py`. Some tests inspect historical launcher contracts; interpret failures before changing production logic.
