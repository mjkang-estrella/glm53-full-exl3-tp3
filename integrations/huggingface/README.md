# Public model upload from Zima

## Completed September 10

Both public repositories now contain verified `UPLOAD_COMPLETE.json` markers. K3 contains 19,559 payload files / 315,867,635,438 bytes. K2.75 contains 19,712 payload files / 292,916,229,281 bytes.

K2.75 finished transferring its payload but the original finalizer rejected its root `.gitattributes`: the upload added four exact-path LFS rules for large JSON files. The new verifier accepts only standard LFS directives for explicitly expected file paths, plus the original safetensors rule. It still verifies every payload size and content identity. It validates the attributes at the same pinned revision as the payload and records their SHA256 and rules in the receipt. Publishing the completion marker uses a parent-commit guard to reject concurrent repository changes.

K2.75 final verification passed at `2026-09-10T04:56:53Z` against payload revision `ef319e5fe2ab1c1f8d7d551963ed7e6148ea53ab`: all 19,718 expected payload/publication files passed. No weights were re-uploaded or changed. The historical failed controller record is retained; the canonical STATUS.json now records completion after explicit finalization. Three new tests cover valid upload-generated rules, unknown paths/wildcards, changed filters and missing/duplicate base rules.

The attempt history below remains for audit. Do not restart the uploader just because an old process receipt records its earlier finalization failure.

## September 9 rate-limit recovery

The first LFS recovery made progress to 201.13 GB committed, then received HTTP 429. Its short retry budget expired before the limit reset. This was not another OOM event.

The current job uses tmux session `glm53-hf-rate-recovery` and the same supervised recovery scope. Transfers remain single-threaded LFS with the same 4 GiB memory cap, but each commit groups up to 64 file paths or 1 GiB. Payloads are streamed sequentially, not buffered as a parallel folder upload. Commit starts are paced at least 60 seconds apart.

HTTP 429 now waits for `Retry-After` or the `RateLimit` reset time. When neither is available it backs off for 10 minutes, increasing to at most one hour between attempts. STATUS.json records `rate_limit_wait` and `retry_at`, while PROCESS.json continues reporting liveness. Rate-limit waiting does not exhaust the five-attempt budget used for other transient errors. Permanent authorization/quota errors still stop without purchasing storage or changing credentials.

The older attempt descriptions below remain as historical evidence. The current supervisor/status paths and completion-marker rules are unchanged.

## September 9 recovery

The first 256-file streaming upload hit its 4 GiB cgroup cap after 54 minutes, with 186.62 GB of K3 committed. K2.75 had not started. The old STATUS.json survived with a stale uploading phase; the original failed scope and pre-recovery status are retained as evidence.

The recovery uses explicit commits of at most eight files or 128 MiB, one LFS upload thread, and disables the Xet client for this run. A larger individual file is streamed alone. Already committed payload files are verified by remote size and content identity and skipped. Transient errors receive bounded retries; permanent authorization/quota errors stop the job. The 4 GiB memory cap and zero process swap remain unchanged.

The active controller is now the `glm53-hf-recovery` tmux session and `glm53-hf-upload-recovery-20260909.scope`. `supervise_upload.py` runs outside that scope, writes PROCESS.json every 15 seconds, and changes STATUS.json to failed if the child exits or is OOM-killed. This makes failures visible; it does not automatically restart a failed upload. Current status commands:

```bash
ssh mj-zima 'cat /home/mj-kang/Dev/state/glm53-full-exl3-tp3/hf-public-20260908/PROCESS.json; cat /home/mj-kang/Dev/state/glm53-full-exl3-tp3/hf-public-20260908/STATUS.json'
```

Start future recoveries through `supervise_upload.py` inside Zima tmux, after inspecting the failure and ensuring no active controller. The original launch details below describe the first attempt and remain for audit.

The operator authorized public upload under the signed-in personal Hugging Face account and required independence from the laptop. The controller runs on Zima, reads the sealed NAS checkpoints, and does not use or restart a Spark GPU.

Repositories:

- `mj-kang/GLM-5.3-EXL3-3.0bpw-TP3`
- `mj-kang/GLM-5.3-EXL3-2.75bpw-TP3`

They are **incomplete until each repository contains `UPLOAD_COMPLETE.json`**. K3 uploads first, then K2.75. Do not infer completion from an existing repository or model card.

## Controller and resource limits

- Zima tmux session: `glm53-hf-upload`.
- User systemd scope: `glm53-hf-upload-20260908.scope`.
- User lingering was verified enabled, so laptop disconnection does not stop the scope.
- Scope limits: 4 GiB memory, no process swap, CPU quota 200%.
- Client: dedicated `/home/mj-kang/Dev/cache/hf-model-upload-20260908-venv`, `huggingface_hub==1.30.0`, `hf_xet==1.6.0`.
- Upload worker count: two. Both uploads are sequential, not competing for NAS bandwidth.
- Disk guard stops the process if Zima root free space falls below 8 GiB.
- The token remains in Zima's existing Hugging Face credential store, never in scripts, Git or logs.

State and log directory: `/home/mj-kang/Dev/state/glm53-full-exl3-tp3/hf-public-20260908`.
Transport staging: `/home/mj-kang/Dev/cache/glm53-hf-public-20260908`.
Xet cache: `/home/mj-kang/Dev/cache/glm53-hf-xet-upload-20260908`.

Staging consists of symlinks to NAS plus small publication files. No checkpoint copies are placed on Zima root storage. The SDK's resumable metadata and bounded Xet cache remain separate from the sealed models.

Check both process state and progress, because a historical STATUS.json is not proof that a process is still alive:

```bash
ssh mj-zima 'systemctl --user show glm53-hf-upload-20260908.scope -p ActiveState -p MemoryCurrent -p MemoryMax; cat /home/mj-kang/Dev/state/glm53-full-exl3-tp3/hf-public-20260908/STATUS.json'
ssh -t mj-zima tmux attach -t glm53-hf-upload
```

The program log is `upload.log` in the state directory; Hugging Face reports transfer progress there. Completed models get `k3-COMPLETE.json` or `k275-COMPLETE.json` locally and `UPLOAD_COMPLETE.json` remotely. Temporary failures are retried by the client. Authentication/quota or scope OOM failures require inspection, not blind retry or a storage purchase. Laptop disconnect is supported; automatic recovery after a Zima reboot is not configured.

## Lossless transport, not requantization

The models have more than 19,000 files in the original top-level layout, above the Hub's 10,000-entry folder limit. Upload names are spread over `checkpoint/part-NNNN/`, at most 512 per folder. Every payload byte, including original indexes, license and seal metadata, is preserved. The top-level `TRANSPORT_MANIFEST.json` maps each published file to its exact original path, size and SHA256.

`restore_checkpoint.py` reconstructs the original flat and nested paths after a complete snapshot download. It refuses traversal paths, incomplete downloads and nonempty destinations. `--verify` checks every downloaded SHA256 before creating links; `--copy` requests an independent full-size copy. Three CPU tests cover exact restoration, overwrite refusal, missing/corrupt payloads and traversal rejection.

Before upload, the controller validates the pinned source manifest/ledger/assembly marker, all file sizes and all non-weight metadata hashes, and scans metadata for common credential patterns. The earlier full NAS audit remains the source weight-integrity evidence. After transfer it checks every remote file's size and LFS SHA256, or Git blob identity for non-LFS files, before writing the completion marker. No new KLD or serving claim is inferred from publication.

## Start or resume deliberately

First verify no controller is running and inspect STATUS.json/logs for any previous failure. The launcher takes an exclusive flock to prevent duplicate controllers. Run on Zima inside tmux, from the canonical `-repro` checkout:

```bash
systemd-run --user --scope --unit=glm53-hf-upload-20260908 \
  -p MemoryMax=4G -p MemorySwapMax=0 -p CPUQuota=200% \
  bash /home/mj-kang/Dev/experiment/glm53-full-exl3-tp3-repro/integrations/huggingface/run_upload.sh
```

Do not remove the stage or SDK cache when resuming. The controller verifies ownership of an existing repository with its `UPLOAD_PLAN.json` before writing, and never replaces an unrelated repository. It does not buy extra Hub storage, change token permissions, or mutate the source checkpoint.
