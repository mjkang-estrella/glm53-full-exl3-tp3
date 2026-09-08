# Commit map

These are retrospective commits made during the September 8 wrap-up. Experiment dates and outcomes are in [PROGRESS.md](PROGRESS.md), not inferred from commit timestamps. Earlier loader/DCP work was imported as a baseline because no original Git history existed; it was not fabricated into independent measured changes.

| Commit | Improvement / responsibility | Verification or measurement |
|---|---|---|
| `b4244b0` | Available pre-speed runtime/encoder baseline | Historical import, not a standalone qualified deployment |
| `2204f16` | Skip LFU host synchronization for fully resident experts | Resident/no-resident observer regression; part of combined hot-path case |
| `a737b15` | DCP padding diagnostic prints once | Padding arithmetic unchanged; part of combined hot-path case |
| `8fb0aba` | Atomic checksum metadata, base-seal validation and archive tools | Hardlink replacement regression and retained archive audit receipts |
| `6c01e59` | Bounded graph/spin configuration controls | Shell syntax and measured graph/spin trials; not selected by default |
| `10d7183` | Fixed-budget quality experiments and rejected-candidate evidence | Training-only selection, scale restoration tests, pilot validation |
| `426bf6b` | Matched speed sweep, no-progress guard and eligibility-aware selection | Full measured matrix, failures retained |
| `eab047e` | Pinned qualified MTP3 profile; Git-safe source replication | Six-run confirmation, long-context and memory/kernel gates |
| `89b7b57` | Read-only live audit, source/results checks, external encoder hashes, benchy revision guard | Offline audit plus live image/profile/mount/watchdog verification |
| Documentation commit after these | Current agent handoff, dependency inventory, timeline and reproduction runbook | Documentation/link review and private archive verification |

NCCL/MTP/KV settings were parameter sweeps of one launcher, not separate numerical implementations. Their per-case configurations and measurements are preserved in `results/speed-results.json`; commits do not pretend every setting was an independent software patch.

## Private copies

The Mac repository is the source-control origin for this wrap-up. A private clone is kept at `/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3-repro` on Zima, and a self-contained Git bundle at `/mnt/unas-models/ZAI/GLM-5.3-EXL3-reproduction-20260908/glm53-reproduction.bundle`.

The `-repro` clone is for review/versioning. Launchers still target the canonical deployed `/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3` path. Review differences before any future source deployment; never overwrite live bind-mounted runtime code as a way to change a running model. During this wrap-up only documentation and the new read-only inspector are installed into that deployed directory. The running runtime and existing launchers are left unchanged.

To restore source into a new, empty directory:

```bash
git clone /mnt/unas-models/ZAI/GLM-5.3-EXL3-reproduction-20260908/glm53-reproduction.bundle NEW_EMPTY_DIRECTORY
```

Then read README.md and the dependency inventory. A source clone alone is not permission to start a model or encode. No public remote or push was added.
