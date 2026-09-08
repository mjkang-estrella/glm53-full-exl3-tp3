# Reproduction inputs

All paths are canonical lab paths. Zima is the always-on controller, not a GPU or model-weight staging disk. Verify `/home/mj-kang/Dev/README.md` and current SSH/routing before changes.

| Input | Pinned identity / location |
|---|---|
| Runtime image, all Sparks | `glm53-exl3:e2-c190db1`; local image ID `sha256:b7ff496564001ee02655ce796370bf823a233a963f04c3d334a549bb5f0628e5` |
| llama-benchy | 0.4.0, clean commit `446dd42fde2ebbaa1d68a0dfe9dc1e5b833f95ad`, Spark 1 `/home/mj-kang/Dev/benchmark/llama-benchy/.venv/bin/llama-benchy` |
| BF16, encoding only | `/mnt/unas-models/ZAI/GLM-5.3-BF16`, revision `304b8051cfb2b260b61ce0cbe330e02a98e73639` |
| K3 basename | `GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z` |
| K275 basename | `GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z` |
| NAS checkpoints | `/mnt/unas-models/ZAI/<basename>` |
| Spark replicas | `/home/mj-kang/Dev/models/<basename>` |
| Packed rank weights | `/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/rank-packs/20260907T231500Z-k275-mixed-rank-local-v1/rank-N` on rank N |
| Replication receipt | `/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/20260907T224500Z/replication/SUMMARY.json` on Zima |
| Controller Python | `/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/eval-venv-numpy2.3.3/bin/python` on Zima; shell tools SSH, rsync, jq, tmux, Docker on Sparks |
| Numeric core | SHA256 `e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032` |
| ABI-specific ExLlama extension | SHA256 `7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2`; preserve the installed ABI-compatible image/cache, do not substitute an arbitrary wheel |

The image ID is a local Docker content identity, not a registry pull URL. This repository does not rebuild that image or the compiled extension from scratch. Missing image/rank packs/replication receipts are blocking dependencies, not permission to fabricate receipts or silently choose another model.

## Archive metadata identities

| Artifact | SHA256 |
|---|---|
| K3 MANIFEST.json | `de425ecadfba3b1fa7f60a500c9e156622830454ffad809a1f559dde020f2999` |
| K3 SHA256SUMS | `1ea4a0c11d6046060cb713f140a3605b8c50cb7260cea038114ceda6de4b1898` |
| K275 MANIFEST.json | `d0384bfd53e403b859e8a221b0429725b22600d48d49f2ada2cf8a0f777332c6` |
| K275 SHA256SUMS | `b3311519773b35a8ede1fe10975df5e97096e1870fc994c51626a1da03ea2457` |

Use the actual sealed receipt when checking files. Hashing metadata is a quick identity check; full `sha256sum -c SHA256SUMS` is an expensive checkpoint-content audit and must be scheduled to avoid storage contention with experiments.

## External encoder source

The ignored `encoder-r10/` directory is required for encoding, not for serving the sealed winner. Upstream source is pinned at Hugging Face `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`, revision `7c73450f05a151439d0f184f216b1eefcc394a31`, `reproducibility/r10`. The lab changes `r7_encoder/trellis.py`, `r10_codec.py`, and `types.py` for K2. The upstream README/SOURCE_SHA256SUMS therefore describe the original bundle, not the modified deployed closure.

`dependencies/encoder-deployed.sha256` pins the complete retrieved lab closure. Preserve upstream licensing/notices; no license is inferred for this project's entire imported source. To hydrate an empty clone on the Mac from the existing lab, without overwriting a differing existing encoder:

```bash
rsync -a --ignore-existing --exclude __pycache__/ --exclude '*.pyc' --exclude '*.pre-*' \
  mj-zima:/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3/encoder-r10/ encoder-r10/
shasum -a 256 -c dependencies/encoder-deployed.sha256
```

The private NAS source snapshot also retains this closure. Do not claim that an upstream-only checkout reproduces K2, or automatically run encoding while the three Sparks are serving.

## Evidence, outside Git

- Speed: `/mnt/unas-models/ZAI/GLM-5.3-EXL3-K275-speed-evidence-20260908`.
- Archive/quality: `/mnt/unas-models/ZAI/GLM-5.3-EXL3-archive-notes-20260908`.
- Private source/Git wrap-up: `/mnt/unas-models/ZAI/GLM-5.3-EXL3-reproduction-20260908`.
- Mutable campaign state: `/home/mj-kang/Dev/state/glm53-full-exl3-tp3/speed/20260908` on Zima.

Git contains curated numerical JSON, not raw JSONL, prompts/captures, weights, secrets, or generated compiler caches. Benchmark TTFR values in `results/speed-results.json` are milliseconds; `final-summary.json` uses seconds. Keep `accepted` separate from `eligible_for_selection`.
