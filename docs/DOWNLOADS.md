# Download and restore the published models

Both original sealed checkpoints are public and verified complete. No weight conversion, rescaling or re-encoding occurred during publication.

They are quantized versions of GLM-5.3 BF16, not additional fine-tunes. The model-card metadata explicitly classifies the relationship as `quantized`.

| Model | Completed download revision | Verified payload |
|---|---|---|
| [K3 / 3.0 bpw](https://huggingface.co/mj-kang/GLM-5.3-EXL3-3.0bpw-TP3) | `814a5ae9668af03d90131d9c6e35b6f6021fee5b` | 19,559 files; 315,867,635,438 bytes |
| [K2.75 / 2.75 bpw](https://huggingface.co/mj-kang/GLM-5.3-EXL3-2.75bpw-TP3) | `a266e4e8b9f9967d395ab8276283cbd5e8452402` | 19,712 files; 292,916,229,281 bytes |

Use these completed revisions, which include the corrected quantization cards, not the earlier `verified_payload_revision` in each receipt. That earlier snapshot precedes the completion marker. The later card-only edits did not change any weight payload. Exact marker hashes, payload revisions, transport-manifest hashes and verification timestamps are preserved in [model-publication.json](../results/model-publication.json).

## Download

Run on a machine with enough storage, such as a Spark or NAS client. Do not download hundreds of gigabytes onto Zima's small root filesystem. Use the Hugging Face CLI in a separately managed environment; the original uploader used `huggingface_hub==1.30.0`.

```bash
hf download mj-kang/GLM-5.3-EXL3-3.0bpw-TP3 \
  --revision 814a5ae9668af03d90131d9c6e35b6f6021fee5b \
  --local-dir ./glm53-k3-download

hf download mj-kang/GLM-5.3-EXL3-2.75bpw-TP3 \
  --revision a266e4e8b9f9967d395ab8276283cbd5e8452402 \
  --local-dir ./glm53-k275-download
```

## Restore the original checkpoint layout

The Hub copies use `checkpoint/part-NNNN/` folders to stay within folder-entry limits. Original indexes, tensor files, license and sealed metadata remain byte-identical; their original names are mapped in `TRANSPORT_MANIFEST.json`.

Choose new or empty destinations:

```bash
python3 ./glm53-k3-download/restore_checkpoint.py /absolute/path/to/GLM-5.3-K3 --verify
python3 ./glm53-k275-download/restore_checkpoint.py /absolute/path/to/GLM-5.3-K275 --verify
```

`--verify` hashes every downloaded file before creating the restored layout. Expect a full disk read of each checkpoint. By default, restoration uses hardlinks where possible and otherwise symlinks, so keep the downloaded snapshot/cache. Add `--copy` only if you want independent copies and have another full checkpoint's worth of disk space.

The restorer requires `UPLOAD_COMPLETE.json`, validates its transport-manifest hash, rejects unsafe paths and refuses to overwrite a populated destination. Root `.gitattributes` includes upload-generated LFS rules; do not strip them. The original sealed checkpoint's own `.gitattributes` is separately preserved in the transport layout.

## Run the custom runtime

These are **not standard Transformers, stock vLLM, llama.cpp/GGUF or drop-in ExLlama checkpoints**. Publication supplies the weights, not a clean-machine installer. The pinned runtime image, compiled extension, TP3 geometry and rank-pack preparation remain required. Read [dependencies](DEPENDENCIES.md) and the [existing-homelab runbook](REPRODUCE.md) before serving.

The 13.06 tok/s, 32K and MTP3 qualification belongs to original K2.75, not K3. Upload verification establishes file integrity, not a new quality, context or speed result. Preserve the supplied model license.

## Verification outcome

K3 passed remote verification on September 9; K2.75 passed on September 10 after its upload-generated LFS rules were validated. The checks covered all 19,565 K3 and 19,718 K2.75 payload/publication files before adding completion markers. Weight payloads were checked using remote LFS SHA256 identities; non-LFS files used Git blob identities, with sizes checked throughout.

The previous K2.75 finalization failure was metadata-only. No missing payload or weight-hash mismatch was found, and finalization required no weight re-upload. See [upload history and verifier details](../integrations/huggingface/README.md).
