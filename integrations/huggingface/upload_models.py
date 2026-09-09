#!/usr/bin/env python3
"""Zima-only sequential, resumable upload of two sealed NAS checkpoints.

Run in tmux with the dedicated HF environment. Never stage weight copies on
Zima root storage; staging contains symlinks and publication metadata only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time

from huggingface_hub import HfApi, CommitOperationAdd, RepoFile, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path('/home/mj-kang/Dev/state/glm53-full-exl3-tp3/hf-public-20260908')
STAGE = Path('/home/mj-kang/Dev/cache/glm53-hf-public-20260908')
NAS = Path('/mnt/unas-models/ZAI')
JOB = 'glm53-sealed-public-20260908-v1'
SPECS = [
    dict(key='k3', bits='3.0', repo='mj-kang/GLM-5.3-EXL3-3.0bpw-TP3',
         source='GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z',
         manifest='de425ecadfba3b1fa7f60a500c9e156622830454ffad809a1f559dde020f2999',
         ledger='1ea4a0c11d6046060cb713f140a3605b8c50cb7260cea038114ceda6de4b1898',
         kld='0.0339743205', top1='93.6004%'),
    dict(key='k275', bits='2.75', repo='mj-kang/GLM-5.3-EXL3-2.75bpw-TP3',
         source='GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z',
         manifest='d0384bfd53e403b859e8a221b0429725b22600d48d49f2ada2cf8a0f777332c6',
         ledger='b3311519773b35a8ede1fe10975df5e97096e1870fc994c51626a1da03ea2457',
         kld='0.0485539331', top1='92.6844%'),
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.publication-')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dump(path, data):
    write(path, json.dumps(data, indent=2, sort_keys=True) + '\n')


def status(phase, **extra):
    value = dict(job=JOB, phase=phase, updated_at=utc(), **extra)
    dump(ROOT / 'STATUS.json', value)
    print(json.dumps(value), flush=True)


def card(s):
    speed = ('The qualified K2.75 profile measured 13.062 decode tok/s over six llama-benchy runs '
             'with pp2048/requested tg256, concurrency 1. A 30,039-token retrieval prompt passed. '
             'The profile uses resident UVA, TP3/DCP3, MTP3/draft TP1, eager, prefill128, '
             'NCCL4/1MiB, FP8 KV 1GiB/rank, 32,768 total context, and a 12GiB host reserve.'
             if s['key'] == 'k275' else
             'The 13.06 tok/s and final 32K/MTP3 qualification in the linked project belong to K2.75, '
             'not this K3 checkpoint. Do not attribute those speed or memory results to K3.')
    return f'''---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model: zai-org/GLM-5.3-BF16
pipeline_tag: text-generation
tags:
- exl3
- quantized
- dgx-spark
- custom-runtime
---
# GLM-5.3 EXL3 {s['bits']} bpw, TP3 rotating uneven

**Upload status: incomplete unless `UPLOAD_COMPLETE.json` exists.** Check that marker before downloading for use.

This is a community quantization of the full GLM-5.3, not GLM-5.3 Flash. It retains all 256 routed experts and official top-8 routing, including the MTP layer. The named bpw is the routed-expert quantization target, not the whole-checkpoint average; passthrough tensors remain at source precision.

Geometry is rotating uneven 768/640/640 with no expert-channel padding. The 768-wide owner rotates across normal MoE layers; MTP layer 78 uses rank 2. K3 uses three-bit routed experts; K2.75 mixes 64 two-bit and 192 three-bit experts per routed layer. These are the original sealed versions, not the rejected scale/re-encoding experiments.

## Custom runtime required

This is NOT a stock Transformers, standard vLLM, llama.cpp/GGUF or drop-in ExLlama checkpoint. Use the custom loader, rank-pack preparation, pinned runtime dependencies and safety gates in [the reproduction repository](https://github.com/mjkang-estrella/glm53-full-exl3-tp3). The runtime image/compiled extension are not included here; the source project is an existing-homelab reproduction bundle, not a clean-machine installer.

## Lossless transport layout

The sealed checkpoint has more than 19,000 files at its root. To respect Hub folder limits, this repository puts the unchanged original files under `checkpoint/part-NNNN/`, at most 512 files per part. The original index and manifests are intentionally unchanged and become usable after restoration. `TRANSPORT_MANIFEST.json` records the exact mapping, byte sizes and SHA256 hashes. The source metadata is also preserved, including its original README and license.

Download the complete repository, then run:

```bash
hf download {s['repo']} --local-dir ./downloaded-model
python3 ./downloaded-model/restore_checkpoint.py /absolute/path/to/restored-checkpoint --verify
```

The default restoration links files without copying hundreds of gigabytes. Keep the downloaded snapshot/cache. Use `--copy` only if you want an independent copy and have enough space. `--verify` reads every file to validate SHA256; it can take substantial time. Never point restoration at an existing populated directory.

## Recorded evaluation

The recorded four-window teacher comparison reports KLD **{s['kld']} nats** and top-1 agreement **{s['top1']}**. These are limited calibration-window measurements, not a broad intelligence benchmark, not a percentage of information retained, and not comparable to unrelated GGUF evaluations without matched protocols. See the source repository for runtime/evaluation details and repeatability limits.

{speed}

## Provenance and integrity

- BF16 source revision: `304b8051cfb2b260b61ce0cbe330e02a98e73639`.
- Sealed source directory: `{s['source']}`.
- Original manifest SHA256: `{s['manifest']}`.
- Original checksum-ledger SHA256: `{s['ledger']}`.
- Full NAS archive audit on September 8 checked both checkpoints, 39,266 ledger entries in total.
- Publication verifies remote file sizes and content identities against the transport manifest before adding `UPLOAD_COMPLETE.json`.

Original model licensing is preserved in `LICENSE`. This repository does not replace that license with MIT or Apache. Quantized weight bytes are unchanged from the sealed NAS checkpoint.
'''


def prepare(s):
    source = NAS / s['source']
    stage = STAGE / s['key']
    if sha(source / 'MANIFEST.json') != s['manifest'] or sha(source / 'SHA256SUMS') != s['ledger']:
        raise RuntimeError('Sealed source metadata identity changed')
    seal = json.loads((source / 'ASSEMBLY_COMPLETE.json').read_text())
    if not seal['passed'] or seal['manifest_sha256'] != s['manifest'] or seal['sha256sums_sha256'] != s['ledger']:
        raise RuntimeError('Assembly marker differs from expected seal')
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    expected = dict(manifest['files'])
    for name in ('MANIFEST.json', 'SHA256SUMS', 'ASSEMBLY_COMPLETE.json'):
        expected[name] = dict(bytes=(source / name).stat().st_size, sha256=sha(source / name))
    forbidden = re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|\bhf_[A-Za-z0-9]{25,}\b|\bghp_[A-Za-z0-9]{30,}\b')
    files = []
    for index, (name, meta) in enumerate(sorted(expected.items())):
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise RuntimeError('Unsafe source manifest path')
        path = source / rel
        if not path.is_file() or path.stat().st_size != meta['bytes']:
            raise RuntimeError(f'Source file missing or size changed: {name}')
        if not name.endswith('.safetensors'):
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != meta['sha256'] or forbidden.search(body):
                raise RuntimeError(f'Metadata checksum or publication scan failed: {name}')
        # Nested original paths are flattened only in transport, then restored from this map.
        published = f'checkpoint/part-{index // 512:04d}/{index:05d}-{rel.name}'
        target = stage / published
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != path.resolve():
                raise RuntimeError('Existing stage symlink points elsewhere')
        elif target.exists():
            raise RuntimeError('Staging path is not the expected symlink')
        else:
            target.symlink_to(path)
        files.append(dict(original=name, published=published, bytes=meta['bytes'], sha256=meta['sha256']))
    plan = dict(job=JOB, repo=s['repo'], source_manifest=s['manifest'], files=files)
    dump(stage / 'TRANSPORT_MANIFEST.json', plan)
    write(stage / 'README.md', card(s))
    write(stage / 'LICENSE', (source / 'LICENSE').read_text())
    write(stage / 'restore_checkpoint.py', Path(__file__).with_name('restore_checkpoint.py').read_text())
    dump(stage / 'UPLOAD_PLAN.json', dict(job=JOB, repo=s['repo'], source_manifest=s['manifest'],
                                        file_count=len(files), payload_bytes=sum(f['bytes'] for f in files)))
    write(stage / '.gitattributes', '*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    return stage, files


def initialize(api, s, stage):
    try:
        info = api.repo_info(s['repo'], repo_type='model')
    except RepositoryNotFoundError:
        api.create_repo(s['repo'], repo_type='model', private=False, exist_ok=False)
    else:
        if info.private:
            raise RuntimeError('Unexpected private repository; refusing visibility mutation')
        old = json.loads(Path(hf_hub_download(s['repo'], 'UPLOAD_PLAN.json', force_download=True)).read_text())
        if old['job'] != JOB or old['source_manifest'] != s['manifest']:
            raise RuntimeError('Existing repository is not this upload job')
    roots = ['README.md', 'LICENSE', 'restore_checkpoint.py', 'TRANSPORT_MANIFEST.json', 'UPLOAD_PLAN.json', '.gitattributes']
    api.create_commit(s['repo'], repo_type='model', commit_message='Prepare sealed checkpoint upload and restoration instructions',
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=stage / name) for name in roots])


def verify_remote(api, s, stage, files):
    revision = api.repo_info(s['repo']).sha
    expected = {f['published']: f for f in files}
    for name in ('README.md', 'LICENSE', 'restore_checkpoint.py', 'TRANSPORT_MANIFEST.json', 'UPLOAD_PLAN.json', '.gitattributes'):
        path = stage / name
        expected[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    seen = set()
    for item in api.list_repo_tree(s['repo'], recursive=True, revision=revision, repo_type='model'):
        if not isinstance(item, RepoFile):
            continue
        name = item.path
        if name not in expected:
            if name == 'UPLOAD_COMPLETE.json':
                continue
            raise RuntimeError(f'Unexpected remote file: {name}')
        e = expected[name]
        if item.size != e['bytes']:
            raise RuntimeError(f'Remote size mismatch: {name}')
        if item.lfs:
            remote_sha = item.lfs.sha256 if hasattr(item.lfs, 'sha256') else item.lfs['sha256']
            if remote_sha != e['sha256']:
                raise RuntimeError(f'Remote SHA256 mismatch: {name}')
        else:
            body = (stage / name).read_bytes()
            git_sha = hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest()
            if item.blob_id != git_sha:
                raise RuntimeError(f'Remote Git blob mismatch: {name}')
        seen.add(name)
    if seen != set(expected):
        raise RuntimeError(f'Remote missing {len(set(expected) - seen)} files')
    complete = dict(job=JOB, verified_at=utc(), verified_payload_revision=revision,
                    verified_files=len(seen), payload_files=len(files),
                    payload_bytes=sum(f['bytes'] for f in files),
                    transport_manifest_sha256=sha(stage / 'TRANSPORT_MANIFEST.json'),
                    source_manifest_sha256=s['manifest'])
    receipt = ROOT / f'{s["key"]}-COMPLETE.json'
    dump(receipt, complete)
    api.upload_file(repo_id=s['repo'], path_in_repo='UPLOAD_COMPLETE.json', path_or_fileobj=receipt,
                    commit_message='Mark upload complete after remote content verification')
    return complete


def bounded_batches(files, max_files=8, max_bytes=128 * 1024**2):
    """Large individual files stand alone and are streamed by the LFS client."""
    batch, size = [], 0
    for item in files:
        if batch and (len(batch) >= max_files or size + item['bytes'] > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += item['bytes']
    if batch:
        yield batch


def upload_bounded(api, s, stage, files, completed):
    # Resume from verified server content, not from stale local progress files.
    expected = {f['published']: f for f in files}
    committed = set()
    revision = api.repo_info(s['repo']).sha
    for item in api.list_repo_tree(s['repo'], recursive=True, revision=revision):
        if not isinstance(item, RepoFile) or item.path not in expected:
            continue
        e = expected[item.path]
        if item.size != e['bytes']:
            raise RuntimeError('Existing remote payload size mismatch')
        if item.lfs:
            actual = item.lfs.sha256 if hasattr(item.lfs, 'sha256') else item.lfs['sha256']
            if actual != e['sha256']:
                raise RuntimeError('Existing remote payload hash mismatch')
        else:
            body = (stage / item.path).read_bytes()
            if item.blob_id != hashlib.sha1(f'blob {len(body)}\0'.encode() + body).hexdigest():
                raise RuntimeError('Existing remote metadata hash mismatch')
        committed.add(item.path)
    count = len(committed)
    committed_bytes = sum(expected[n]['bytes'] for n in committed)
    total = sum(f['bytes'] for f in files)
    for batch in bounded_batches([f for f in files if f['published'] not in committed]):
        status('uploading', repo=s['repo'], transport='single-thread-lfs',
               committed_files=count, committed_bytes=committed_bytes,
               total_files=len(files), payload_bytes=total,
               batch_files=len(batch), batch_bytes=sum(f['bytes'] for f in batch), completed=completed)
        for attempt in range(5):
            try:
                # Fresh operations after a retry, since SDK operations are mutable.
                api.create_commit(s['repo'], repo_type='model', num_threads=1,
                    commit_message=f'Upload sealed payload files {count + 1}-{count + len(batch)}',
                    operations=[CommitOperationAdd(path_in_repo=f['published'], path_or_fileobj=stage / f['published']) for f in batch])
                break
            except Exception as error:
                code = getattr(getattr(error, 'response', None), 'status_code', None)
                if code in (400, 401, 402, 403, 404, 409, 422) or attempt == 4:
                    raise
                status('retry_wait', repo=s['repo'], attempt=attempt + 1,
                       http_status=code, committed_files=count, committed_bytes=committed_bytes)
                time.sleep(min(15 * 2**attempt, 120))
        count += len(batch)
        committed_bytes += sum(f['bytes'] for f in batch)
    status('payload_committed', repo=s['repo'], committed_files=count,
           committed_bytes=committed_bytes, payload_bytes=total, completed=completed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    api = HfApi()
    if api.whoami()['name'] != 'mj-kang':
        raise RuntimeError('Unexpected Hugging Face account')
    def disk_guard():
        while True:
            if shutil.disk_usage('/home/mj-kang/Dev/cache').free < 8 * 1024**3:
                status('stopped_disk_guard', reserve_gib=8)
                os._exit(3)
            time.sleep(15)
    threading.Thread(target=disk_guard, daemon=True).start()
    prepared = []
    for s in SPECS:
        status('preparing', repo=s['repo'])
        stage, files = prepare(s)
        prepared.append((s, stage, files))
    if args.prepare_only:
        status('prepared', repos=[s['repo'] for s in SPECS])
        return
    for s, stage, files in prepared:
        initialize(api, s, stage)
    completed = []
    for s, stage, files in prepared:
        status('uploading', repo=s['repo'], files=len(files), payload_bytes=sum(f['bytes'] for f in files), completed=completed)
        upload_bounded(api, s, stage, files, completed)
        status('verifying_remote', repo=s['repo'], completed=completed)
        verify_remote(api, s, stage, files)
        completed.append(s['repo'])
    status('complete', repos=completed)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Do not emit HTTP headers, credentials or raw SDK exception details.
        status('failed', error_type=type(e).__name__, action='Inspect sanitized logs; do not retry a permanent quota/auth error blindly')
        raise SystemExit(1)
