#!/usr/bin/env python3
"""Restore a downloaded transport layout, without changing tensor bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe relative path: {name}')
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def restore(source, target, *, verify=False, copy=False):
    manifest_path = source / 'TRANSPORT_MANIFEST.json'
    complete = json.loads((source / 'UPLOAD_COMPLETE.json').read_text())
    if complete['transport_manifest_sha256'] != digest(manifest_path):
        raise ValueError('Completion marker and transport manifest differ')
    manifest = json.loads(manifest_path.read_text())
    files = manifest['files']
    if target.exists() and any(target.iterdir()):
        raise ValueError('Destination must be new or empty; refusing to overwrite files')
    for item in files:
        safe_relative(item['original'])
        path = source / safe_relative(item['published'])
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f'Missing or incomplete download: {item["published"]}')
        if verify and digest(path) != item['sha256']:
            raise ValueError(f'Checksum mismatch: {item["published"]}')
    target.mkdir(parents=True, exist_ok=True)
    for item in files:
        src = (source / item['published']).resolve()
        dst = target / item['original']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError:
                dst.symlink_to(src)
    print(f'Restored {len(files)} files to {target}. Tensor bytes were not converted.')
    if not copy:
        print('Keep the downloaded snapshot/cache: cross-filesystem links may reference it.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    p.add_argument('--verify', action='store_true', help='Hash every downloaded file before restoration')
    p.add_argument('--copy', action='store_true', help='Copy instead of linking; needs another full checkpoint of disk space')
    args = p.parse_args()
    restore(Path(__file__).resolve().parent, args.destination.resolve(), verify=args.verify, copy=args.copy)
