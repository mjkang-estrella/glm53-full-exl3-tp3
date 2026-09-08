#!/usr/bin/env python3
"""Zima: free only this campaign's Spark-side logit parts after NAS verification.

Keeps the local capture receipts and all NAS evidence. The Spark container is
used only to remove its own root-owned diagnostic files, never model files.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

PROJECT = Path('/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3')
ROOT = Path('/home/mj-kang/Dev/state/glm53-full-exl3-tp3/quality/20260908-scale-v2')
EVIDENCE = Path('/mnt/unas-models/ZAI/GLM-5.3-EXL3-K275-quality-experiments-20260908-scale-v2')
CONTAINER = 'glm53-k3-cand-k275-quality-v2-rank0'


def verify(path, row):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']:
        raise ValueError(f'file shape differs: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if digest.hexdigest() != row['sha256']:
        raise ValueError(f'file digest differs: {path}')


# This code has a literal, narrow /capture root. It accepts only manifest part
# names, checks the completed receipt, and never removes directories or models.
REMOTE = r'''
import hashlib,json,os,re,sys
from pathlib import Path
p=json.load(sys.stdin)
assert re.fullmatch(r'(selection|confirmation)-\d{4}-\d{10}',p['capture_id'])
base=Path('/capture')/p['capture_id']
assert not base.is_symlink() and base.resolve().parent==Path('/capture')
assert not (Path('/capture')/'ARM.json').exists()
receipt=json.loads((base/'CAPTURE_COMPLETE.json').read_text())
assert receipt==p['receipt'] and receipt['passed'] is True
existing=sum((base/row['path']).exists() for row in receipt['parts'])
assert existing in (0,len(receipt['parts'])), 'partial cleanup requires inspection'
for row in receipt['parts']:
    assert re.fullmatch(r'part-\d{4}\.f32',row['path'])
    if existing==0: continue
    path=base/row['path']
    assert not path.is_symlink() and path.is_file() and path.stat().st_size==row['bytes']
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(1<<20): h.update(chunk)
        os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    assert h.hexdigest()==row['sha256'], str(path)
for row in receipt['parts']:
    if existing: (base/row['path']).unlink()
print(json.dumps({'capture_id':p['capture_id'],'bytes_released':sum(r['bytes'] for r in receipt['parts']),
                  'nas_copy':p['nas_copy'],'local_receipt_preserved':True,'already_absent':existing==0}))
'''


def main():
    complete = json.loads((ROOT/'REPEATED_CONFIRMATION.json').read_text())
    assert complete['phase'] == 'complete'
    output = ROOT/'ARCHIVED_LOCAL_DUPLICATES.jsonl'
    completed = {json.loads(line)['capture_id'] for line in output.read_text().splitlines()} if output.exists() else set()
    manifests = sorted(EVIDENCE.glob('*/*/SCHEDULER_CAPTURE_COMPLETE.json'))
    assert len(manifests) == 38, f'unexpected capture count {len(manifests)}'
    seen = set()
    with output.open('a') as record:
        for manifest in manifests:
            receipt = json.loads(manifest.read_text())
            capture_id = receipt['capture_id']
            assert re.fullmatch(r'(selection|confirmation)-\d{4}-\d{10}', capture_id)
            assert capture_id not in seen and receipt['passed'] is True
            seen.add(capture_id)
            if capture_id in completed:
                continue
            for row in receipt['parts']:
                assert re.fullmatch(r'part-\d{4}\.f32', row['path'])
                verify(manifest.parent/row['path'], row)
            payload = dict(capture_id=capture_id, receipt=receipt, nas_copy=str(manifest.parent))
            result = subprocess.run(
                ['ssh','-F',str(PROJECT/'zima-ssh-config'),'mj-spark-1',
                 shlex.join(['docker','exec','-i',CONTAINER,'python3','-S','-c',REMOTE])],
                input=json.dumps(payload),text=True,capture_output=True,check=True,timeout=90)
            archived = json.loads(result.stdout)
            record.write(json.dumps(archived)+'\n')
            record.flush()
            os.fsync(record.fileno())
            print(json.dumps(archived),flush=True)


if __name__ == '__main__':
    main()
