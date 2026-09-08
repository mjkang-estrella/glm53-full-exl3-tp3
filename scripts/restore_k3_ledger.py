#!/usr/bin/env python3
"""Restore exact sealed K3 ledger bytes, breaking the accidentally shared link."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

NAME='GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z'
EXPECTED='1ea4a0c11d6046060cb713f140a3605b8c50cb7260cea038114ceda6de4b1898'
BAD='b3311519773b35a8ede1fe10975df5e97096e1870fc994c51626a1da03ea2457'


def digest(b): return hashlib.sha256(b).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True)
    args=ap.parse_args()
    assert args.root.name==NAME and args.root.parent in (Path('/mnt/unas-models/ZAI'),Path('/home/mj-kang/Dev/models'))
    assert not args.root.is_symlink()
    raw=(args.root/'MANIFEST.json').read_bytes()
    marker=json.loads((args.root/'ASSEMBLY_COMPLETE.json').read_text())
    assert digest(raw)==marker['manifest_sha256']
    assert marker['sha256sums_sha256']==EXPECTED
    files=json.loads(raw)['files']
    records={k:v['sha256'] for k,v in files.items()}
    records['MANIFEST.json']=digest(raw)
    restored=''.join(f'{h}  {name}\n' for name,h in sorted(records.items())).encode()
    assert digest(restored)==EXPECTED, 'reconstruction must match original sealed hash exactly'
    target=args.root/'SHA256SUMS'
    before=target.read_bytes()
    assert digest(before) in (BAD,EXPECTED)
    args.evidence.mkdir(parents=True,exist_ok=True)
    backup=args.evidence/'K3-SHA256SUMS.before-repair'
    if not backup.exists(): backup.write_bytes(before)
    assert digest(backup.read_bytes()) in (BAD,EXPECTED)
    if digest(before)!=EXPECTED:
        fd,name=tempfile.mkstemp(prefix='.SHA256SUMS.repair-',dir=args.root)
        with os.fdopen(fd,'wb') as f:
            f.write(restored);f.flush();os.fsync(f.fileno())
        os.chmod(name,0o644)
        os.replace(name,target)
    assert digest(target.read_bytes())==EXPECTED
    receipt=dict(path=str(target),before_sha256=digest(before),restored_sha256=EXPECTED,
                 exact_original_bytes_proven=True,weight_files_modified=False)
    (args.evidence/'LEDGER_REPAIR.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__': main()
