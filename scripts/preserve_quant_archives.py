#!/usr/bin/env python3
"""Zima: audit existing immutable NAS checkpoints; never modify model files."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

BASE = Path('/mnt/unas-models/ZAI')
NOTES = BASE/'GLM-5.3-EXL3-archive-notes-20260908'
MODELS = [
 'GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-assembled-20260906T034145Z',
 'GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-rotating-uneven-v1-assembled-20260907T210500Z']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8<<20): h.update(chunk)
    return h.hexdigest()


def write(name, data):
    temp=NOTES/(name+'.tmp')
    temp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    temp.replace(NOTES/name)


def main():
    NOTES.mkdir(exist_ok=True)
    roots=[]
    for name in MODELS:
        root=BASE/name
        assert root.is_dir() and not root.is_symlink()
        m=json.loads((root/'MANIFEST.json').read_text())
        c=json.loads((root/'ASSEMBLY_COMPLETE.json').read_text())
        ledger={line.split(None,1)[1].strip():line.split(None,1)[0] for line in (root/'SHA256SUMS').read_text().splitlines()}
        assert c['passed'] and c['indexed_tensors']==701633
        assert sha(root/'MANIFEST.json')==c['manifest_sha256']
        if 'MANIFEST.json' in ledger:
            assert ledger['MANIFEST.json']==c['manifest_sha256']
        assert sha(root/'SHA256SUMS')==c['sha256sums_sha256']
        assert set(ledger) in (set(m['files']),set(m['files'])|{'MANIFEST.json'})
        assert not any(p.is_symlink() for p in root.rglob('*'))
        assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(m['files'])|{'MANIFEST.json','SHA256SUMS','ASSEMBLY_COMPLETE.json'}
        for file,row in m['files'].items():
            path=root/file
            assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes']
            assert ledger[file]==row['sha256']
        roots.append(dict(path=str(root),files=len(m['files']),bytes=m['file_bytes'],manifest_sha256=c['manifest_sha256'],ledger=ledger))
    write('PRESERVED.json',dict(phase='metadata_verified_full_hash_in_progress',models=[{k:v for k,v in r.items() if k!='ledger'} for r in roots]))
    unique={}
    done=0
    for row in roots:
        root=Path(row['path'])
        for name,expected in sorted(row['ledger'].items()):
            path=root/name
            s=path.stat()
            key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            if key in unique:
                assert unique[key]['expected']==expected, 'shared inode has conflicting expected digests'
                unique[key]['aliases']+=1
            else:
                unique[key]=dict(path=path,expected=expected,aliases=1)
    def check(item):
        assert sha(item['path'])==item['expected'], str(item['path'])
        return item
    write('HASH_PROGRESS.json',dict(files=0,unique_files=len(unique),workers=4,phase='parallel_full_recheck'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count,item in enumerate(pool.map(check,unique.values()),1):
            done+=item['aliases']
            if count%100==0:
                write('HASH_PROGRESS.json',dict(files=done,unique_verified=count,unique_total=len(unique),last_file=str(item['path']),workers=4,updated_at=datetime.now(timezone.utc).isoformat()))
    print('HASH_VERIFIED_BOTH',done,flush=True)
    write('PRESERVED.json',dict(phase='complete',passed=True,verified_files=done,completed_at=datetime.now(timezone.utc).isoformat(),models=[{k:v for k,v in r.items() if k!='ledger'} for r in roots]))


if __name__=='__main__':
    main()
