#!/usr/bin/env python3
"""Zima: remove explicitly abandoned raw captures and stopped v1 containers.

Keep detailed metadata and logs. Invalid raw captures are intentionally not
recoverable; no valid checkpoint or latest comparison capture is deleted.
"""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

PROJECT=Path('/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3')
BASE=Path('/mnt/unas-models/ZAI')
NOTES=BASE/'GLM-5.3-EXL3-archive-notes-20260908'
TARGETS=[
 'GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-kld-evidence-prefill256-20260907T224500Z',
 'GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2K3-kld-evidence-prefill256-retry2-20260907T224500Z']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()


def ssh(node,args):
    return subprocess.run(['ssh','-F',str(PROJECT/'zima-ssh-config'),node,shlex.join(args)],
                          capture_output=True,check=True,timeout=120).stdout


def main():
    assert (NOTES/'PRESERVED.json').is_file()
    assert (NOTES/'ARCHIVE_AND_REENCODE.md').is_file()
    inventory=[]
    for name in TARGETS:
        root=BASE/name
        assert root.is_dir() and not root.is_symlink()
        failures=list(root.rglob('TRANSFER_FAILED.json'))
        assert len(failures)==1 and json.loads(failures[0].read_text())['errors']
        for path in sorted(root.rglob('*.f32')):
            assert not path.is_symlink() and re.fullmatch(r'part-\d{4}\.f32',path.name)
            assert root in path.resolve().parents
            inventory.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
    out=NOTES/'INVALID_RAW_DELETION_INVENTORY.json'
    assert not out.exists(), 'inspect previous cleanup before retrying'
    out.write_text(json.dumps(dict(files=inventory,reason='failed row-order/row-count validation',
        recoverable=False,metadata_retained=True),indent=2)+'\n')
    for row in inventory:
        path=Path(row['path'])
        assert path.is_file() and path.stat().st_size==row['bytes']
        path.unlink()
    deleted=[]
    for rank in range(3):
        node=f'mj-spark-{rank+1}'
        name=f'glm53-k3-cand-k275-quality-v1-rank{rank}'
        info=ssh(node,['docker','inspect',name])
        record=json.loads(info)[0]
        assert record['Name']=='/'+name and record['State']['Running'] is False
        folder=NOTES/f'failed-quality-v1-rank{rank}'
        folder.mkdir(exist_ok=False)
        (folder/'container-inspect.json').write_bytes(info)
        # Docker emits some log records on stderr; retain both streams.
        log=subprocess.run(['ssh','-F',str(PROJECT/'zima-ssh-config'),node,
                            shlex.join(['docker','logs',name])],capture_output=True,check=True,timeout=120)
        (folder/'container-stdout.log').write_bytes(log.stdout)
        (folder/'container-stderr.log').write_bytes(log.stderr)
        ssh(node,['docker','rm',name])
        deleted.append(dict(node=node,container=name,configuration_and_logs=str(folder)))
    result=dict(invalid_raw_files=len(inventory),bytes_removed=sum(r['bytes'] for r in inventory),
                containers_removed=deleted,checkpoint_files_removed=0)
    (NOTES/'CLEANUP_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
