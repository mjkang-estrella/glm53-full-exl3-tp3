#!/usr/bin/env python3
"""Zima tmux parent, outside the upload cgroup, records external/OOM exits."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path('/home/mj-kang/Dev/state/glm53-full-exl3-tp3/hf-public-20260908')
UNIT = 'glm53-hf-upload-recovery-20260909'
SCRIPT = Path(__file__).with_name('run_upload.sh')


def save(path, data):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.supervisor-')
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(name, path)


def scope():
    result = subprocess.run(['systemctl', '--user', 'show', UNIT + '.scope',
        '-p', 'ActiveState', '-p', 'Result', '-p', 'MemoryCurrent', '-p', 'MemorySwapCurrent'],
        capture_output=True, text=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ROOT / 'supervisor.log').open('a') as log:
            child = subprocess.Popen(['systemd-run', '--user', '--scope', '--unit=' + UNIT,
                '-p', 'MemoryMax=4G', '-p', 'MemorySwapMax=0', '-p', 'CPUQuota=200%',
                'bash', str(SCRIPT)], stdout=log, stderr=subprocess.STDOUT)
            while True:
                code = child.poll()
                live = dict(updated_at=datetime.now(timezone.utc).isoformat(),
                            supervisor_pid=os.getpid(), child_exit_code=code, scope=scope())
                save(ROOT / 'PROCESS.json', live)
                if code is not None:
                    path = ROOT / 'STATUS.json'
                    state = json.loads(path.read_text()) if path.exists() else {}
                    if code != 0 or state.get('phase') != 'complete':
                        state.update(previous_phase=state.get('phase'), phase='failed',
                                     process_exit_code=code, scope_result=live['scope'].get('Result'),
                                     updated_at=live['updated_at'])
                        save(path, state)
                    return code
                time.sleep(15)


if __name__ == '__main__':
    raise SystemExit(main())
