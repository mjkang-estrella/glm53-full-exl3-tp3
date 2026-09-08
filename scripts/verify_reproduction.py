#!/usr/bin/env python3
"""Mac or Zima: offline source/results audit; no SSH, GPU, or model writes."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
    ).decode().split('\0')
    checked = 0
    for name in sorted(set(files) - {''}):
        path = ROOT / name
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
            checked += 1
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
            checked += 1
    summary = json.loads((ROOT / 'results/final-summary.json').read_text())
    results = json.loads((ROOT / 'results/speed-results.json').read_text())
    eligible = {r['label']: r for r in results['rows']
                if r['accepted'] and r.get('eligible_for_selection', True)}
    values = eligible['kv1g']['tg']['values'] + eligible['qualified-final']['tg']['values']
    assert values == summary['decode_values'] and len(values) == 6
    assert abs(statistics.mean(values) - summary['decode_mean']) < 1e-10
    assert abs(statistics.pstdev(values) - summary['decode_std']) < 1e-10
    assert abs(summary['decode_mean'] / summary['baseline_decode_mean'] - summary['speedup']) < 1e-10
    assert 'prefill256-kv1g' not in eligible and 'spinwait16' not in eligible
    assert summary['long_context']['usage']['prompt_tokens'] == 30039
    assert summary['long_context']['passed'] and summary['kernel_audit']['passed']
    for row in summary['memory']['ranks'].values():
        assert row['min_host_available_bytes'] >= 12 * 2**30
        assert row['max_cgroup_swap_bytes'] == 0
    for test in ('test_speed_safety.py', 'test_reencode_safety.py'):
        subprocess.run([sys.executable, str(ROOT / 'scripts' / test)], check=True)
    ledger = ROOT / 'dependencies/encoder-deployed.sha256'
    if (ROOT / 'encoder-r10').exists():
        for line in ledger.read_text().splitlines():
            digest, name = line.split('  ', 1)
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        print('Deployed external encoder hashes: PASS')
    else:
        print('External encoder absent: serving reproduction can use sealed weights; encoding is blocked.')
    print(f'PASS: {checked} Python/shell syntax checks, safety regressions, and curated result invariants')


if __name__ == '__main__':
    main()
