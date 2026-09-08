#!/usr/bin/env python3
"""Targeted pre-publication scan of every reachable commit, without printing secrets.

This is a supplemental pattern/artifact audit, not a guarantee that source has
no sensitive information. Review findings and publication scope before pushing.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private_key': rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----',
    'github_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
    'huggingface_token': rb'\bhf_[A-Za-z0-9]{25,}\b',
    'openai_token': rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}\b',
    'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'slack_token': rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b',
    'jwt_literal': rb'\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{20,}\b',
    'url_credentials': rb'https?://[^\s/:@]{2,}:[^\s/@]{8,}@',
}
FORBIDDEN = re.compile(r'(^|/)(?:\.env(?:\..*)?|zima-ssh-config|id_rsa.*|id_ed25519.*)$|'
                       r'(^|/)(?:outputs|state|logs|cache|models|encoder-r10)/|'
                       r'\.(?:safetensors|jsonl|csv|log|pem|key)$')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    commits = git('rev-list', '--all').decode().splitlines()
    seen, findings = set(), []
    for commit in commits:
        for entry in git('ls-tree', '-rz', commit).split(b'\0'):
            if not entry:
                continue
            metadata, name = entry.split(b'\t', 1)
            mode, kind, oid = metadata.decode().split()
            name = name.decode()
            if FORBIDDEN.search(name) and not name.endswith('.example'):
                findings.append({'commit': commit[:12], 'file': name, 'kind': 'excluded_artifact'})
            if kind != 'blob' or oid in seen:
                continue
            seen.add(oid)
            body = git('cat-file', 'blob', oid)
            if len(body) > 5 * 1024 * 1024 or b'\0' in body:
                findings.append({'commit': commit[:12], 'file': name, 'kind': 'binary_or_large_blob'})
            for label, pattern in PATTERNS.items():
                if re.search(pattern, body):
                    findings.append({'commit': commit[:12], 'file': name, 'kind': label})
    print(json.dumps({'commits': len(commits), 'unique_blobs': len(seen), 'findings': findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()
