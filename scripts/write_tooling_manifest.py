#!/usr/bin/env python3
"""Seal the exact non-git project code closure used for a real-test attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = ("runtime", "scripts", "tests", "tp3k3")
    files = []
    for root_name in roots:
        for path in sorted((args.project / root_name).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = str(path.relative_to(args.project))
            files.append({"path": relative, "bytes": path.stat().st_size, "mode": oct(path.stat().st_mode & 0o777), "sha256": sha256_file(path)})
    for name in ("AGENTS.md", "PLAN.md", "README.md", "RECOVERY.md", "RESULTS.md", "STATUS.md", "TASK_PROMPT.md"):
        path = args.project / name
        files.append({"path": name, "bytes": path.stat().st_size, "mode": oct(path.stat().st_mode & 0o777), "sha256": sha256_file(path)})
    result = {
        "schema": "glm53-full-exl3-tp3.tooling-manifest.v1",
        "project": str(args.project),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "files": files,
        "file_count": len(files),
    }
    encoded = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
