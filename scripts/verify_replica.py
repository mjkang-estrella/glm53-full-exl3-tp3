#!/usr/bin/env python3
"""Checksum verification for a local-NVMe copy of the sealed checkpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import GEOMETRY_ID
from tp3k3.safetensors_stream import sha256_file


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_ledger(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        if name in rows:
            raise ValueError(f"duplicate checksum row: {name}")
        rows[name] = digest
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.checkpoint
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "ASSEMBLY_COMPLETE.json").read_text(encoding="utf-8"))
    ledger = parse_ledger(root / "SHA256SUMS")
    expected = set(manifest["files"]) | {"MANIFEST.json"}
    errors: list[str] = []
    if set(ledger) != expected:
        errors.append("checksum ledger file set differs from manifest")
    if manifest.get("geometry_id") != GEOMETRY_ID or complete.get("geometry_id") != GEOMETRY_ID:
        errors.append("geometry id differs")
    if complete.get("passed") is not True or complete.get("indexed_tensors") != 701633:
        errors.append("assembly completion contract differs")
    if complete.get("manifest_sha256") != sha256_file(root / "MANIFEST.json"):
        errors.append("assembly marker manifest hash differs")
    if complete.get("sha256sums_sha256") != sha256_file(root / "SHA256SUMS"):
        errors.append("assembly marker checksum-ledger hash differs")

    actual_files = {path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()}
    expected_files = expected | {"SHA256SUMS", "ASSEMBLY_COMPLETE.json"}
    if actual_files != expected_files:
        errors.append(
            f"replica file set differs missing={sorted(expected_files-actual_files)[:5]} "
            f"extra={sorted(actual_files-expected_files)[:5]}"
        )
    if any(path.is_symlink() for path in root.iterdir()):
        errors.append("replica contains symbolic links")

    def verify(item: tuple[str, str]) -> str | None:
        name, expected_digest = item
        path = root / name
        record = manifest["files"].get(name)
        if not path.is_file():
            return name
        if record is not None and path.stat().st_size != int(record["bytes"]):
            return name
        return None if sha256_file(path) == expected_digest else name

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        failures = [name for name in pool.map(verify, sorted(ledger.items())) if name]
    if failures:
        errors.append(f"replica checksum failures: {failures[:10]}")
    result = {
        "schema": "glm53-full-exl3-tp3.replica-verification.v1",
        "host": socket.gethostname(),
        "checkpoint": str(root),
        "geometry_id": GEOMETRY_ID,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "files": len(expected_files),
        "bytes": sum(path.stat().st_size for path in root.iterdir() if path.is_file()),
        "assembly_complete_sha256": sha256_file(root / "ASSEMBLY_COMPLETE.json"),
        "manifest_sha256": sha256_file(root / "MANIFEST.json"),
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
        "checksum_failures": failures,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(args.output, result)
    if errors:
        raise SystemExit("replica verification failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
