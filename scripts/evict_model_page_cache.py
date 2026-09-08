#!/usr/bin/env python3
"""Evict clean checkpoint pages without reading or modifying model files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def meminfo() -> dict[str, int]:
    wanted = {"MemFree", "MemAvailable", "Cached", "SReclaimable", "SwapFree"}
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        if key in wanted:
            result[f"{key.lower()}_bytes"] = int(value.split()[0]) * 1024
    return result


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


def advise(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--rank-pack-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        parser.error("POSIX_FADV_DONTNEED is unavailable")
    checkpoint = args.checkpoint_dir.resolve(strict=True)
    if not checkpoint.is_dir():
        parser.error("checkpoint path is not a directory")
    files = sorted(checkpoint.glob("bf16-passthrough-*.safetensors"))
    checkpoint_count = len(files)
    if checkpoint_count == 0:
        parser.error("no BF16 passthrough shards found")
    rank_pack_count = 0
    if args.rank_pack_dir is not None:
        rank_pack = args.rank_pack_dir.resolve(strict=True)
        if not rank_pack.is_dir() or not (rank_pack / "COMPLETE.json").is_file():
            parser.error("rank-pack directory is unsealed")
        packs = sorted(rank_pack.glob("k3-layer-*-rank*.safetensors"))
        if not packs:
            parser.error("no rank-pack files found")
        rank_pack_count = len(packs)
        files.extend(packs)
    before = meminfo()
    errors = []
    advised_bytes = 0
    for path in files:
        try:
            size = path.stat().st_size
            advise(path)
            advised_bytes += size
        except OSError as error:
            errors.append({"path": str(path), "error": repr(error)})
    result = {
        "schema": "glm53-full-exl3-tp3.page-cache-eviction.v1",
        "created_at": utc(),
        "passed": not errors,
        "method": "posix_fadvise-POSIX_FADV_DONTNEED",
        "checkpoint": str(checkpoint),
        "checkpoint_files": checkpoint_count,
        "rank_pack_files": rank_pack_count,
        "advised_files": len(files) - len(errors),
        "advised_bytes": advised_bytes,
        "before": before,
        "after": meminfo(),
        "errors": errors,
    }
    atomic_json(args.output, result)
    if errors:
        raise SystemExit("one or more page-cache eviction hints failed")


if __name__ == "__main__":
    main()
