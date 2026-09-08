#!/usr/bin/env python3
"""Atomically promote an independently verified checkpoint directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import GEOMETRY_ID
from tp3k3.safetensors_stream import sha256_file


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.final.exists():
        raise SystemExit(f"final checkpoint already exists: {args.final}")
    verification_raw = args.verification.read_bytes()
    verification = json.loads(verification_raw)
    if not (
        verification.get("passed") is True
        and verification.get("geometry_id") == GEOMETRY_ID
        and Path(verification.get("checkpoint", "")) == args.incoming
        and verification.get("full_file_checksums_verified") is True
        and verification.get("indexed_tensors") == 701633
    ):
        raise SystemExit("independent checkpoint verification gate failed")
    complete_path = args.incoming / "ASSEMBLY_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.update({
        "output_dir": str(args.final),
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "checkpoint_verification_sha256": sha256_file(args.verification),
    })
    atomic_json(complete_path, complete)
    complete_sha = sha256_file(complete_path)
    os.replace(args.incoming, args.final)
    receipt = {
        "schema": "glm53-full-exl3-tp3.checkpoint-seal.v1",
        "passed": True,
        "geometry_id": GEOMETRY_ID,
        "checkpoint": str(args.final),
        "sealed_at": complete["sealed_at"],
        "verification": str(args.verification),
        "verification_sha256": sha256_file(args.verification),
        "assembly_complete_sha256": complete_sha,
        "manifest_sha256": sha256_file(args.final / "MANIFEST.json"),
        "sha256sums_sha256": sha256_file(args.final / "SHA256SUMS"),
    }
    atomic_json(args.receipt, receipt)


if __name__ == "__main__":
    main()
