#!/usr/bin/env python3
"""Verify the bounded four-window BF16 teacher subset against its Hub manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_window = {row["window_id"]: row for row in manifest["logit_files"]}
    windows = [f"confirmation-{index:04d}" for index in range(4)]

    def verify(window_id: str) -> dict:
        record = by_window[window_id]
        logit_path = args.root / record["path"]
        token_path = args.root / f"reference-full-panel/calibration/panel-v1/arrays/{window_id}.tokens.npy"
        errors = []
        if not logit_path.is_file() or logit_path.stat().st_size != record["bytes"]:
            errors.append("logit file size differs")
        logit_sha = sha256_file(logit_path) if logit_path.is_file() else None
        token_sha = sha256_file(token_path) if token_path.is_file() else None
        if logit_sha != record["sha256"]:
            errors.append("logit SHA-256 differs")
        if token_sha != record["token_ids_sha256"]:
            errors.append("token SHA-256 differs")
        if logit_path.is_file():
            with logit_path.open("rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            if header.get("logits", {}).get("shape") != [2047, 154880] or header.get("logits", {}).get("dtype") != "F32":
                errors.append("teacher tensor geometry differs")
            metadata = header.get("__metadata__", {})
            if metadata.get("model_revision") != manifest["model_revision"] or metadata.get("window_id") != window_id:
                errors.append("teacher metadata identity differs")
        return {"window_id": window_id, "logit": str(logit_path), "logit_sha256": logit_sha, "tokens": str(token_path), "token_sha256": token_sha, "errors": errors, "passed": not errors}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(verify, windows))
    result = {
        "schema": "glm53-full-exl3-tp3.teacher-subset-verification.v1",
        "passed": all(row["passed"] for row in rows),
        "reference_repo": manifest["repo_id"],
        "reference_revision": "427368f12a4bdc21668bc4171ce0dc54f8990200",
        "bf16_model_revision": manifest["model_revision"],
        "reference_scope": manifest["scope"],
        "windows": rows,
        "prediction_positions": 4 * 2047,
        "vocab_size": 154880,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("teacher subset verification failed")


if __name__ == "__main__":
    main()
