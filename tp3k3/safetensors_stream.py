"""Bounded-memory helpers for byte-preserving safetensors assembly."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import BinaryIO, Iterable


CHUNK_BYTES = 16 << 20


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        raw = handle.read(header_bytes)
        if len(raw) != header_bytes:
            raise ValueError(f"truncated safetensors header: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return 8 + header_bytes, value


def encoded_header(entries: dict[str, dict], metadata: dict[str, str] | None = None) -> bytes:
    value: dict[str, object] = {}
    if metadata:
        value["__metadata__"] = {str(k): str(v) for k, v in sorted(metadata.items())}
    value.update(entries)
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def sha256_range(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            block = handle.read(min(CHUNK_BYTES, remaining))
            if not block:
                raise OSError(f"short read from {path} at {offset + length - remaining}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _copy_range(source: BinaryIO, target: BinaryIO, length: int, digest: hashlib._Hash) -> None:
    remaining = length
    while remaining:
        block = source.read(min(CHUNK_BYTES, remaining))
        if not block:
            raise OSError("short source read while copying safetensors payload")
        target.write(block)
        digest.update(block)
        remaining -= len(block)


def write_subset(
    source_path: Path,
    target_path: Path,
    tensors: Iterable[dict],
    *,
    metadata: dict[str, str],
) -> dict:
    """Write selected source payload ranges without deserializing tensors.

    ``payload_start`` and ``payload_end`` are absolute source-file offsets.
    The returned tensor hashes prove that the new payloads are byte-identical.
    """

    selected = sorted(tensors, key=lambda row: (int(row["payload_start"]), row["name"]))
    if not selected:
        raise ValueError("cannot create an empty safetensors subset")
    entries: dict[str, dict] = {}
    cursor = 0
    for row in selected:
        length = int(row["payload_end"]) - int(row["payload_start"])
        if length != int(row["nbytes"]) or length < 0:
            raise ValueError(f"invalid source range for {row['name']}")
        entries[row["name"]] = {
            "dtype": row["dtype"],
            "shape": row["shape"],
            "data_offsets": [cursor, cursor + length],
        }
        cursor += length
    prefix = encoded_header(entries, metadata)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target_path.name}.", suffix=".partial", dir=target_path.parent)
    tensor_hashes: dict[str, str] = {}
    whole = hashlib.sha256()
    try:
        with source_path.open("rb") as source, os.fdopen(descriptor, "wb") as target:
            target.write(prefix)
            whole.update(prefix)
            for row in selected:
                source.seek(int(row["payload_start"]))
                digest = hashlib.sha256()
                _copy_range(source, target, int(row["nbytes"]), digest)
                tensor_hashes[row["name"]] = digest.hexdigest()
            target.flush()
            os.fsync(target.fileno())
        # Hashing only the prefix above is insufficient once multiple tensor
        # digests have consumed the payload stream. Read the bounded output.
        output_hash = sha256_file(Path(temporary))
        os.replace(temporary, target_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "bytes": target_path.stat().st_size,
        "sha256": output_hash,
        "tensor_count": len(selected),
        "payload_bytes": cursor,
        "tensor_sha256": tensor_hashes,
    }


def atomic_copy_verified(source: Path, target: Path, expected_sha256: str | None = None) -> dict:
    """Stream-copy one file, atomically publishing it only after hashing."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
            while block := src.read(CHUNK_BYTES):
                dst.write(block)
                digest.update(block)
            dst.flush()
            os.fsync(dst.fileno())
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {source}: {actual} != {expected_sha256}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"bytes": target.stat().st_size, "sha256": digest.hexdigest()}

