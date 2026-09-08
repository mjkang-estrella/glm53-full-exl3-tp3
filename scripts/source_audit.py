#!/usr/bin/env python3
"""Revalidate the immutable GLM-5.3 BF16 source without loading tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any


PINNED = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
EXPECTED_SHARDS = 282
EXPECTED_SHARD_FILE_BYTES = 1_506_667_387_408
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while data := handle.read(chunk):
            h.update(data)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, sort_keys=True, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def metadata(checkpoint: Path, filename: str) -> tuple[str, str]:
    path = checkpoint / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"invalid Hugging Face metadata: {path}")
    return lines[0], lines[1]


def structure(checkpoint: Path, expected_path: Path, output: Path) -> None:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    index_path = checkpoint / "model.safetensors.index.json"
    config_path = checkpoint / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    errors: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    actual_names: set[str] = set()
    tensor_bytes = 0
    for name in shard_names:
        path = checkpoint / name
        revision, oid = metadata(checkpoint, name)
        if revision != PINNED:
            errors.append(f"{name}: metadata revision {revision}")
        exp = expected.get("shards", {}).get(name, {})
        size = path.stat().st_size
        if size != exp.get("size"):
            errors.append(f"{name}: size {size} != {exp.get('size')}")
        if oid != exp.get("file_sha256"):
            errors.append(f"{name}: LFS oid {oid} != expected file sha256")
        with path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"truncated safetensors header length: {name}")
            header_len = struct.unpack("<Q", raw_len)[0]
            header_raw = handle.read(header_len)
        header_sha = hashlib.sha256(header_raw).hexdigest()
        if header_sha != exp.get("header_sha256"):
            errors.append(f"{name}: header sha256 mismatch")
        header = json.loads(header_raw)
        names = sorted(key for key in header if key != "__metadata__")
        if len(names) != exp.get("tensor_count"):
            errors.append(f"{name}: tensor count {len(names)} != {exp.get('tensor_count')}")
        payload_base = 8 + header_len
        max_end = 0
        for tensor_name in names:
            if tensor_name in actual_names:
                errors.append(f"duplicate tensor: {tensor_name}")
            actual_names.add(tensor_name)
            record = header[tensor_name]
            start, end = map(int, record["data_offsets"])
            shape = tuple(map(int, record["shape"]))
            elements = 1
            for dim in shape:
                elements *= dim
            expected_nbytes = elements * DTYPE_BYTES[record["dtype"]]
            if end - start != expected_nbytes:
                errors.append(f"{tensor_name}: offset byte count differs from dtype/shape")
            if weight_map.get(tensor_name) != name:
                errors.append(f"{tensor_name}: index maps to {weight_map.get(tensor_name)!r}, actual {name}")
            source_record = expected.get("entries", {}).get(tensor_name)
            if source_record and (
                source_record.get("shard") != name
                or source_record.get("dtype") != record["dtype"]
                or tuple(source_record.get("shape", ())) != shape
                or source_record.get("nbytes") != expected_nbytes
            ):
                errors.append(f"{tensor_name}: header differs from sealed source inventory")
            tensor_bytes += expected_nbytes
            max_end = max(max_end, end)
        if payload_base + max_end != size:
            errors.append(f"{name}: payload end {payload_base + max_end} != file size {size}")
        rows[name] = {
            "size": size,
            "lfs_sha256": oid,
            "header_sha256": header_sha,
            "tensor_count": len(names),
            "revision": revision,
        }
    indexed_names = set(weight_map)
    missing = sorted(indexed_names - actual_names)
    unindexed = sorted(actual_names - indexed_names)
    if missing:
        errors.append(f"{len(missing)} indexed tensors missing")
    if unindexed:
        errors.append(f"{len(unindexed)} tensors absent from index")
    if len(shard_names) != EXPECTED_SHARDS:
        errors.append(f"shard count {len(shard_names)} != {EXPECTED_SHARDS}")
    shard_file_bytes = sum(row["size"] for row in rows.values())
    if shard_file_bytes != EXPECTED_SHARD_FILE_BYTES:
        errors.append(f"shard file bytes {shard_file_bytes} != {EXPECTED_SHARD_FILE_BYTES}")
    if tensor_bytes != int(index.get("metadata", {}).get("total_size", -1)):
        errors.append("tensor payload bytes differ from index metadata total_size")
    config_sha = sha256_file(config_path)
    index_sha = sha256_file(index_path)
    if config_sha != expected.get("config_sha256"):
        errors.append("config sha256 differs")
    if index_sha != expected.get("index_sha256"):
        errors.append("index sha256 differs")
    aux_names = [".gitattributes", "LICENSE", "README.md", "chat_template.jinja", "config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
    auxiliary = {}
    for name in aux_names:
        path = checkpoint / name
        revision, metadata_oid = metadata(checkpoint, name)
        digest = sha256_file(path)
        auxiliary[name] = {"bytes": path.stat().st_size, "sha256": digest, "revision": revision, "metadata_oid": metadata_oid}
        if revision != PINNED:
            errors.append(f"{name}: metadata revision differs")
        sealed = expected.get("auxiliary_files_sha256", {}).get(name)
        if sealed and digest != sealed:
            errors.append(f"{name}: sha256 differs from sealed inventory")
    result = {
        "schema": "glm53-full-exl3-tp3.source-structure.v1",
        "checkpoint": str(checkpoint),
        "pinned_revision": PINNED,
        "expected_inventory": str(expected_path),
        "expected_inventory_sha256": sha256_file(expected_path),
        "shard_count": len(shard_names),
        "tensor_count": len(actual_names),
        "weight_bytes": tensor_bytes,
        "shard_file_bytes": shard_file_bytes,
        "index_total_size": int(index.get("metadata", {}).get("total_size", -1)),
        "missing_indexed_tensors": missing,
        "unindexed_tensors": unindexed,
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "auxiliary": auxiliary,
        "shards": rows,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(output, result)
    if errors:
        raise SystemExit("source structural audit failed; see " + str(output))


def hash_shards(checkpoint: Path, expected_path: Path, state_dir: Path, output: Path, *, reverse: bool = False) -> None:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    parts = state_dir / "source-sha256-parts"
    parts.mkdir(parents=True, exist_ok=True)
    failures = []
    rows = []
    for name, exp in sorted(expected["shards"].items(), reverse=reverse):
        part = parts / f"{name}.json"
        path = checkpoint / name
        stat = path.stat()
        row = None
        if part.exists():
            candidate = json.loads(part.read_text(encoding="utf-8"))
            if candidate.get("size") == stat.st_size and candidate.get("mtime_ns") == stat.st_mtime_ns:
                row = candidate
        if row is None:
            digest = sha256_file(path)
            row = {"file": name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest, "expected_sha256": exp["file_sha256"], "passed": digest == exp["file_sha256"]}
            atomic_json(part, row)
        rows.append(row)
        if not row["passed"]:
            failures.append(name)
    result = {
        "schema": "glm53-full-exl3-tp3.source-sha256.v1",
        "checkpoint": str(checkpoint),
        "pinned_revision": PINNED,
        "shard_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "failures": failures,
        "passed": not failures and len(rows) == EXPECTED_SHARDS,
        "files": rows,
    }
    atomic_json(output, result)
    if not result["passed"]:
        raise SystemExit("source content hash audit failed; see " + str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("structure", "hash-shards"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-inventory", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true", help="replay shards from the end for a cooperating worker")
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "structure":
        structure(args.checkpoint.resolve(), args.expected_inventory.resolve(), args.state_dir / "source-structure.json")
    else:
        hash_shards(
            args.checkpoint.resolve(),
            args.expected_inventory.resolve(),
            args.state_dir,
            args.state_dir / "source-sha256.json",
            reverse=args.reverse,
        )


if __name__ == "__main__":
    main()
