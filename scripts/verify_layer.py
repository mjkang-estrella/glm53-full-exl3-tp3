#!/usr/bin/env python3
"""Verify a sealed K3 TP3 layer and every expert artifact/receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import (
    GEOMETRY_ID,
    layer_geometry,
    rank_offsets,
    rank_widths,
    slice_spec,
)


DTYPE_BYTES = {"I16": 2, "F16": 2, "I32": 4}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def expected_shape(layer: int, rank: int, projection: str, suffix: str) -> tuple[int, ...]:
    width = rank_widths(layer)[rank]
    if suffix == "trellis":
        return (width // 16, 384, 48) if projection == "down_proj" else (384, width // 16, 48)
    if suffix == "suh":
        return (width,) if projection == "down_proj" else (6144,)
    if suffix == "svh":
        return (6144,) if projection == "down_proj" else (width,)
    if suffix == "mcg":
        return (1,)
    raise AssertionError(suffix)


def expected_dtype(suffix: str) -> str:
    return {"trellis": "I16", "suh": "F16", "svh": "F16", "mcg": "I32"}[suffix]


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        raw = handle.read(length)
        if len(raw) != length:
            raise ValueError(f"truncated safetensors header: {path}")
    return 8 + length, json.loads(raw)


def payload_sha256(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            block = handle.read(min(16 << 20, remaining))
            if not block:
                raise IOError(f"short payload read: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def verify_expert(layer_dir: Path, layer: int, expert: int, layer_entry: dict) -> dict:
    shard = layer_dir / "experts" / f"expert-{expert:03d}.safetensors"
    receipt_path = layer_dir / "receipts" / f"expert-{expert:03d}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "glm53-full-exl3-tp3.expert-receipt.v2"
        or receipt.get("geometry_id") != GEOMETRY_ID
        or receipt.get("layer") != layer
        or receipt.get("expert") != expert
        or receipt.get("layer_geometry") != layer_geometry(layer)
        or receipt.get("padding_channels") != 0
        or not receipt.get("passed")
    ):
        raise ValueError(f"invalid expert receipt {expert}")
    digest = sha256_file(shard)
    if digest != receipt["output_sha256"] or digest != layer_entry["sha256"]:
        raise ValueError(f"expert {expert} artifact SHA-256 mismatch")
    if shard.stat().st_size != receipt["output_bytes"] or shard.stat().st_size != layer_entry["bytes"]:
        raise ValueError(f"expert {expert} artifact size mismatch")
    records = {
        (row["slice"]["projection"], int(row["slice"]["rank"])): row
        for row in receipt["slices"]
    }
    if set(records) != {(projection, rank) for projection in ("gate_proj", "up_proj", "down_proj") for rank in range(3)}:
        raise ValueError(f"expert {expert} slice set differs")
    data_start, header = read_header(shard)
    metadata = header.pop("__metadata__", {})
    if (
        metadata.get("schema") != "glm53-full-exl3-tp3.expert-shard.v2"
        or metadata.get("geometry_id") != GEOMETRY_ID
        or metadata.get("layer") != str(layer)
        or metadata.get("expert") != str(expert)
        or metadata.get("bits") != "3"
        or metadata.get("tp") != "3"
        or metadata.get("rank_widths") != ",".join(map(str, rank_widths(layer)))
        or metadata.get("rank_offsets") != ",".join(map(str, rank_offsets(layer)))
    ):
        raise ValueError(f"expert {expert} safetensors metadata differs")
    expected_names = set()
    cursor = 0
    for projection, rank in records:
        record = records[(projection, rank)]
        expected_spec = slice_spec(layer, expert, projection, rank)
        recorded_spec = record["slice"]
        if (
            recorded_spec.get("geometry_id") != GEOMETRY_ID
            or recorded_spec.get("semantic_start") != expected_spec.semantic_start
            or recorded_spec.get("semantic_stop") != expected_spec.semantic_stop
            or recorded_spec.get("physical_channels") != expected_spec.physical_channels
            or recorded_spec.get("semantic_channels") != expected_spec.semantic_channels
            or recorded_spec.get("pad_channels") != 0
            or recorded_spec.get("scale_mask") is not None
            or not record["padding_exact_zero"]
            or float(record.get("padding_max_abs", -1)) != 0.0
            or not all(math.isfinite(float(record[name])) for name in ("error_squared", "source_squared", "relative_squared_error", "relative_rmse", "max_abs_error"))
        ):
            raise ValueError(f"expert {expert} numeric receipt differs")
        for suffix in ("trellis", "suh", "svh", "mcg"):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}.{suffix}"
            expected_names.add(name)
            item = header[name]
            dtype = expected_dtype(suffix)
            shape = expected_shape(layer, rank, projection, suffix)
            if item["dtype"] != dtype or tuple(item["shape"]) != shape:
                raise ValueError(f"{name} dtype/shape differs")
            start, end = map(int, item["data_offsets"])
            nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
            if start != cursor or end - start != nbytes:
                raise ValueError(f"{name} payload range differs")
            component = record["components"][suffix]
            if component["bytes"] != nbytes or payload_sha256(shard, data_start + start, nbytes) != component["sha256"]:
                raise ValueError(f"{name} component hash differs")
            cursor = end
    if set(header) != expected_names or data_start + cursor != shard.stat().st_size:
        raise ValueError(f"expert {expert} tensor set or file boundary differs")
    return {"expert": expert, "bytes": shard.stat().st_size, "sha256": digest, "relative_rmse": receipt["relative_rmse"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layer_receipt_path = args.layer_dir / "LAYER_RECEIPT.json"
    layer_receipt = json.loads(layer_receipt_path.read_text(encoding="utf-8"))
    layer = int(layer_receipt["layer"])
    if (
        layer_receipt.get("schema") != "glm53-full-exl3-tp3.layer-receipt.v2"
        or layer_receipt.get("geometry_id") != GEOMETRY_ID
        or layer_receipt.get("layer_geometry") != layer_geometry(layer)
        or layer_receipt.get("experts") != 256
        or layer_receipt.get("slice_encodes") != 2304
        or not layer_receipt.get("passed")
    ):
        raise SystemExit("invalid layer receipt")
    entries = {int(row["expert"]): row for row in layer_receipt["expert_artifacts"]}
    if set(entries) != set(range(256)):
        raise SystemExit("layer expert artifact set differs")
    verified = [verify_expert(args.layer_dir, layer, expert, entries[expert]) for expert in range(256)]
    result = {
        "schema": "glm53-full-exl3-tp3.layer-verification.v2",
        "geometry_id": GEOMETRY_ID,
        "layer_geometry": layer_geometry(layer),
        "layer": layer,
        "experts": len(verified),
        "tensors": len(verified) * 36,
        "slice_encodes": len(verified) * 9,
        "bytes": sum(row["bytes"] for row in verified),
        "layer_receipt_sha256": sha256_file(layer_receipt_path),
        "worst_relative_rmse": max(verified, key=lambda row: (row["relative_rmse"], -row["expert"])),
        "passed": True,
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
