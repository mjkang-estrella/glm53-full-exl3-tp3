#!/usr/bin/env python3
"""Verify a resumable K2 expert subset for the mixed 2.75bpw candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import GEOMETRY_ID, layer_geometry, rank_widths

SOURCE_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
SUFFIXES = ("trellis", "suh", "svh", "mcg")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


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
                raise ValueError(f"short payload read: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def expected_shape(layer: int, rank: int, projection: str, suffix: str) -> tuple[int, ...]:
    width = rank_widths(layer)[rank]
    if suffix == "trellis":
        return (384, width // 16, 32) if projection != "down_proj" else (width // 16, 384, 32)
    if suffix == "suh":
        return (width,) if projection == "down_proj" else (6144,)
    if suffix == "svh":
        return (6144,) if projection == "down_proj" else (width,)
    if suffix == "mcg":
        return (1,)
    raise AssertionError(suffix)


def expected_dtype(suffix: str) -> str:
    return {"trellis": "I16", "suh": "F16", "svh": "F16", "mcg": "I32"}[suffix]


def verify_expert(layer_dir: Path, layer: int, expert: int) -> dict:
    shard = layer_dir / "experts" / f"expert-{expert:03d}.safetensors"
    receipt_path = layer_dir / "receipts" / f"expert-{expert:03d}.json"
    if not shard.is_file() or not receipt_path.is_file():
        raise ValueError("missing paired shard or receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = (
        receipt.get("schema") == "glm53-full-exl3-tp3.expert-receipt.v2",
        receipt.get("source_revision") == SOURCE_REVISION,
        receipt.get("geometry_id") == GEOMETRY_ID,
        receipt.get("layer") == layer,
        receipt.get("expert") == expert,
        receipt.get("bits") == 2,
        receipt.get("tp") == 3,
        receipt.get("padding_channels") == 0,
        receipt.get("layer_geometry") == layer_geometry(layer),
        receipt.get("passed") is True,
        shard.stat().st_size == receipt.get("output_bytes"),
        sha256_file(shard) == receipt.get("output_sha256"),
    )
    if not all(checks):
        raise ValueError("receipt identity, size, hash, or pass marker differs")
    data_start, header = read_header(shard)
    metadata = header.pop("__metadata__", {})
    if metadata != {
        "bits": "2",
        "expert": str(expert),
        "geometry_id": GEOMETRY_ID,
        "layer": str(layer),
        "rank_offsets": ",".join(
            map(str, (layer_geometry(layer)["ranks"][i]["offset"] for i in range(3)))
        ),
        "rank_widths": ",".join(map(str, rank_widths(layer))),
        "schema": "glm53-full-exl3-tp3.expert-shard.v2",
        "tp": "3",
    }:
        raise ValueError("safetensors metadata differs")
    records = {(int(row["slice"]["rank"]), row["slice"].get("projection")): row for row in receipt.get("slices", [])}
    # The K2 encoder's slice records carry projection/rank inside the slice spec.
    if len(receipt.get("slices", [])) != 9:
        raise ValueError("receipt does not contain 9 rank/projection slices")
    expected_names = set()
    cursor = 0
    for projection in PROJECTIONS:
        for rank in range(3):
            spec_record = next((row for row in receipt["slices"] if row["slice"]["projection"] == projection and int(row["slice"]["rank"]) == rank), None)
            if spec_record is None:
                raise ValueError(f"missing slice record {projection}/rank{rank}")
            if not spec_record.get("padding_exact_zero") or float(spec_record.get("padding_max_abs", -1)) != 0.0:
                raise ValueError(f"slice padding audit failed {projection}/rank{rank}")
            for suffix in SUFFIXES:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}.{suffix}"
                expected_names.add(name)
                item = header.get(name)
                if item is None:
                    raise ValueError(f"missing tensor {name}")
                shape = expected_shape(layer, rank, projection, suffix)
                dtype = expected_dtype(suffix)
                if item.get("dtype") != dtype or tuple(item.get("shape", ())) != shape:
                    raise ValueError(f"shape/dtype differs for {name}")
                start, end = map(int, item["data_offsets"])
                nbytes = math.prod(shape) * {"I16": 2, "F16": 2, "I32": 4}[dtype]
                if start != cursor or end - start != nbytes:
                    raise ValueError(f"payload layout differs for {name}")
                component = spec_record["components"][suffix]
                if int(component["bytes"]) != nbytes or payload_sha256(shard, data_start + start, nbytes) != component["sha256"]:
                    raise ValueError(f"component hash differs for {name}")
                cursor = end
    if set(header) != expected_names or data_start + cursor != shard.stat().st_size:
        raise ValueError("tensor set or file boundary differs")
    return {"expert": expert, "bytes": shard.stat().st_size, "sha256": receipt["output_sha256"], "relative_rmse": receipt["relative_rmse"]}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
    Path(temporary).replace(path)


def parse_experts(raw: str) -> list[int]:
    values: set[int] = set()
    for part in raw.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            values.update(range(start, stop + 1))
        else:
            values.add(int(part))
    if not values or min(values) < 0 or max(values) >= 256:
        raise ValueError("experts must be in 0..255")
    return sorted(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experts = parse_experts(args.experts)
    verified = [verify_expert(args.layer_dir, args.layer, expert) for expert in experts]
    atomic_json(args.output, {
        "schema": "glm53-full-exl3-tp3.k275-subset-verification.v1",
        "geometry_id": GEOMETRY_ID,
        "layer": args.layer,
        "bits": 2,
        "expected_experts": experts,
        "verified_experts": len(verified),
        "bytes": sum(row["bytes"] for row in verified),
        "worst_relative_rmse": max(verified, key=lambda row: (row["relative_rmse"], -row["expert"])),
        "passed": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
