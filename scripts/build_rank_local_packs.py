#!/usr/bin/env python3
"""Build resumable byte-exact rank-local K3 layer packs on Spark NVMe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tp3k3.geometry import GEOMETRY_ID, rank_geometry


LAYERS = tuple(range(3, 79))
EXPERTS = 256
TENSORS_PER_EXPERT_RANK = 12
RESERVE_BYTES = 12 * 1024**3


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def valid_receipt(path: Path, output: Path, *, layer: int, rank: int) -> bool:
    if not path.is_file() or not output.is_file():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        row.get("passed") is True
        and row.get("geometry_id") == GEOMETRY_ID
        and row.get("layer") == layer
        and row.get("rank") == rank
        and row.get("tensor_count") == EXPERTS * TENSORS_PER_EXPERT_RANK
        and row.get("bytes") == output.stat().st_size
        and row.get("sha256") == sha256_file(output)
    )


def build_layer(model: Path, output: Path, receipt: Path, *, layer: int, rank: int) -> dict:
    started_at = utc()
    started = time.monotonic()
    tensors: dict[str, torch.Tensor] = {}
    source_files = []
    marker = f".rank{rank}."
    for expert in range(EXPERTS):
        if mem_available() < RESERVE_BYTES:
            raise RuntimeError("12 GiB host-memory reserve breached while packing")
        source = model / f"k3-layer-{layer:03d}-expert-{expert:03d}.safetensors"
        if not source.is_file():
            raise FileNotFoundError(source)
        with safe_open(source, framework="pt", device="cpu") as handle:
            keys = sorted(key for key in handle.keys() if marker in key)
            if len(keys) != TENSORS_PER_EXPERT_RANK:
                raise RuntimeError(f"rank-local tensor count differs in {source}: {len(keys)}")
            for key in keys:
                if key in tensors:
                    raise RuntimeError(f"duplicate rank-local tensor: {key}")
                tensors[key] = handle.get_tensor(key)
        source_files.append(source.name)
    expected = EXPERTS * TENSORS_PER_EXPERT_RANK
    if len(tensors) != expected:
        raise RuntimeError(f"layer tensor count differs: {len(tensors)}/{expected}")
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    metadata = {
        "schema": "glm53-full-exl3-tp3.rank-local-layer-pack.v1",
        "geometry_id": GEOMETRY_ID,
        "layer": str(layer),
        "rank": str(rank),
        "offset": str(rank_geometry(layer, rank)[0]),
        "width": str(rank_geometry(layer, rank)[1]),
    }
    save_file(tensors, temporary, metadata=metadata)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    # Verify every typed value before the atomic publication rename.
    with safe_open(temporary, framework="pt", device="cpu") as packed:
        if set(packed.keys()) != set(tensors):
            raise RuntimeError("packed tensor index differs")
        for key, source_tensor in tensors.items():
            if not torch.equal(packed.get_tensor(key), source_tensor):
                raise RuntimeError(f"packed tensor bytes differ: {key}")
    digest = sha256_file(temporary)
    size = temporary.stat().st_size
    os.replace(temporary, output)
    row = {
        "schema": "glm53-full-exl3-tp3.rank-local-layer-pack-receipt.v1",
        "passed": True,
        "geometry_id": GEOMETRY_ID,
        "layer": layer,
        "rank": rank,
        "offset": rank_geometry(layer, rank)[0],
        "width": rank_geometry(layer, rank)[1],
        "experts": EXPERTS,
        "tensor_count": len(tensors),
        "source_files": len(source_files),
        "bytes": size,
        "sha256": digest,
        "started_at": started_at,
        "completed_at": utc(),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(receipt, row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(3), required=True)
    args = parser.parse_args()
    complete = args.model_dir / "ASSEMBLY_COMPLETE.json"
    if not complete.is_file():
        raise FileNotFoundError(complete)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipts = args.output_dir / "receipts"
    receipts.mkdir(exist_ok=True)
    started_at = utc()
    completed_rows = []
    for index, layer in enumerate(LAYERS):
        output = args.output_dir / f"k3-layer-{layer:03d}-rank{args.rank}.safetensors"
        receipt = receipts / f"layer-{layer:03d}.json"
        atomic_json(
            args.output_dir / "STATUS.json",
            {
                "schema": "glm53-full-exl3-tp3.rank-local-pack-status.v1",
                "phase": "packing",
                "rank": args.rank,
                "layer": layer,
                "completed_layers": index,
                "total_layers": len(LAYERS),
                "updated_at": utc(),
            },
        )
        if valid_receipt(receipt, output, layer=layer, rank=args.rank):
            row = json.loads(receipt.read_text(encoding="utf-8"))
        else:
            if output.exists() or receipt.exists():
                raise RuntimeError(
                    f"unverified existing rank-pack artifact requires preservation: {output}"
                )
            row = build_layer(
                args.model_dir, output, receipt, layer=layer, rank=args.rank
            )
        completed_rows.append(row)
    result = {
        "schema": "glm53-full-exl3-tp3.rank-local-pack-complete.v1",
        "passed": True,
        "geometry_id": GEOMETRY_ID,
        "rank": args.rank,
        "layers": len(completed_rows),
        "first_layer": min(LAYERS),
        "last_layer": max(LAYERS),
        "experts_per_layer": EXPERTS,
        "tensor_count": sum(row["tensor_count"] for row in completed_rows),
        "bytes": sum(row["bytes"] for row in completed_rows),
        "assembly_complete_sha256": sha256_file(complete),
        "started_at": started_at,
        "completed_at": utc(),
    }
    atomic_json(args.output_dir / "COMPLETE.json", result)
    atomic_json(args.output_dir / "STATUS.json", {**result, "phase": "complete"})
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
