#!/usr/bin/env python3
"""Fresh whole-file checksum and geometry audit of the sealed layer archive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_layer import expected_dtype, expected_shape
from tp3k3.geometry import GEOMETRY_ID, layer_geometry
from tp3k3.safetensors_stream import read_header, sha256_file


def parse_ledger(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.strip()
        if relative.startswith("./"):
            relative = relative[2:]
        if relative in rows:
            raise ValueError(f"duplicate checksum entry {relative}")
        rows[relative] = digest
    return rows


def verify_layer(root: Path, layer: int, workers: int) -> dict:
    layer_dir = root / f"layer-{layer:03d}"
    ledger = parse_ledger(layer_dir / "ARTIFACT_SHA256SUMS")
    expected_paths = {"LAYER_RECEIPT.json", "VERIFICATION.json"}
    expected_paths |= {f"experts/expert-{expert:03d}.safetensors" for expert in range(256)}
    expected_paths |= {f"receipts/expert-{expert:03d}.json" for expert in range(256)}
    if set(ledger) != expected_paths:
        raise ValueError(f"layer {layer}: checksum ledger file set differs")
    paths = [(name, layer_dir / name, digest) for name, digest in sorted(ledger.items())]

    def check(item: tuple[str, Path, str]) -> tuple[str, int]:
        name, path, expected = item
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"layer {layer}: checksum mismatch {name}")
        return name, path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        checked = dict(pool.map(check, paths))
    receipt = json.loads((layer_dir / "LAYER_RECEIPT.json").read_text(encoding="utf-8"))
    verification = json.loads((layer_dir / "VERIFICATION.json").read_text(encoding="utf-8"))
    geometry = layer_geometry(layer)
    if not (
        receipt.get("passed") is True
        and receipt.get("schema") == "glm53-full-exl3-tp3.layer-receipt.v2"
        and receipt.get("geometry_id") == GEOMETRY_ID
        and receipt.get("layer") == layer
        and receipt.get("layer_geometry") == geometry
        and receipt.get("experts") == 256
        and receipt.get("slice_encodes") == 2304
        and verification.get("passed") is True
        and verification.get("layer_geometry") == geometry
        and verification.get("experts") == 256
        and verification.get("tensors") == 9216
    ):
        raise ValueError(f"layer {layer}: receipt or verification contract differs")
    artifacts = {int(row["expert"]): row for row in receipt["expert_artifacts"]}
    if set(artifacts) != set(range(256)):
        raise ValueError(f"layer {layer}: expert artifact set differs")
    tensor_count = 0
    payload_bytes = 0
    for expert in range(256):
        relative = f"experts/expert-{expert:03d}.safetensors"
        artifact = artifacts[expert]
        if artifact["sha256"] != ledger[relative] or artifact["bytes"] != checked[relative]:
            raise ValueError(f"layer {layer}: expert {expert} receipt differs")
        data_start, header = read_header(layer_dir / relative)
        metadata = header.pop("__metadata__", {})
        if metadata.get("geometry_id") != GEOMETRY_ID or metadata.get("layer") != str(layer) or metadata.get("expert") != str(expert):
            raise ValueError(f"layer {layer}: expert {expert} metadata differs")
        names = set()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for rank in range(3):
                for suffix in ("trellis", "suh", "svh", "mcg"):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}.{suffix}"
                    names.add(name)
                    item = header.get(name)
                    if item is None or item["dtype"] != expected_dtype(suffix) or tuple(item["shape"]) != expected_shape(layer, rank, projection, suffix):
                        raise ValueError(f"layer {layer}: invalid tensor {name}")
                    payload_bytes += int(item["data_offsets"][1]) - int(item["data_offsets"][0])
        if set(header) != names:
            raise ValueError(f"layer {layer}: expert {expert} tensor set differs")
        if max(int(row["data_offsets"][1]) for row in header.values()) + data_start != checked[relative]:
            raise ValueError(f"layer {layer}: expert {expert} file boundary differs")
        tensor_count += len(header)
    return {
        "layer": layer,
        "geometry": geometry,
        "files": len(ledger),
        "expert_files": 256,
        "tensors": tensor_count,
        "payload_bytes": payload_bytes,
        "artifact_bytes": sum(checked.values()),
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    completion = json.loads(args.completion.read_text(encoding="utf-8"))
    if completion.get("layers") != 76 or Path(completion.get("publish_root", "")) != args.archive:
        raise SystemExit("bulk completion marker does not identify this 76-layer archive")
    results = []
    for layer in range(3, 79):
        row = verify_layer(args.archive, layer, max(1, args.workers))
        results.append(row)
        print(f"verified layer={layer} bytes={row['artifact_bytes']}", flush=True)
    value = {
        "schema": "glm53-full-exl3-tp3.layer-archive-verification.v1",
        "geometry_id": GEOMETRY_ID,
        "archive": str(args.archive),
        "completion": str(args.completion),
        "layers": len(results),
        "experts": sum(row["expert_files"] for row in results),
        "tensors": sum(row["tensors"] for row in results),
        "payload_bytes": sum(row["payload_bytes"] for row in results),
        "artifact_bytes": sum(row["artifact_bytes"] for row in results),
        "layer_results": results,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
