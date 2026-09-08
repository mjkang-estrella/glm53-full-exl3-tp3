#!/usr/bin/env python3
"""Checksum and classify resumable expert artifacts in an unsealed layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tp3k3.geometry import GEOMETRY_ID, layer_geometry


SOURCE_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_partial(layer_dir: Path, layer: int) -> dict[str, object]:
    receipt_dir = layer_dir / "receipts"
    expert_dir = layer_dir / "experts"
    receipts = sorted(receipt_dir.glob("expert-*.json")) if receipt_dir.is_dir() else []
    shards = sorted(expert_dir.glob("expert-*.safetensors")) if expert_dir.is_dir() else []
    errors: list[str] = []
    verified: list[dict[str, object]] = []
    receipt_experts: set[int] = set()

    for receipt_path in receipts:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expert = int(receipt["expert"])
            expected_name = f"expert-{expert:03d}.safetensors"
            shard = expert_dir / expected_name
            if expert in receipt_experts:
                raise ValueError("duplicate expert receipt")
            receipt_experts.add(expert)
            if not 0 <= expert < 256:
                raise ValueError("expert outside 0..255")
            checks = (
                receipt.get("schema") == "glm53-full-exl3-tp3.expert-receipt.v2",
                receipt.get("geometry_id") == GEOMETRY_ID,
                receipt.get("layer") == layer,
                receipt.get("bits") == 3,
                receipt.get("tp") == 3,
                receipt.get("padding_channels") == 0,
                receipt.get("layer_geometry") == layer_geometry(layer),
                receipt.get("source_revision") == SOURCE_REVISION,
                receipt.get("output_file") == expected_name,
                receipt.get("passed") is True,
                shard.is_file(),
            )
            if not all(checks):
                raise ValueError("receipt identity or paired shard differs")
            size = shard.stat().st_size
            digest = sha256_file(shard)
            if size != receipt.get("output_bytes") or digest != receipt.get("output_sha256"):
                raise ValueError("artifact size or SHA-256 differs")
            verified.append({"expert": expert, "bytes": size, "sha256": digest})
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{receipt_path.name}: {exc}")

    shard_names = {path.name for path in shards}
    expected_names = {f"expert-{expert:03d}.safetensors" for expert in receipt_experts}
    extras = sorted(shard_names - expected_names)
    if extras:
        errors.append(f"unpaired expert shards: {extras}")

    return {
        "schema": "glm53-full-exl3-tp3.partial-layer-verification.v1",
        "geometry_id": GEOMETRY_ID,
        "layer": layer,
        "layer_dir": str(layer_dir),
        "complete_layer_receipt_present": (layer_dir / "LAYER_RECEIPT.json").is_file(),
        "verified_experts": len(verified),
        "verified_bytes": sum(int(row["bytes"]) for row in verified),
        "expert_ids": [int(row["expert"]) for row in verified],
        "errors": errors,
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = verify_partial(args.layer_dir, args.layer)
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(encoded, end="")
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
