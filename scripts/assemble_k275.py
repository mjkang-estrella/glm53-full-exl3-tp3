#!/usr/bin/env python3
"""Assemble a mixed K2/K3 2.75bpw checkpoint from immutable K3 + K2 layers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

GEOMETRY_ID = "rotating-uneven-768-640-640-v1"
SOURCE_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
ROUTED_LAYERS = range(3, 79)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def payload_bytes(path: Path) -> int:
    data_start, header = read_header(path)
    return sum(int(item["data_offsets"][1]) - int(item["data_offsets"][0]) for key, item in header.items() if key != "__metadata__")


def copy_tree_links(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for root, dirs, files in os.walk(source):
        relative = Path(root).relative_to(source)
        destination = target / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = Path(root) / name
            dst = destination / name
            if src.is_symlink():
                raise ValueError(f"symlink in checkpoint: {src}")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def selection_map(path: Path) -> dict[int, list[int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "glm53-full-exl3-tp3.k275-bit-selection.v1":
        raise ValueError("unexpected K2 selection schema")
    result = {}
    for layer in ROUTED_LAYERS:
        row = value["layers"][str(layer)]
        selected = [int(x) for x in row["k2_experts"]]
        if len(selected) != 64 or selected != sorted(set(selected)):
            raise ValueError(f"invalid K2 selection for layer {layer}")
        result[layer] = selected
    return result


def validate_k2_layer(path: Path, layer: int, selected: list[int]) -> dict:
    receipt = json.loads((path / "LAYER_K2_RECEIPT.json").read_text(encoding="utf-8"))
    verification = json.loads((path / "VERIFICATION.json").read_text(encoding="utf-8"))
    if not (
        receipt.get("passed") is True
        and receipt.get("geometry_id") == GEOMETRY_ID
        and receipt.get("layer") == layer
        and receipt.get("bits") == 2
        and receipt.get("k2_experts") == selected
        and verification.get("passed") is True
        and verification.get("geometry_id") == GEOMETRY_ID
        and verification.get("layer") == layer
        and verification.get("bits") == 2
        and verification.get("expected_experts") == selected
        and verification.get("verified_experts") == 64
    ):
        raise ValueError(f"K2 layer receipt differs: {path}")
    for expert in selected:
        shard = path / "experts" / f"expert-{expert:03d}.safetensors"
        if not shard.is_file():
            raise ValueError(f"missing K2 expert {shard}")
        _start, header = read_header(shard)
        metadata = header.get("__metadata__", {})
        if metadata.get("bits") != "2" or metadata.get("geometry_id") != GEOMETRY_ID:
            raise ValueError(f"K2 expert metadata differs: {shard}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--k2-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_checkpoint.resolve()
    k2_root = args.k2_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    base_complete = json.loads((base / "ASSEMBLY_COMPLETE.json").read_text(encoding="utf-8"))
    if not (base_complete.get("passed") is True and base_complete.get("geometry_id") == GEOMETRY_ID and base_complete.get("layers") == 76 and base_complete.get("experts") == 19456):
        raise SystemExit("base checkpoint is not the sealed 3.0bpw K3 assembly")
    selected = selection_map(args.selection)
    k2_receipts = {}
    for layer in ROUTED_LAYERS:
        layer_path = k2_root / f"layer-{layer:03d}"
        k2_receipts[layer] = validate_k2_layer(layer_path, layer, selected[layer])
    staging = output.with_name(output.name + f".incoming-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    if staging.exists():
        raise SystemExit(f"staging output already exists: {staging}")
    copy_tree_links(base, staging)
    try:
        # Replace only the selected K3 expert files. All other files remain
        # hardlinks/copies of the immutable sealed K3 base.
        for layer in ROUTED_LAYERS:
            for expert in selected[layer]:
                source = k2_root / f"layer-{layer:03d}/experts/expert-{expert:03d}.safetensors"
                target = staging / f"k3-layer-{layer:03d}-expert-{expert:03d}.safetensors"
                temporary = target.with_name(f".{target.name}.k2.tmp")
                shutil.copy2(source, temporary)
                os.replace(temporary, target)
        expert_bits = {
            "schema": "glm53-full-exl3-tp3.k275-expert-bits.v1",
            "geometry_id": GEOMETRY_ID,
            "target_bpw": "2.75",
            "bits": {str(layer): {"k2": selected[layer], "k3": [e for e in range(256) if e not in selected[layer]]} for layer in ROUTED_LAYERS},
        }
        atomic_json(staging / "expert-bits.json", expert_bits)
        quant = json.loads((staging / "quantization_config.json").read_text(encoding="utf-8"))
        quant.update({"bits": 3, "target_bpw": "2.75", "mixed_expert_bits": True, "k2_experts_per_layer": 64, "k3_experts_per_layer": 192, "expert_bits_file": "expert-bits.json", "serving_reader_qualified": False})
        atomic_json(staging / "quantization_config.json", quant)
        tier = json.loads((staging / "tier-map.json").read_text(encoding="utf-8"))
        tier["summary"].update({"target_bpw": "2.75", "k2_expert_count": 76 * 64, "k3_expert_count": 76 * 192, "k2_tensors": 76 * 64 * 36, "k3_tensors": 76 * 192 * 36, "k2_source": str(k2_root), "k2_selection": str(args.selection)})
        atomic_json(staging / "tier-map.json", tier)
        evidence_dir = staging / "k275-evidence"
        for layer in ROUTED_LAYERS:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(k2_root / f"layer-{layer:03d}/LAYER_K2_RECEIPT.json", evidence_dir / f"layer-{layer:03d}-LAYER_K2_RECEIPT.json")
            shutil.copy2(k2_root / f"layer-{layer:03d}/VERIFICATION.json", evidence_dir / f"layer-{layer:03d}-VERIFICATION.json")
        index = json.loads((staging / "model.safetensors.index.json").read_text(encoding="utf-8"))
        index["metadata"]["total_size"] = sum(payload_bytes(path) for path in staging.glob("*.safetensors"))
        index["metadata"]["target_bpw"] = "2.75"
        atomic_json(staging / "model.safetensors.index.json", index)
        files = {}
        for root, _dirs, names in os.walk(staging):
            for name in names:
                path = Path(root) / name
                if path.name in {"SHA256SUMS", "MANIFEST.json", "ASSEMBLY_COMPLETE.json"}:
                    continue
                relative = str(path.relative_to(staging))
                files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        ledger = "".join(f"{row['sha256']}  {name}\n" for name, row in sorted(files.items()))
        (staging / "SHA256SUMS").write_text(ledger, encoding="utf-8")
        manifest = {
            "schema": "glm53-full-exl3-tp3.k275-assembly-manifest.v1",
            "geometry_id": GEOMETRY_ID,
            "source_revision": SOURCE_REVISION,
            "target_bpw": "2.75",
            "base_checkpoint": str(base),
            "k2_root": str(k2_root),
            "selection": str(args.selection),
            "tensor_counts": {"total": len(index["weight_map"]), "bf16_carried": 1217, "k2_payload": 76 * 64 * 36, "k3_payload": 76 * 192 * 36},
            "file_count": len(files) + 1,
            "file_bytes": sum(row["bytes"] for row in files.values()) + (staging / "SHA256SUMS").stat().st_size,
            "indexed_payload_bytes": index["metadata"]["total_size"],
            "files": files,
        }
        atomic_json(staging / "MANIFEST.json", manifest)
        complete = {
            "schema": "glm53-full-exl3-tp3.k275-assembly-complete.v1",
            "passed": True,
            "geometry_id": GEOMETRY_ID,
            "target_bpw": "2.75",
            "output_dir": str(output),
            "layers": 76,
            "experts": 76 * 256,
            "k2_experts": 76 * 64,
            "k3_experts": 76 * 192,
            "indexed_tensors": len(index["weight_map"]),
            "file_bytes": manifest["file_bytes"],
            "indexed_payload_bytes": index["metadata"]["total_size"],
            "manifest_sha256": sha256_file(staging / "MANIFEST.json"),
            "sha256sums_sha256": sha256_file(staging / "SHA256SUMS"),
        }
        atomic_json(staging / "ASSEMBLY_COMPLETE.json", complete)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(complete, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
