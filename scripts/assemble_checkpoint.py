#!/usr/bin/env python3
"""Resumable, bounded-memory assembly of the full rotating-uneven K3 checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import GEOMETRY_ID, checkpoint_geometry, layer_geometry
from tp3k3.safetensors_stream import atomic_copy_verified, read_header, sha256_file, write_subset


AUXILIARY = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def atomic_json(path: Path, value: Any) -> None:
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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def status(state_dir: Path, phase: str, completed: int, total: int, **extra: Any) -> None:
    atomic_json(state_dir / "STATUS.json", {
        "schema": "glm53-full-exl3-tp3.assembly-status.v1",
        "phase": phase,
        "completed": completed,
        "total": total,
        **extra,
    })


def copy_or_resume(source: Path, target: Path, expected_sha256: str) -> dict:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        actual = sha256_file(target)
        if actual == expected_sha256:
            return {"bytes": target.stat().st_size, "sha256": actual, "resumed": True}
    result = atomic_copy_verified(source, target, expected_sha256)
    result["resumed"] = False
    return result


def validate_expert_header(path: Path, layer: int, expert: int) -> tuple[dict[str, str], int]:
    _, header = read_header(path)
    metadata = header.pop("__metadata__", {})
    if (
        metadata.get("schema") != "glm53-full-exl3-tp3.expert-shard.v2"
        or metadata.get("geometry_id") != GEOMETRY_ID
        or metadata.get("layer") != str(layer)
        or metadata.get("expert") != str(expert)
        or metadata.get("bits") != "3"
        or len(header) != 36
    ):
        raise ValueError(f"expert header contract differs: {path}")
    sizes = {}
    for name, item in header.items():
        start, end = map(int, item["data_offsets"])
        sizes[name] = end - start
    return sizes, sum(sizes.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--source-structure", type=Path, required=True)
    parser.add_argument("--tensor-plan", type=Path, required=True)
    parser.add_argument("--layer-archive", type=Path, required=True)
    parser.add_argument("--archive-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise SystemExit("output path exists and is not a directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)

    inventory_raw = args.source_inventory.read_bytes()
    inventory = json.loads(inventory_raw)
    structure_raw = args.source_structure.read_bytes()
    structure = json.loads(structure_raw)
    plan_raw = args.tensor_plan.read_bytes()
    plan = json.loads(plan_raw)
    archive_verification_raw = args.archive_verification.read_bytes()
    archive_verification = json.loads(archive_verification_raw)
    if not (
        plan.get("passed") is True
        and plan.get("geometry_id") == GEOMETRY_ID
        and plan.get("routed_tensor_count") == 58368
        and plan.get("passthrough_tensor_count") == 1217
        and plan.get("includes_mtp_layer") is True
        and archive_verification.get("passed") is True
        and archive_verification.get("layers") == 76
        and Path(archive_verification.get("archive", "")) == args.layer_archive
    ):
        raise SystemExit("source plan or archive verification gate failed")
    output_files: dict[str, dict] = {}
    weight_map: dict[str, str] = {}
    tensor_sizes: dict[str, int] = {}

    # Preserve the public model/tokenizer contract byte-for-byte.
    for index, name in enumerate(AUXILIARY, 1):
        record = structure["auxiliary"][name]
        result = copy_or_resume(args.source_root / name, args.output_dir / name, record["sha256"])
        output_files[name] = result
        status(args.state_dir, "auxiliary", index, len(AUXILIARY), file=name)

    carry_by_shard: dict[str, list[dict]] = defaultdict(list)
    tier_rows = []
    inventory_entries = inventory["entries"]
    for row in plan["tensors"]:
        tier = {k: row[k] for k in row if k not in {"source_payload_sha256", "output_payload_sha256"}}
        tier["payload_sha256"] = row["source_payload_sha256"]
        tier_rows.append(tier)
        if row["action"] == "carry_byte_exact":
            source = inventory_entries[row["name"]]
            carry_by_shard[row["source_shard"]].append({"name": row["name"], **source})

    passthrough_receipts = {}
    source_shards = sorted(carry_by_shard)
    for index, source_name in enumerate(source_shards, 1):
        suffix = source_name.removeprefix("model-")
        output_name = f"bf16-passthrough-{suffix}"
        target = args.output_dir / output_name
        receipt_path = args.state_dir / "passthrough" / f"{output_name}.json"
        rows = carry_by_shard[source_name]
        expected_payload = {row["name"]: row["payload_sha256"] for row in rows}
        receipt = None
        if target.is_file() and receipt_path.is_file():
            candidate = json.loads(receipt_path.read_text(encoding="utf-8"))
            if candidate.get("source_shard_sha256") == inventory["shards"][source_name]["file_sha256"] and candidate.get("tensor_sha256") == expected_payload and target.stat().st_size == candidate.get("bytes") and sha256_file(target) == candidate.get("sha256"):
                receipt = candidate
        if receipt is None:
            receipt = write_subset(
                args.source_root / source_name,
                target,
                rows,
                metadata={
                    "schema": "glm53-full-exl3-tp3.bf16-passthrough-shard.v1",
                    "source_revision": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
                    "source_shard": source_name,
                },
            )
            if receipt["tensor_sha256"] != expected_payload:
                raise ValueError(f"byte identity failed while copying {source_name}")
            receipt.update({
                "schema": "glm53-full-exl3-tp3.passthrough-receipt.v1",
                "source_shard": source_name,
                "source_shard_sha256": inventory["shards"][source_name]["file_sha256"],
                "output_file": output_name,
            })
            atomic_json(receipt_path, receipt)
        passthrough_receipts[output_name] = receipt
        output_files[output_name] = {"bytes": receipt["bytes"], "sha256": receipt["sha256"]}
        for row in rows:
            weight_map[row["name"]] = output_name
            tensor_sizes[row["name"]] = int(row["nbytes"])
        status(args.state_dir, "passthrough", index, len(source_shards), file=output_name)

    expert_total = 76 * 256
    expert_done = 0
    for layer in range(3, 79):
        receipt_path = args.layer_archive / f"layer-{layer:03d}" / "LAYER_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("passed") is not True or receipt.get("layer_geometry") != layer_geometry(layer):
            raise ValueError(f"sealed layer receipt differs for layer {layer}")
        artifacts = {int(row["expert"]): row for row in receipt["expert_artifacts"]}
        if set(artifacts) != set(range(256)):
            raise ValueError(f"sealed expert set differs for layer {layer}")
        for expert in range(256):
            source = args.layer_archive / f"layer-{layer:03d}" / "experts" / f"expert-{expert:03d}.safetensors"
            output_name = f"k3-layer-{layer:03d}-expert-{expert:03d}.safetensors"
            artifact = artifacts[expert]
            result = copy_or_resume(source, args.output_dir / output_name, artifact["sha256"])
            if result["bytes"] != artifact["bytes"]:
                raise ValueError(f"expert size differs for layer {layer} expert {expert}")
            sizes, _ = validate_expert_header(args.output_dir / output_name, layer, expert)
            output_files[output_name] = result
            for name, size in sizes.items():
                if name in weight_map:
                    raise ValueError(f"duplicate output tensor {name}")
                weight_map[name] = output_name
                tensor_sizes[name] = size
            expert_done += 1
            if expert_done % 16 == 0 or expert_done == expert_total:
                status(args.state_dir, "experts", expert_done, expert_total, layer=layer, expert=expert)

    geometry = checkpoint_geometry()
    geometry.update({
        "schema": "checkpoint-geometry.v2",
        "attention": {"semantic_heads": 64, "physical_heads": 66, "heads_per_rank": 22, "padding_heads": 2},
        "vocabulary": {"semantic_tokens": 154880, "physical_tokens": 154944, "padding_tokens": 64},
    })
    generated = {
        "checkpoint-geometry.json": geometry,
        "quantization_config.json": {
            "bits": 3,
            "codebook": "mcg",
            "geometry_id": GEOMETRY_ID,
            "head_bits": 16,
            "non_routed_dtype_policy": "official_source_native",
            "quant_method": "exl3",
            "scope": "glm53_routed_experts_only",
            "serving_reader_qualified": False,
            "version": "0.0.43",
        },
        "tier-map.json": {
            "schema": "glm53-full-exl3-tp3.tier-map.v1",
            "geometry_id": GEOMETRY_ID,
            "summary": {
                "k3_source_tensors": 58368,
                "bf16_carried_tensors": 1217,
                "k4_tensors": 0,
            },
            "tensors": tier_rows,
        },
        "source-manifest.json": json.loads(structure_raw),
        "source-tensor-plan.json": json.loads(plan_raw),
        "layer-archive-verification.json": archive_verification,
    }
    for name, value in generated.items():
        atomic_json(args.output_dir / name, value)
        output_files[name] = {"bytes": (args.output_dir / name).stat().st_size, "sha256": sha256_file(args.output_dir / name)}

    if len(weight_map) != 1217 + 76 * 256 * 36:
        raise ValueError(f"output index tensor count differs: {len(weight_map)}")
    index = {
        "metadata": {
            "total_size": sum(tensor_sizes.values()),
            "geometry_id": GEOMETRY_ID,
            "source_revision": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    atomic_json(args.output_dir / "model.safetensors.index.json", index)
    output_files["model.safetensors.index.json"] = {
        "bytes": (args.output_dir / "model.safetensors.index.json").stat().st_size,
        "sha256": sha256_file(args.output_dir / "model.safetensors.index.json"),
    }
    manifest = {
        "schema": "glm53-full-exl3-tp3.checkpoint-manifest.v1",
        "geometry_id": GEOMETRY_ID,
        "source_revision": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
        "layer_archive": str(args.layer_archive),
        "layer_archive_verification_sha256": hashlib.sha256(archive_verification_raw).hexdigest(),
        "tensor_counts": {
            "total": len(weight_map),
            "bf16_carried": 1217,
            "k3_payload": 76 * 256 * 36,
            "semantic_routed_weights_replaced": 58368,
        },
        "file_count": len(output_files),
        "file_bytes": sum(int(row["bytes"]) for row in output_files.values()),
        "indexed_payload_bytes": index["metadata"]["total_size"],
        "files": dict(sorted(output_files.items())),
    }
    atomic_json(args.output_dir / "MANIFEST.json", manifest)
    output_files["MANIFEST.json"] = {"bytes": (args.output_dir / "MANIFEST.json").stat().st_size, "sha256": sha256_file(args.output_dir / "MANIFEST.json")}
    ledger = "".join(f"{row['sha256']}  {name}\n" for name, row in sorted(output_files.items()))
    atomic_text(args.output_dir / "SHA256SUMS", ledger)
    complete = {
        "schema": "glm53-full-exl3-tp3.assembly-complete.v1",
        "passed": True,
        "geometry_id": GEOMETRY_ID,
        "output_dir": str(args.output_dir),
        "layers": 76,
        "experts": expert_total,
        "indexed_tensors": len(weight_map),
        "file_bytes": manifest["file_bytes"],
        "manifest_sha256": output_files["MANIFEST.json"]["sha256"],
        "sha256sums_sha256": sha256_file(args.output_dir / "SHA256SUMS"),
    }
    atomic_json(args.output_dir / "ASSEMBLY_COMPLETE.json", complete)
    atomic_json(args.state_dir / "COMPLETE.json", complete)
    status(args.state_dir, "complete", expert_total, expert_total, output_dir=str(args.output_dir))


if __name__ == "__main__":
    main()
