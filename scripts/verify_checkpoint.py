#!/usr/bin/env python3
"""Independent structural, checksum, geometry, and BF16 identity verifier."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_layer import expected_dtype, expected_shape
from tp3k3.geometry import GEOMETRY_ID, checkpoint_geometry, layer_geometry
from tp3k3.safetensors_stream import read_header, sha256_file, sha256_range


K3_NAME = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.rank(?P<rank>[0-2])\."
    r"(?P<suffix>trellis|suh|svh|mcg)$"
)


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


def parse_ledger(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        if name in rows:
            raise ValueError(f"duplicate checksum row: {name}")
        rows[name] = digest
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-file-checksums", action="store_true")
    args = parser.parse_args()
    root = args.checkpoint
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "ASSEMBLY_COMPLETE.json").read_text(encoding="utf-8"))
    ledger = parse_ledger(root / "SHA256SUMS")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tier = json.loads((root / "tier-map.json").read_text(encoding="utf-8"))
    geometry = json.loads((root / "checkpoint-geometry.json").read_text(encoding="utf-8"))
    source_plan = json.loads((root / "source-tensor-plan.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_geometry = checkpoint_geometry()
    if geometry.get("geometry_id") != GEOMETRY_ID or geometry.get("layers") != expected_geometry["layers"]:
        errors.append("checkpoint geometry differs from rotating uneven v1")
    if geometry.get("attention") != {"semantic_heads": 64, "physical_heads": 66, "heads_per_rank": 22, "padding_heads": 2}:
        errors.append("attention padding metadata differs")
    if geometry.get("vocabulary") != {"semantic_tokens": 154880, "physical_tokens": 154944, "padding_tokens": 64}:
        errors.append("vocabulary padding metadata differs")

    files = manifest["files"]
    expected_files = set(files) | {"MANIFEST.json", "SHA256SUMS", "ASSEMBLY_COMPLETE.json"}
    actual_files = {path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()}
    if actual_files != expected_files:
        errors.append(
            f"checkpoint file set differs missing={sorted(expected_files-actual_files)[:5]} "
            f"extra={sorted(actual_files-expected_files)[:5]}"
        )
    if any(path.is_symlink() for path in root.iterdir()):
        errors.append("checkpoint contains symbolic links")
    if set(ledger) != set(files) | {"MANIFEST.json"}:
        errors.append("checksum ledger file set differs")
    elif (
        ledger["MANIFEST.json"] != sha256_file(root / "MANIFEST.json")
        or any(ledger[name] != record["sha256"] for name, record in files.items())
    ):
        errors.append("checksum ledger differs from manifest")
    if not (
        complete.get("passed") is True
        and complete.get("geometry_id") == GEOMETRY_ID
        and complete.get("indexed_tensors") == 701633
        and complete.get("manifest_sha256") == sha256_file(root / "MANIFEST.json")
        and complete.get("sha256sums_sha256") == sha256_file(root / "SHA256SUMS")
        and Path(complete.get("output_dir", "")) == root
    ):
        errors.append("assembly completion marker differs")
    if manifest.get("file_count") != len(files) or manifest.get("file_bytes") != sum(int(row["bytes"]) for row in files.values()):
        errors.append("manifest file count or byte total differs")
    missing_files = [name for name in files if not (root / name).is_file()]
    if missing_files:
        errors.append(f"missing manifest files: {missing_files[:10]}")
    checksum_failures = []
    if not args.skip_file_checksums and not missing_files:
        def verify_file(item: tuple[str, dict]) -> str | None:
            name, record = item
            path = root / name
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                return name
            return None
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            checksum_failures = [name for name in pool.map(verify_file, sorted(files.items())) if name]
        if checksum_failures:
            errors.append(f"file checksum failures: {checksum_failures[:10]}")

    physical: dict[str, tuple[str, dict, int]] = {}
    duplicate_physical = []
    k3_coordinates: dict[tuple[int, int], set[tuple[str, int, str]]] = {}
    payload_total = 0
    k3_payload_by_rank = {str(rank): 0 for rank in range(3)}
    k3_target_payload_by_rank = {str(rank): 0 for rank in range(3)}
    safetensors_files = sorted(name for name in files if name.endswith(".safetensors"))
    for file_name in safetensors_files:
        data_start, header = read_header(root / file_name)
        metadata = header.pop("__metadata__", {})
        ranges = []
        for name, item in header.items():
            if name in physical:
                duplicate_physical.append(name)
            start, end = map(int, item["data_offsets"])
            if start < 0 or end <= start:
                errors.append(f"invalid payload range: {name}")
            ranges.append((start, end, name))
            physical[name] = (file_name, item, data_start)
            payload_total += end - start
            match = K3_NAME.match(name)
            if match:
                layer = int(match.group("layer"))
                expert = int(match.group("expert"))
                projection = match.group("projection")
                rank = int(match.group("rank"))
                suffix = match.group("suffix")
                tensor_bytes = end - start
                k3_payload_by_rank[str(rank)] += tensor_bytes
                if layer < 78:
                    k3_target_payload_by_rank[str(rank)] += tensor_bytes
                k3_coordinates.setdefault((layer, expert), set()).add((projection, rank, suffix))
                if layer not in range(3, 79) or expert not in range(256):
                    errors.append(f"out-of-contract K3 name: {name}")
                elif item["dtype"] != expected_dtype(suffix) or tuple(item["shape"]) != expected_shape(layer, rank, projection, suffix):
                    errors.append(f"K3 dtype/shape differs: {name}")
            elif ".mlp.experts." in name:
                errors.append(f"unrecognized routed-expert tensor: {name}")
        ranges.sort()
        cursor = 0
        for start, end, name in ranges:
            if start != cursor:
                errors.append(f"gap/overlap before {name} in {file_name}")
                break
            cursor = end
        if data_start + cursor != (root / file_name).stat().st_size:
            errors.append(f"unindexed bytes in {file_name}")
        if file_name.startswith("k3-") and metadata.get("geometry_id") != GEOMETRY_ID:
            errors.append(f"K3 file metadata differs: {file_name}")

    indexed = index["weight_map"]
    if duplicate_physical:
        errors.append(f"duplicate physical tensors: {duplicate_physical[:10]}")
    if set(indexed) != set(physical):
        missing = sorted(set(indexed) - set(physical))
        extra = sorted(set(physical) - set(indexed))
        errors.append(f"index/physical mismatch missing={missing[:5]} extra={extra[:5]}")
    wrong_file = [name for name, file_name in indexed.items() if name in physical and physical[name][0] != file_name]
    if wrong_file:
        errors.append(f"index points to wrong files: {wrong_file[:10]}")

    expected_payloads = {(projection, rank, suffix) for projection in ("gate_proj", "up_proj", "down_proj") for rank in range(3) for suffix in ("trellis", "suh", "svh", "mcg")}
    missing_experts = []
    for layer in range(3, 79):
        if geometry["layers"].get(str(layer)) != layer_geometry(layer):
            errors.append(f"layer geometry differs: {layer}")
        for expert in range(256):
            if k3_coordinates.get((layer, expert)) != expected_payloads:
                missing_experts.append((layer, expert))
    if missing_experts:
        errors.append(f"K3 expert payload sets differ: {missing_experts[:10]}")

    plan_by_name = {row["name"]: row for row in source_plan["tensors"]}
    carry_names = {name for name, row in plan_by_name.items() if row["action"] == "carry_byte_exact"}
    actual_carry = set(physical) - {name for name in physical if K3_NAME.match(name)}
    if actual_carry != carry_names:
        errors.append("carried BF16 tensor set differs from source plan")
    carry_hash_failures = []
    for count, name in enumerate(sorted(carry_names), 1):
        if name not in physical:
            continue
        file_name, item, data_start = physical[name]
        start, end = map(int, item["data_offsets"])
        digest = sha256_range(root / file_name, data_start + start, end - start)
        if digest != plan_by_name[name]["source_payload_sha256"]:
            carry_hash_failures.append(name)
        if count % 100 == 0:
            print(f"verified carried tensors={count}/{len(carry_names)}", flush=True)
    if carry_hash_failures:
        errors.append(f"BF16 byte-identity failures: {carry_hash_failures[:10]}")

    auxiliary_hash_failures = []
    source_manifest = json.loads((root / "source-manifest.json").read_text(encoding="utf-8"))
    for name, record in source_manifest["auxiliary"].items():
        if name == "model.safetensors.index.json":
            continue
        if not (root / name).is_file() or sha256_file(root / name) != record["sha256"]:
            auxiliary_hash_failures.append(name)
    if auxiliary_hash_failures:
        errors.append(f"source auxiliary identity failures: {auxiliary_hash_failures}")

    expected_total = 1217 + 76 * 256 * 36
    if len(indexed) != expected_total or len(physical) != expected_total:
        errors.append(f"tensor count differs: indexed={len(indexed)} physical={len(physical)} expected={expected_total}")
    if int(index["metadata"]["total_size"]) != payload_total:
        errors.append(f"index payload total differs: {index['metadata']['total_size']} != {payload_total}")
    if tier.get("summary") != {"k3_source_tensors": 58368, "bf16_carried_tensors": 1217, "k4_tensors": 0}:
        errors.append("tier summary differs")

    result = {
        "schema": "glm53-full-exl3-tp3.checkpoint-verification.v1",
        "checkpoint": str(root),
        "geometry_id": GEOMETRY_ID,
        "layers": 76,
        "experts": len(k3_coordinates),
        "k3_payload_tensors": sum(len(value) for value in k3_coordinates.values()),
        "k3_payload_bytes_by_rank_all_76_layers": k3_payload_by_rank,
        "k3_payload_bytes_by_rank_target_layers_3_through_77": k3_target_payload_by_rank,
        "bf16_carried_tensors": len(actual_carry),
        "indexed_tensors": len(indexed),
        "physical_tensors": len(physical),
        "safetensors_files": len(safetensors_files),
        "indexed_payload_bytes": payload_total,
        "manifest_file_bytes": manifest["file_bytes"],
        "assembly_complete_sha256": sha256_file(root / "ASSEMBLY_COMPLETE.json"),
        "full_file_checksums_verified": not args.skip_file_checksums,
        "checksum_failures": checksum_failures,
        "carry_hash_failures": carry_hash_failures,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(args.output, result)
    if errors:
        raise SystemExit("checkpoint verification failed: " + "; ".join(errors[:5]))


if __name__ == "__main__":
    main()
