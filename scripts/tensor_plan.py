#!/usr/bin/env python3
"""Seal the routed-quantize versus byte-exact-carry tensor disposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import GEOMETRY_ID, checkpoint_geometry, layer_geometry


ROUTED_LAYERS = set(range(3, 79))
PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}


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


def routed_coordinate(name: str) -> tuple[int, int, str] | None:
    parts = name.split(".")
    # model.layers.<layer>.mlp.experts.<expert>.<projection>.weight
    if len(parts) != 8 or parts[:2] != ["model", "layers"]:
        return None
    if parts[3:5] != ["mlp", "experts"] or parts[7] != "weight":
        return None
    try:
        layer, expert = int(parts[2]), int(parts[5])
    except ValueError:
        return None
    projection = parts[6]
    if layer not in ROUTED_LAYERS or expert not in range(256) or projection not in PROJECTIONS:
        return None
    return layer, expert, projection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_inventory.read_bytes()
    source = json.loads(raw)
    config_raw = args.config.read_bytes()
    config = json.loads(config_raw)
    rows = []
    routed = 0
    passthrough = 0
    routed_bytes = 0
    passthrough_bytes = 0
    coordinate_set: set[tuple[int, int, str]] = set()
    for name, record in sorted(source["entries"].items()):
        coordinate = routed_coordinate(name)
        common = {
            "name": name,
            "source_shard": record["shard"],
            "source_dtype": record["dtype"],
            "source_shape": record["shape"],
            "source_bytes": record["nbytes"],
            "source_payload_sha256": record["payload_sha256"],
        }
        if coordinate is None:
            common.update(action="carry_byte_exact", output_payload_sha256=record["payload_sha256"])
            passthrough += 1
            passthrough_bytes += record["nbytes"]
        else:
            layer, expert, projection = coordinate
            coordinate_set.add(coordinate)
            common.update(
                action="encode_k3_tp3_rotating_uneven",
                geometry_id=GEOMETRY_ID,
                layer=layer,
                expert=expert,
                projection=projection,
                rank_geometry=layer_geometry(layer)["ranks"],
            )
            routed += 1
            routed_bytes += record["nbytes"]
        rows.append(common)
    expected_coordinates = {
        (layer, expert, projection)
        for layer in ROUTED_LAYERS
        for expert in range(256)
        for projection in PROJECTIONS
    }
    missing = sorted(expected_coordinates - coordinate_set)
    unexpected = sorted(coordinate_set - expected_coordinates)
    model_contract = {
        "architectures": config.get("architectures"),
        "normal_hidden_layers": config.get("num_hidden_layers"),
        "next_token_prediction_layers": config.get("num_nextn_predict_layers"),
        "hidden_size": config.get("hidden_size"),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "routed_experts": config.get("n_routed_experts"),
        "experts_per_token": config.get("num_experts_per_tok"),
        "shared_experts": config.get("n_shared_experts"),
        "index_top_k": config.get("index_topk"),
        "index_frequency": config.get("index_topk_freq"),
    }
    expected_contract = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "normal_hidden_layers": 78,
        "next_token_prediction_layers": 1,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "routed_experts": 256,
        "experts_per_token": 8,
        "shared_experts": 1,
        "index_top_k": 2048,
        "index_frequency": 4,
    }
    contract_passed = model_contract == expected_contract
    result = {
        "schema": "glm53-full-exl3-tp3.tensor-plan.v2",
        "geometry_id": GEOMETRY_ID,
        "checkpoint_geometry": checkpoint_geometry(),
        "source_inventory": str(args.source_inventory.resolve()),
        "source_inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "model_contract": model_contract,
        "model_contract_passed": contract_passed,
        "routed_layers": sorted(ROUTED_LAYERS),
        "includes_mtp_layer": 78 in {row[0] for row in coordinate_set},
        "routed_experts_per_layer": 256,
        "routed_top_k": 8,
        "index_top_k": 2048,
        "index_frequency": 4,
        "routed_tensor_count": routed,
        "routed_source_bytes": routed_bytes,
        "passthrough_tensor_count": passthrough,
        "passthrough_source_bytes": passthrough_bytes,
        "total_tensor_count": len(rows),
        "total_source_bytes": routed_bytes + passthrough_bytes,
        "missing_routed_coordinates": missing,
        "unexpected_routed_coordinates": unexpected,
        "passed": contract_passed and not missing and not unexpected and routed == 76 * 256 * 3,
        "tensors": rows,
    }
    atomic_json(args.output, result)
    if not result["passed"]:
        raise SystemExit("tensor disposition audit failed")


if __name__ == "__main__":
    main()
