#!/usr/bin/env python3
"""Project optimistic rotating-uneven TP3 checkpoint and 32K residency."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import GEOMETRY_ID, ROUTED_LAYERS, rank_widths


GIB = 1 << 30
_RANK_FILE = re.compile(r"rank-(?P<rank>[0-2])\.metrics\.csv$")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-dir", type=Path, required=True)
    parser.add_argument("--tensor-plan", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--reserve-gib", type=int, default=12)
    args = parser.parse_args()

    tensor_plan = json.loads(args.tensor_plan.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    layer = json.loads((args.layer_dir / "LAYER_RECEIPT.json").read_text(encoding="utf-8"))
    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.layer_dir / "receipts").glob("expert-*.json"))
    ]
    if (
        len(receipts) != 256
        or layer.get("schema") != "glm53-full-exl3-tp3.layer-receipt.v2"
        or layer.get("geometry_id") != GEOMETRY_ID
        or not layer.get("passed")
        or not all(row.get("passed") and row.get("geometry_id") == GEOMETRY_ID for row in receipts)
    ):
        raise SystemExit("capacity projection requires one sealed uneven 256-expert layer")

    qualified_layer = int(layer["layer"])
    rank_payload = {rank: 0 for rank in range(3)}
    for receipt in receipts:
        for record in receipt["slices"]:
            rank = int(record["slice"]["rank"])
            rank_payload[rank] += sum(component["bytes"] for component in record["components"].values())
    payload_samples: dict[int, set[int]] = {640: set(), 768: set()}
    for rank, payload in rank_payload.items():
        payload_samples[rank_widths(qualified_layer)[rank]].add(payload)
    if any(len(values) != 1 for values in payload_samples.values()):
        raise SystemExit(f"same-width physical payloads differ: {payload_samples}")
    payload_by_width = {width: next(iter(values)) for width, values in payload_samples.items()}

    routed_rank_bytes = {
        rank: sum(payload_by_width[rank_widths(layer_number)[rank]] for layer_number in ROUTED_LAYERS)
        for rank in range(3)
    }
    total_component_bytes = sum(rank_payload.values())
    qualified_layer_container_overhead = int(layer["output_bytes"]) - total_component_bytes
    if qualified_layer_container_overhead < 0:
        raise SystemExit("qualified layer output is smaller than its tensor payload")
    projected_routed_component_bytes = sum(routed_rank_bytes.values())
    projected_routed_container_overhead = qualified_layer_container_overhead * len(ROUTED_LAYERS)

    metrics_by_rank: dict[int, list[dict]] = {rank: [] for rank in range(3)}
    metric_paths = sorted(args.runtime_gate_dir.glob("rotations/layer-*/ranks/rank-*.metrics.csv"))
    for path in metric_paths:
        match = _RANK_FILE.search(path.name)
        if match is None:
            continue
        rank = int(match.group("rank"))
        with path.open(newline="", encoding="utf-8") as handle:
            values = [int(row["host_available_bytes"]) for row in csv.DictReader(handle)]
        if values:
            metrics_by_rank[rank].append(
                {"file": str(path), "maximum_bytes": max(values), "minimum_bytes": min(values)}
            )
    if any(len(rows) != 3 for rows in metrics_by_rank.values()):
        raise SystemExit("capacity projection requires all three rotations' metrics on every rank")

    passthrough_bytes = int(tensor_plan["passthrough_source_bytes"])
    # Deliberately favorable: passthrough and compressed KV shard perfectly;
    # engine, CUDA, NCCL, allocator, graph, activation and indexer overhead are
    # all zero. A miss remains conclusive; a pass only authorizes bulk encode.
    passthrough_rank_lower_bound = (passthrough_bytes + 2) // 3
    kv_elements = (
        int(config["num_hidden_layers"])
        * args.context_tokens
        * (int(config["kv_lora_rank"]) + int(config["qk_rope_head_dim"]))
    )
    kv_rank_lower_bound = (kv_elements * 2 + 2) // 3
    reserve_required = args.reserve_gib * GIB
    optimistic_idle_by_rank = {
        rank: max(row["maximum_bytes"] for row in metrics_by_rank[rank]) for rank in range(3)
    }
    runtime_required_by_rank = {
        rank: routed_rank_bytes[rank] + passthrough_rank_lower_bound + kv_rank_lower_bound
        for rank in range(3)
    }
    reserve_by_rank = {
        rank: optimistic_idle_by_rank[rank] - runtime_required_by_rank[rank]
        for rank in range(3)
    }
    passed = all(value >= reserve_required for value in reserve_by_rank.values())
    checkpoint_bytes = (
        projected_routed_component_bytes
        + projected_routed_container_overhead
        + passthrough_bytes
    )
    result = {
        "schema": "glm53-full-exl3-tp3.capacity-projection.v2",
        "geometry_id": GEOMETRY_ID,
        "method": "rotating_uneven_optimistic_lower_bound",
        "qualified_layer": qualified_layer,
        "context_tokens": args.context_tokens,
        "checkpoint_projected_bytes": checkpoint_bytes,
        "checkpoint_projected_gib": checkpoint_bytes / GIB,
        "qualified_layer_rank_payload_bytes": rank_payload,
        "measured_layer_payload_by_width_bytes": payload_by_width,
        "qualified_layer_container_overhead_bytes": qualified_layer_container_overhead,
        "projected_routed_component_bytes": projected_routed_component_bytes,
        "projected_routed_container_overhead_bytes": projected_routed_container_overhead,
        "routed_rank_payload_76_layers_bytes": routed_rank_bytes,
        "wide_layer_count_by_rank": {"0": 25, "1": 25, "2": 26},
        "passthrough_total_bytes": passthrough_bytes,
        "passthrough_rank_perfect_shard_lower_bound_bytes": passthrough_rank_lower_bound,
        "kv_rank_perfect_shard_lower_bound_bytes": kv_rank_lower_bound,
        "runtime_rank_required_lower_bound_bytes": runtime_required_by_rank,
        "runtime_gate_idle_available": metrics_by_rank,
        "optimistic_idle_available_bytes": optimistic_idle_by_rank,
        "optimistic_reserve_bytes": reserve_by_rank,
        "optimistic_reserve_gib": {rank: value / GIB for rank, value in reserve_by_rank.items()},
        "minimum_optimistic_reserve_bytes": min(reserve_by_rank.values()),
        "minimum_optimistic_reserve_gib": min(reserve_by_rank.values()) / GIB,
        "required_reserve_bytes": reserve_required,
        "ignored_runtime_overheads": [
            "engine",
            "CUDA context",
            "NCCL",
            "allocator",
            "CUDA graphs",
            "activations",
            "indexer",
        ],
        "passed": passed,
    }
    atomic_json(args.output, result)
    if not passed:
        raise SystemExit("rotating uneven optimistic 32K bound misses the 12 GiB reserve")


if __name__ == "__main__":
    main()
