#!/usr/bin/env python3
"""Verify the exact three-rank construction-time model census."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


GIB = 1 << 30
EXPECTED_K3_BYTES = 91_383_628_800
EXPECTED_K3_BYTES_PER_FULL_LAYER = 3_655_345_152
EXPECTED_GEOMETRY = "rotating-uneven-768-640-640-v1"


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
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--placement", choices=("tp3", "tp3-lazy", "pp3"), default="tp3")
    args = parser.parse_args()
    errors: list[str] = []
    ranks: list[dict] = []
    for rank in range(3):
        profile_path = args.state / "ranks" / f"rank-{rank}.profile.json"
        inspect_path = args.state / "ranks" / f"rank-{rank}.inspect.json"
        metrics_path = args.state / "ranks" / f"rank-{rank}.metrics.csv"
        log_path = args.logs / f"rank-{rank}.log"
        if not all(path.is_file() for path in (profile_path, inspect_path, metrics_path, log_path)):
            errors.append(f"rank {rank}: evidence file missing")
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        inspected = json.loads(inspect_path.read_text(encoding="utf-8"))[0]
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        log = log_path.read_text(encoding="utf-8", errors="replace")
        minimum = min((int(row["host_available_bytes"]) for row in metrics), default=0)
        categories = profile.get("parameter_bytes_by_category", {})
        layers = profile.get("parameter_bytes_by_layer", {})
        if not profile.get("passed"):
            errors.append(f"rank {rank}: profile did not pass")
        if profile.get("rank") != rank or profile.get("world_size") != 3:
            errors.append(f"rank {rank}: rank/world metadata differs")
        if profile.get("placement") != args.placement:
            errors.append(f"rank {rank}: placement metadata differs")
        if profile.get("geometry_id") != EXPECTED_GEOMETRY:
            errors.append(f"rank {rank}: geometry metadata differs")
        if args.placement in ("tp3", "tp3-lazy"):
            start_layer, end_layer = 0, 78
            expected_k3 = EXPECTED_K3_BYTES if args.placement == "tp3" else 0
            expected_parallel = (3, 1)
        else:
            start_layer, end_layer = ((0, 26), (26, 52), (52, 78))[rank]
            expected_k3 = max(0, end_layer - max(3, start_layer)) * EXPECTED_K3_BYTES_PER_FULL_LAYER
            expected_parallel = (1, 3)
        if (profile.get("tensor_parallel_size"), profile.get("pipeline_parallel_size")) != expected_parallel:
            errors.append(f"rank {rank}: TP/PP metadata differs")
        if (profile.get("start_layer"), profile.get("end_layer")) != (start_layer, end_layer):
            errors.append(f"rank {rank}: allocated layer bounds differ")
        actual_k3 = categories.get("k3_routed_experts", 0)
        if actual_k3 != expected_k3:
            errors.append(
                f"rank {rank}: K3 bytes {actual_k3} "
                f"!= {expected_k3}"
            )
        if any(str(layer) not in layers for layer in range(max(3, start_layer), end_layer)):
            errors.append(f"rank {rank}: one or more target layers missing")
        if any(str(layer) in layers for layer in range(3, 78) if not start_layer <= layer < end_layer):
            errors.append(f"rank {rank}: nonlocal target layer was allocated")
        if "78" in layers:
            errors.append(f"rank {rank}: disabled MTP layer 78 was allocated")
        if inspected["State"]["ExitCode"] != 0 or inspected["State"]["OOMKilled"]:
            errors.append(f"rank {rank}: container exit/OOM state differs")
        if minimum < 12 * GIB:
            errors.append(f"rank {rank}: minimum host reserve {minimum} below 12 GiB")
        markers = [
            "GLM53_META_MODEL_PROFILE_OK",
            "GLM53_META_PROFILE_RUNNER_COMPLETE",
            "GLM53_FULL_TP3_PADDING_APPLIED",
        ]
        if args.placement == "tp3":
            markers.append("GLM53_EXL3_TP3_UNEVEN_RANK_SLICES_PATCH_INSTALLED")
        elif args.placement == "tp3-lazy":
            markers.append("GLM53_EXL3_LAZY_K3_PATCH_INSTALLED")
        else:
            markers.append("GLM53_EXL3_PP3_ALL_SLICES_PATCH_INSTALLED")
        for marker in markers:
            if marker not in log:
                errors.append(f"rank {rank}: log marker missing: {marker}")
        ranks.append(
            {
                "rank": rank,
                "parameter_bytes": profile.get("parameter_bytes"),
                "buffer_bytes": profile.get("buffer_bytes"),
                "parameter_count": profile.get("parameter_count"),
                "minimum_host_available_bytes": minimum,
                "parameter_bytes_by_category": categories,
            }
        )
    if args.placement in ("tp3", "tp3-lazy") and len(ranks) == 3 and len({row["parameter_bytes"] for row in ranks}) != 1:
        errors.append("final parameter bytes differ across ranks")
    audit_path = args.state / "kernel-audit.json"
    if not audit_path.is_file() or not json.loads(audit_path.read_text(encoding="utf-8")).get("passed"):
        errors.append("strict kernel audit missing or failed")
    summary = {
        "schema": "glm53-full-exl3-tp3.meta-model-profile-summary.v1",
        "passed": not errors,
        "errors": errors,
        "expected_k3_bytes_per_rank": EXPECTED_K3_BYTES,
        "placement": args.placement,
        "ranks": ranks,
    }
    atomic_json(args.output, summary)
    if errors:
        raise SystemExit("meta model profile failed: " + "; ".join(errors[:8]))


if __name__ == "__main__":
    main()
