#!/usr/bin/env python3
"""Uneven-width K3 codec, slicing, pack, and LinearEXL3 qualification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("exl3", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exl3"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numeric-core", type=Path, required=True)
    parser.add_argument("--numeric-core-sha256", required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--runtime-exl3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from r7_encoder.r10_codec import R10TrellisCodec
    from r7_encoder.trellis import CodecConfig
    from tp3k3.encoder import (
        IdentityMetric,
        IdentityR10Codec,
        LocalTensorId,
        deterministic_vector,
        encode_projection_rank,
        pack_unpack_oracle,
        physical_slice,
        tensor_sha256,
    )
    from tp3k3.geometry import GEOMETRY_ID, slice_spec

    torch.use_deterministic_algorithms(True)
    config = CodecConfig(
        device="cuda:0",
        sigma_reg=0.025,
        numeric_core=args.numeric_core,
        numeric_core_sha256=args.numeric_core_sha256,
        extension=args.extension,
        extension_sha256=args.extension_sha256,
        verify_files=True,
    )
    codec = IdentityR10Codec(config)
    generator = torch.Generator(device="cpu").manual_seed(20260903)
    results = {
        "schema": "glm53-full-exl3-tp3.synthetic-codec-gate.v2",
        "geometry_id": GEOMETRY_ID,
        "target_geometry": {},
        "identity_factor_equivalence": {},
        "determinism": {},
        "runtime": {},
    }

    # Prove that the identity specialization emits the same bytes as the full
    # R10 covariance path on a complete 128x128 trellis tile domain.
    small = torch.randn((128, 128), generator=generator, dtype=torch.float32)
    tid = LocalTensorId("identity-equivalence", 128, 128, 3, 0, "gate_proj")
    suh = deterministic_vector(128, 3, 0, "gate_proj", 0, "suh")
    svh = deterministic_vector(128, 3, 0, "gate_proj", 0, "svh")
    full = R10TrellisCodec(config).encode(
        tensor_id=tid,
        weight_hf=small,
        covariance=torch.eye(128, dtype=torch.float32),
        bits=3,
        suh=suh,
        svh=svh,
        sigma_reg=0.025,
        provenance={"test": "full-identity"},
    )
    fast = codec.encode(
        tensor_id=tid,
        weight_hf=small,
        covariance=IdentityMetric(128),
        bits=3,
        suh=suh,
        svh=svh,
        sigma_reg=0.025,
        provenance={"test": "specialized-identity"},
    )
    if not torch.equal(full.trellis, fast.trellis):
        raise AssertionError("identity factor specialization changed packed K3 bytes")
    results["identity_factor_equivalence"] = {
        "passed": True,
        "packed_sha256": tensor_sha256(fast.trellis),
        "full_equals_specialized": True,
    }
    again = codec.encode(
        tensor_id=tid,
        weight_hf=small,
        covariance=IdentityMetric(128),
        bits=3,
        suh=suh,
        svh=svh,
        sigma_reg=0.025,
        provenance={"test": "repeat"},
    )
    if not torch.equal(fast.trellis, again.trellis):
        raise AssertionError("repeat identity encode differs")
    results["determinism"] = {
        "passed": True,
        "repeat_packed_sha256": tensor_sha256(again.trellis),
        "seed_vectors_equal": suh == deterministic_vector(128, 3, 0, "gate_proj", 0, "suh"),
    }

    source_weights = {
        "gate_proj": torch.randn((2048, 6144), generator=generator, dtype=torch.float32).bfloat16(),
        "up_proj": torch.randn((2048, 6144), generator=generator, dtype=torch.float32).bfloat16(),
        "down_proj": torch.randn((6144, 2048), generator=generator, dtype=torch.float32).bfloat16(),
    }
    # Layer 3 rank 0 is 768 wide and rank 1 is 640 wide. Exercise every
    # projection at both supported widths using exact semantic slices.
    cases = tuple(
        (projection, rank, source_weights[projection])
        for rank in (0, 1)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    encoded_cases = []
    for projection, rank, weight in cases:
        start = time.monotonic()
        layer = 3
        spec = slice_spec(layer, 0, projection, rank)
        sliced = physical_slice(weight, projection, layer, rank)
        direct = (
            weight[:, spec.semantic_start : spec.semantic_stop].contiguous()
            if projection == "down_proj"
            else weight[spec.semantic_start : spec.semantic_stop, :].contiguous()
        )
        if not torch.equal(sliced, direct):
            raise AssertionError("gate/up/down physical slicing differs")
        tensors, record = encode_projection_rank(codec, weight, layer=layer, expert=0, projection=projection, rank=rank)
        if tuple(record["slice"]["physical_shape"]) != tuple(spec.physical_shape):
            raise AssertionError("target physical geometry differs")
        record["wall_seconds"] = time.monotonic() - start
        if not record["padding_exact_zero"] or record["slice"]["pad_channels"] != 0:
            raise AssertionError("target geometry unexpectedly contains padding")
        record["direct_slice_sha256"] = tensor_sha256(direct)
        results["target_geometry"][f"width{spec.physical_channels}.{projection}"] = record
        encoded_cases.append((projection, rank, tensors, record))

    runtime = load_runtime(args.runtime_exl3)
    for projection, rank, tensors, record in encoded_cases:
        k = int(tensors["suh"].numel())
        n = int(tensors["svh"].numel())
        reconstructed = codec.decode_to_original(
            tensors["trellis"].cuda(), tensors["suh"], tensors["svh"], 3
        ).half()
        xgen = torch.Generator(device="cpu").manual_seed(1000 + rank + len(projection))
        x = torch.randn((5, k), generator=xgen, dtype=torch.float32, device="cpu").half().cuda()
        actual = runtime.execute_exl3_linear(
            x,
            tensors["trellis"].cuda(),
            tensors["suh"].cuda(),
            tensors["svh"].cuda(),
            tensors["mcg"].cuda(),
            out_dtype=torch.float32,
        )
        reference = x.float() @ reconstructed.float()
        delta = actual - reference
        relative = float(delta.norm().item() / max(reference.norm().item(), 1e-30))
        if not math.isfinite(relative) or relative > 0.01:
            raise AssertionError(f"LinearEXL3 reconstruction parity failed for {projection}: {relative}")
        row = {
            "relative_l2": relative,
            "max_abs": float(delta.abs().max().item()),
            "width": record["slice"]["physical_channels"],
            "unpadded": record["slice"]["pad_channels"] == 0,
        }
        results["runtime"][f"width{record['slice']['physical_channels']}.{projection}"] = row
    if {row["width"] for row in results["runtime"].values()} != {640, 768}:
        raise AssertionError("both uneven slice widths were not qualified")
    results["passed"] = True
    results["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(args.output), "peak": results["cuda_peak_allocated_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
