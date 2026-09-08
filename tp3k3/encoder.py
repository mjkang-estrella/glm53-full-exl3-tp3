"""Crash-resumable identity-H R10 encoder for rotating uneven TP3 slices."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

from .geometry import (
    BITS,
    EXPERTS,
    GEOMETRY_ID,
    HIDDEN,
    PHYSICAL_INTERMEDIATE,
    SEMANTIC_INTERMEDIATE,
    SOURCE_REVISION,
    checkpoint_key,
    layer_geometry,
    slice_spec,
)


MCG_SIGNED = -877_912_083
MCG_UNSIGNED = 0xCBAC1FED
MAX_QUALIFICATION_RELATIVE_RMSE = 0.5


@dataclass(frozen=True)
class LocalTensorId:
    key: str
    k: int
    n: int
    layer: int
    expert: int
    projection: str


@dataclass(frozen=True)
class IdentityMetric:
    size: int


def tensor_sha256(value: Any) -> str:
    import torch

    raw = torch.as_tensor(value).detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    from r7_encoder.determinism import atomic_write_json

    atomic_write_json(path, value)
    path.chmod(0o644)


def deterministic_vector(length: int, layer: int, expert: int, projection: str, rank: int, side: str):
    from r7_encoder.rotations import rademacher_vector

    return rademacher_vector(
        length,
        f"glm53-full-exl3-tp3-k3:{GEOMETRY_ID}",
        SOURCE_REVISION,
        layer,
        expert,
        projection,
        rank,
        side,
    )


def deterministic_rotation_seed(layer: int, expert: int, projection: str, rank: int, side: str) -> int:
    from r7_encoder.determinism import derive_seed

    return derive_seed(
        "rotation",
        f"glm53-full-exl3-tp3-k3:{GEOMETRY_ID}",
        SOURCE_REVISION,
        layer,
        expert,
        projection,
        rank,
        side,
        bits=64,
    )


class IdentityR10Codec:
    """R10 numeric path with the exact LDL factor for an identity Hessian.

    Rademacher diagonals and orthonormal block Hadamards leave identity
    invariant.  The Hessian factor is formed from those signs before v3.1's
    absolute weight normalization changes the checkpoint vectors.  After the
    mandatory diagonal removal, the identity-H error-feedback factor is
    exactly zero. Cache one zero factor per K instead of refactorizing the
    same identity matrix 175,104 times.
    """

    def __init__(self, config) -> None:
        import torch
        from r7_encoder.r10_codec import R10TrellisCodec

        class _Codec(R10TrellisCodec):
            def __init__(self, outer, *args, **kwargs):
                self._identity_outer = outer
                super().__init__(*args, **kwargs)

            def _covariance_sha256(self, covariance):
                if isinstance(covariance, IdentityMetric):
                    return hashlib.sha256(f"identity-f32:{covariance.size}".encode()).hexdigest()
                return super()._covariance_sha256(covariance)

            def _factor_covariance_cached(self, covariance, su, bits, sigma_reg, covariance_sha256):
                if not isinstance(covariance, IdentityMetric):
                    return super()._factor_covariance_cached(covariance, su, bits, sigma_reg, covariance_sha256)
                size = int(su.numel())
                if size != covariance.size:
                    raise ValueError("identity metric size differs from tensor K")
                factor = self._identity_outer._zero_factors.get(size)
                if factor is None:
                    factor = torch.zeros((size, size), dtype=torch.float32, device=self.config.device)
                    self._identity_outer._zero_factors[size] = factor
                return factor, self._quant_args(bits, sigma_reg)

        self._zero_factors: dict[int, Any] = {}
        self.codec = _Codec(self, config)

    def encode(self, **kwargs):
        return self.codec.encode(**kwargs)

    def decode_to_original(self, *args, **kwargs):
        return self.codec.decode_to_original(*args, **kwargs)

    def normalized_vectors(self, weight_hf, suh_signs, svh_signs, bits: int):
        """Fit the pinned v3.1 absolute/GSS scales at the FP16 boundary.

        This is the data-free branch of the published encoder: identity H
        selects output scaling, row RMS sets the absolute MCG amplitude, and
        the sealed 13-evaluation GSS scalar is folded back into ``suh``.
        Every transform consumed here is supplied by the hash-pinned numeric
        core.  The returned vectors are exactly the FP16 values consumed by
        both the encoder and loader.
        """
        import torch

        weight_kn = torch.as_tensor(
            weight_hf, dtype=torch.float32, device=self.config.device
        ).T.contiguous()
        k, n = (int(weight_kn.shape[0]), int(weight_kn.shape[1]))
        su_sign = torch.as_tensor(
            suh_signs, dtype=torch.float32, device=self.config.device
        ).flatten()
        sv_sign = torch.as_tensor(
            svh_signs, dtype=torch.float32, device=self.config.device
        ).flatten()
        if tuple(su_sign.shape) != (k,) or tuple(sv_sign.shape) != (n,):
            raise ValueError("normalization sign geometry mismatch")
        if not torch.all((su_sign == 1) | (su_sign == -1)) or not torch.all(
            (sv_sign == 1) | (sv_sign == -1)
        ):
            raise ValueError("normalization requires exact Rademacher signs")

        # Identity H has uniform diagonal, so the published skew rule enables
        # output-channel scaling.  Round at each real checkpoint boundary.
        output_scale = self.core.block_rms(
            weight_kn, dim=0, keepdim=True
        ).flatten().float()
        output_mean = float(output_scale.mean().item())
        if not math.isfinite(output_mean) or output_mean <= 1e-30:
            raise ValueError("source slice has degenerate output RMS")
        output_scale.div_(output_mean)
        zero_outputs = output_scale <= 1e-30
        output_scale[zero_outputs] = 0.1
        stored_sv = (sv_sign * output_scale).half()
        # A nonzero placeholder keeps the codec transform invertible for any
        # genuine all-zero source channel. The uneven layout itself is unpadded.
        stored_sv[zero_outputs] = 1.0

        right = weight_kn.clone()
        right.div_(stored_sv.float().unsqueeze(0))
        self.core.blockwise_preapply_had_r_(right, 128)
        row_rms = self.core.block_rms(right, dim=1, keepdim=True).flatten().float()
        zero_rows = row_rms <= 1e-30
        row_rms[zero_rows] = 0.1
        codebook_factor = -float(self.codec.codebook_scale)
        pre_gss_su = (su_sign * row_rms / codebook_factor).half()
        if not torch.isfinite(pre_gss_su).all() or (pre_gss_su == 0).any():
            raise ValueError("absolute suh is not representable as finite nonzero FP16")

        target = right
        target.div_(pre_gss_su.float().unsqueeze(1))
        self.core.blockwise_preapply_had_l_(target, 128)
        quant_args = self.codec._quant_args(bits, self.config.sigma_reg)
        g_scale, _objective = self.core.g_scale_gss(target, quant_args)
        g_scale = float(g_scale)
        if not math.isfinite(g_scale) or g_scale <= 0:
            raise ValueError("sealed GSS returned a non-positive scale")
        stored_su = (pre_gss_su.float() / g_scale).half()
        if not torch.isfinite(stored_su).all() or (stored_su == 0).any():
            raise ValueError("GSS-folded suh is not representable as finite nonzero FP16")

        return stored_su.detach().cpu(), stored_sv.detach().cpu(), {
            "schema": "glm53-full-exl3-tp3.v31-absolute-normalization.v1",
            "identity_h": True,
            "apply_output_scales": True,
            "codebook_scale": float(self.codec.codebook_scale),
            "g_scale": g_scale,
            "gss_algorithm": "encode_tr3_v31.g_scale_gss",
            "gss_evaluations": 13,
            "zero_output_channels": int(zero_outputs.sum().item()),
            "zero_input_rows": int(zero_rows.sum().item()),
            "stored_suh_sha256": tensor_sha256(stored_su),
            "stored_svh_sha256": tensor_sha256(stored_sv),
            "stored_suh_abs_min": float(stored_su.abs().min().item()),
            "stored_suh_abs_max": float(stored_su.abs().max().item()),
            "stored_svh_abs_min": float(stored_sv.abs().min().item()),
            "stored_svh_abs_max": float(stored_sv.abs().max().item()),
        }

    @property
    def core(self):
        return self.codec.core

    @property
    def config(self):
        return self.codec.config


class StagedSource:
    def __init__(self, checkpoint: Path, inventory: Path) -> None:
        self.checkpoint = checkpoint.resolve()
        self.inventory_path = inventory.resolve()
        self.inventory = json.loads(inventory.read_text(encoding="utf-8"))
        self.entries = self.inventory["entries"]
        self.index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
        if file_sha256(checkpoint / "model.safetensors.index.json") != self.inventory["index_sha256"]:
            raise ValueError("staged source index differs from sealed BF16 inventory")

    @staticmethod
    def name(layer: int, expert: int, projection: str) -> str:
        return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"

    def tensor(self, layer: int, expert: int, projection: str):
        import torch
        from safetensors import safe_open

        name = self.name(layer, expert, projection)
        record = self.entries[name]
        expected_shape = (HIDDEN, SEMANTIC_INTERMEDIATE) if projection == "down_proj" else (SEMANTIC_INTERMEDIATE, HIDDEN)
        if record["dtype"] != "BF16" or tuple(record["shape"]) != expected_shape:
            raise ValueError(f"sealed source geometry differs for {name}")
        shard_name = record["shard"]
        if self.index["weight_map"].get(name) != shard_name:
            raise ValueError(f"source index mapping differs for {name}")
        shard = (self.checkpoint / shard_name).resolve()
        if self.checkpoint not in shard.parents or not shard.is_file() or shard.is_symlink():
            raise ValueError(f"invalid staged shard for {name}: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name).contiguous()
        if value.dtype != torch.bfloat16 or tuple(value.shape) != expected_shape:
            raise ValueError(f"loaded source geometry differs for {name}")
        digest = tensor_sha256(value)
        if digest != record["payload_sha256"]:
            raise ValueError(f"source payload hash differs for {name}")
        return value, {"name": name, "shard": shard_name, "bytes": record["nbytes"], "sha256": digest}


def physical_slice(weight, projection: str, layer: int, rank: int):
    spec = slice_spec(layer, 0, projection, rank)
    start = spec.semantic_start
    stop = spec.semantic_stop
    if projection == "down_proj":
        return weight[:, start:stop].contiguous()
    return weight[start:stop, :].contiguous()


def pack_unpack_oracle(codec: IdentityR10Codec, packed, k: int, n: int, bits: int) -> str:
    import torch

    device_packed = packed.to(codec.config.device)
    unpacked = torch.empty((k // 16, n // 16, 256), dtype=torch.int16, device=codec.config.device)
    _torch_module, extension = codec.core._lazy_torch()
    extension.unpack_trellis(unpacked, device_packed, bits)
    repacked = codec.core.pack_trellis(unpacked, codec.codec._quant_args(bits, 0.025))
    if not torch.equal(repacked, device_packed):
        raise AssertionError("TRELLIS pack/unpack/repack equality failed")
    return tensor_sha256(unpacked)


def gpu_metrics() -> dict[str, Any]:
    query = "temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used"
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        temp, power, graphics, memory, used = [part.strip() for part in raw.split(",")]
        optional_int = lambda value: None if value in {"[N/A]", "N/A", ""} else int(value)
        return {"temperature_c": float(temp), "power_w": float(power), "graphics_clock_mhz": optional_int(graphics), "memory_clock_mhz": optional_int(memory), "reported_memory_mib": optional_int(used)}
    except Exception as exc:
        return {"error": repr(exc)}


def process_metrics() -> dict[str, int]:
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, raw, _unit = line.split()
            values[name.rstrip(":") + "_kib"] = int(raw)
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            values["host_available_kib"] = int(line.split()[1])
    for line in Path("/proc/self/io").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"rchar", "wchar", "read_bytes", "write_bytes", "cancelled_write_bytes"}:
            values[f"process_io_{name}"] = int(raw.strip())
    return values


class ExpertMetricSampler:
    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="expert-metrics", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append({"unix": time.time(), "gpu": gpu_metrics(), "process": process_metrics()})
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        import torch

        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval * 2))
        self.samples.append({"unix": time.time(), "gpu": gpu_metrics(), "process": process_metrics()})
        gpu_rows = [row["gpu"] for row in self.samples if "error" not in row["gpu"]]
        process_rows = [row["process"] for row in self.samples]
        return {
            "sample_interval_seconds": self.interval,
            "samples": self.samples,
            "summary": {
                "process_vm_hwm_peak_kib": max((row.get("VmHWM_kib", 0) for row in process_rows), default=0),
                "process_vm_rss_peak_kib": max((row.get("VmRSS_kib", 0) for row in process_rows), default=0),
                "host_available_min_kib": min((row.get("host_available_kib", 1 << 62) for row in process_rows), default=0),
                "cuda_allocator_peak_bytes": int(torch.cuda.max_memory_allocated()),
                "temperature_peak_c": max((row["temperature_c"] for row in gpu_rows), default=None),
                "power_peak_w": max((row["power_w"] for row in gpu_rows), default=None),
                "graphics_clock_min_mhz": min((row["graphics_clock_mhz"] for row in gpu_rows if row["graphics_clock_mhz"] is not None), default=None),
                "graphics_clock_max_mhz": max((row["graphics_clock_mhz"] for row in gpu_rows if row["graphics_clock_mhz"] is not None), default=None),
                "memory_clock_min_mhz": min((row["memory_clock_mhz"] for row in gpu_rows if row["memory_clock_mhz"] is not None), default=None),
                "memory_clock_max_mhz": max((row["memory_clock_mhz"] for row in gpu_rows if row["memory_clock_mhz"] is not None), default=None),
            },
        }


def _component_record(value) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.numel() * value.element_size(), "sha256": tensor_sha256(value)}


def encode_projection_rank(codec: IdentityR10Codec, weight, *, layer: int, expert: int, projection: str, rank: int):
    import torch

    spec = slice_spec(layer, expert, projection, rank)
    sliced = physical_slice(weight, projection, layer, rank)
    if tuple(sliced.shape) != spec.physical_shape or spec.pad_channels != 0:
        raise AssertionError(f"unpadded slice geometry differs for {spec.key}")
    k = int(sliced.shape[1])
    n = int(sliced.shape[0])
    tensor_id = LocalTensorId(spec.key, k, n, layer, expert, projection)
    suh_signs = deterministic_vector(k, layer, expert, projection, rank, "suh")
    svh_signs = deterministic_vector(n, layer, expert, projection, rank, "svh")
    suh, svh, normalization = codec.normalized_vectors(
        sliced, suh_signs, svh_signs, BITS
    )
    rotation_seeds = {
        "suh": deterministic_rotation_seed(layer, expert, projection, rank, "suh"),
        "svh": deterministic_rotation_seed(layer, expert, projection, rank, "svh"),
    }
    start = time.monotonic()
    encoded = codec.encode(
        tensor_id=tensor_id,
        weight_hf=sliced,
        covariance=IdentityMetric(k),
        bits=BITS,
        suh=suh,
        svh=svh,
        sigma_reg=0.025,
        provenance={"identity_h": True, "coordinate_seed": spec.seed, "rotation_seeds": rotation_seeds, "slice": spec.to_dict(), "normalization": normalization},
    )
    elapsed = time.monotonic() - start
    unpacked_sha = pack_unpack_oracle(codec, encoded.trellis, k, n, BITS)
    stored_suh = encoded.suh.clone()
    stored_svh = encoded.svh.clone()
    if spec.pad_channels or spec.scale_mask is not None:
        raise AssertionError("rotating uneven layout must not declare padding")
    reconstructed_kn = codec.decode_to_original(
        encoded.trellis.to(codec.config.device), stored_suh, stored_svh, BITS
    )
    target_kn = sliced.to(device=codec.config.device, dtype=torch.float32).T.contiguous()
    if projection == "down_proj":
        semantic_recon = reconstructed_kn[: spec.semantic_channels, :]
        semantic_target = target_kn[: spec.semantic_channels, :]
        padded = reconstructed_kn[spec.semantic_channels :, :]
    else:
        semantic_recon = reconstructed_kn[:, : spec.semantic_channels]
        semantic_target = target_kn[:, : spec.semantic_channels]
        padded = reconstructed_kn[:, spec.semantic_channels :]
    difference = semantic_recon.double() - semantic_target.double()
    numerator = float(difference.square().sum().item())
    denominator = float(semantic_target.double().square().sum().item())
    pad_max = float(padded.abs().max().item()) if padded.numel() else 0.0
    if pad_max != 0.0:
        raise AssertionError(f"padding leakage for {spec.key}: {pad_max}")
    if not torch.isfinite(reconstructed_kn).all():
        raise AssertionError(f"non-finite reconstruction for {spec.key}")
    marker = torch.tensor([MCG_SIGNED], dtype=torch.int32)
    tensors = {"trellis": encoded.trellis, "suh": stored_suh, "svh": stored_svh, "mcg": marker}
    record = {
        "slice": spec.to_dict(),
        "bits": BITS,
        "identity_h": True,
        "sigma_reg": 0.025,
        "rotation_seeds": rotation_seeds,
        "normalization": normalization,
        "elapsed_seconds": elapsed,
        "error_squared": numerator,
        "source_squared": denominator,
        "relative_squared_error": numerator / max(denominator, 1e-300),
        "relative_rmse": math.sqrt(numerator / max(denominator, 1e-300)),
        "max_abs_error": float(difference.abs().max().item()),
        "padding_max_abs": pad_max,
        "padding_exact_zero": pad_max == 0.0,
        "unpacked_indices_sha256": unpacked_sha,
        "runtime_reconstruction_fp16_sha256": tensor_sha256(reconstructed_kn.half()),
        "components": {name: _component_record(value) for name, value in tensors.items()},
    }
    del reconstructed_kn, target_kn, difference, sliced
    return tensors, record


def load_receipt_if_valid(shard: Path, receipt_path: Path, layer: int, expert: int, codec: IdentityR10Codec) -> dict[str, Any] | None:
    if not shard.is_file() or not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "glm53-full-exl3-tp3.expert-receipt.v2" or receipt.get("layer") != layer or receipt.get("expert") != expert:
            return None
        expected_identity = {
            "implementation_sha256": file_sha256(Path(__file__)),
            "numeric_core_sha256": codec.config.numeric_core_sha256,
            "extension_sha256": codec.config.extension_sha256,
            "factorization_policy": codec.config.factorization_policy,
            "identity_hessian": True,
            "absolute_normalization": "v31-gss-fp16-boundary-v1",
        }
        if receipt.get("source_revision") != SOURCE_REVISION or receipt.get("encoder_identity") != expected_identity:
            return None
        if (
            receipt.get("bits") != BITS
            or receipt.get("tp") != 3
            or receipt.get("geometry_id") != GEOMETRY_ID
            or receipt.get("layer_geometry") != layer_geometry(layer)
        ):
            return None
        if shard.stat().st_size != receipt["output_bytes"] or file_sha256(shard) != receipt["output_sha256"]:
            return None
        return receipt
    except Exception:
        return None


def encode_expert(codec: IdentityR10Codec, source: StagedSource, output: Path, layer: int, expert: int) -> dict[str, Any]:
    import torch
    from r7_encoder.safetensors_io import torch_tensor_entry, write_safetensors_atomic

    expert_dir = output / f"layer-{layer:03d}" / "experts"
    receipt_dir = output / f"layer-{layer:03d}" / "receipts"
    shard = expert_dir / f"expert-{expert:03d}.safetensors"
    receipt_path = receipt_dir / f"expert-{expert:03d}.json"
    prior = load_receipt_if_valid(shard, receipt_path, layer, expert, codec)
    if prior is not None:
        return prior
    expert_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    sampler = ExpertMetricSampler()
    sampler.start()
    before = {"gpu": gpu_metrics(), "process": process_metrics()}
    entries = []
    slices = []
    sources = {}
    error_squared = 0.0
    source_squared = 0.0
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weight, source_record = source.tensor(layer, expert, projection)
        sources[projection] = source_record
        for rank in range(3):
            tensors, record = encode_projection_rank(codec, weight, layer=layer, expert=expert, projection=projection, rank=rank)
            for suffix, value in tensors.items():
                entries.append(torch_tensor_entry(checkpoint_key(layer, expert, projection, rank, suffix), value))
            slices.append(record)
            error_squared += record["error_squared"]
            source_squared += record["source_squared"]
            del tensors
            torch.cuda.empty_cache()
        del weight
    metadata = {
        "schema": "glm53-full-exl3-tp3.expert-shard.v2",
        "geometry_id": GEOMETRY_ID,
        "layer": str(layer),
        "expert": str(expert),
        "bits": str(BITS),
        "tp": "3",
        "rank_widths": ",".join(str(item["width"]) for item in layer_geometry(layer)["ranks"]),
        "rank_offsets": ",".join(str(item["offset"]) for item in layer_geometry(layer)["ranks"]),
    }
    _payload_hashes, output_hash = write_safetensors_atomic(shard, entries, metadata=metadata)
    shard.chmod(0o644)
    after = {"gpu": gpu_metrics(), "process": process_metrics()}
    sampled_metrics = sampler.finish()
    receipt = {
        "schema": "glm53-full-exl3-tp3.expert-receipt.v2",
        "source_revision": SOURCE_REVISION,
        "geometry_id": GEOMETRY_ID,
        "layer": layer,
        "expert": expert,
        "bits": BITS,
        "tp": 3,
        "layer_geometry": layer_geometry(layer),
        "semantic_intermediate": SEMANTIC_INTERMEDIATE,
        "physical_intermediate": PHYSICAL_INTERMEDIATE,
        "padding_channels": 0,
        "encoder_identity": {
            "implementation_sha256": file_sha256(Path(__file__)),
            "numeric_core_sha256": codec.config.numeric_core_sha256,
            "extension_sha256": codec.config.extension_sha256,
            "factorization_policy": codec.config.factorization_policy,
            "identity_hessian": True,
            "absolute_normalization": "v31-gss-fp16-boundary-v1",
        },
        "source_tensors": sources,
        "source_bytes": sum(item["bytes"] for item in sources.values()),
        "output_file": shard.name,
        "output_bytes": shard.stat().st_size,
        "output_sha256": output_hash,
        "started_unix": started_wall,
        "elapsed_seconds": time.monotonic() - started,
        "error_squared": error_squared,
        "source_squared": source_squared,
        "relative_squared_error": error_squared / max(source_squared, 1e-300),
        "relative_rmse": math.sqrt(error_squared / max(source_squared, 1e-300)),
        "quality_sanity_gate": {
            "metric": "aggregate_weight_relative_rmse",
            "maximum": MAX_QUALIFICATION_RELATIVE_RMSE,
        },
        "measurements": {"before": before, "after": after, **sampled_metrics},
        "slices": slices,
        "passed": all(
            item["padding_exact_zero"]
            and item["slice"]["pad_channels"] == 0
            and item["slice"]["physical_channels"] == item["slice"]["semantic_channels"]
            for item in slices
        )
        and math.sqrt(error_squared / max(source_squared, 1e-300))
        <= MAX_QUALIFICATION_RELATIVE_RMSE,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def parse_experts(raw: str) -> list[int]:
    values: set[int] = set()
    for part in raw.split(","):
        if "-" in part:
            start, stop = map(int, part.split("-", 1))
            values.update(range(start, stop + 1))
        else:
            values.add(int(part))
    if not values or min(values) < 0 or max(values) >= EXPERTS:
        raise ValueError("experts must be in 0..255")
    return sorted(values)


def seal_layer(output: Path, layer: int, receipts: Iterable[dict[str, Any]]) -> Path:
    receipts = list(receipts)
    if len(receipts) != EXPERTS or {item["expert"] for item in receipts} != set(range(EXPERTS)):
        raise ValueError("complete layer seal requires all 256 experts")
    layer_dir = output / f"layer-{layer:03d}"
    result = {
        "schema": "glm53-full-exl3-tp3.layer-receipt.v2",
        "geometry_id": GEOMETRY_ID,
        "layer_geometry": layer_geometry(layer),
        "layer": layer,
        "experts": EXPERTS,
        "slice_encodes": EXPERTS * 3 * 3,
        "bits": BITS,
        "source_bytes": sum(item["source_bytes"] for item in receipts),
        "output_bytes": sum(item["output_bytes"] for item in receipts),
        "elapsed_expert_seconds": sum(item["elapsed_seconds"] for item in receipts),
        "error_ranking_descending": [
            {"expert": item["expert"], "relative_squared_error": item["relative_squared_error"], "relative_rmse": item["relative_rmse"]}
            for item in sorted(receipts, key=lambda row: (-row["relative_squared_error"], row["expert"]))
        ],
        "expert_artifacts": [
            {"expert": item["expert"], "file": item["output_file"], "bytes": item["output_bytes"], "sha256": item["output_sha256"]}
            for item in sorted(receipts, key=lambda row: row["expert"])
        ],
        "passed": all(item["passed"] for item in receipts),
    }
    path = layer_dir / "LAYER_RECEIPT.json"
    atomic_json(path, result)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", default="0-255")
    parser.add_argument("--numeric-core", type=Path, required=True)
    parser.add_argument("--numeric-core-sha256", required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.layer not in range(3, 79):
        parser.error("layer must be 3..78")
    from r7_encoder.trellis import CodecConfig
    import torch

    torch.use_deterministic_algorithms(True)

    config = CodecConfig(
        device=args.device,
        sigma_reg=0.025,
        numeric_core=args.numeric_core,
        numeric_core_sha256=args.numeric_core_sha256,
        extension=args.extension,
        extension_sha256=args.extension_sha256,
        verify_files=True,
    )
    codec = IdentityR10Codec(config)
    source = StagedSource(args.source, args.source_inventory)
    experts = parse_experts(args.experts)
    receipts = []
    for expert in experts:
        try:
            receipt = encode_expert(codec, source, args.output, args.layer, expert)
            receipts.append(receipt)
            print(json.dumps({"event": "expert_complete", "layer": args.layer, "expert": expert, "elapsed_seconds": receipt["elapsed_seconds"], "relative_rmse": receipt["relative_rmse"], "output_sha256": receipt["output_sha256"]}, sort_keys=True), flush=True)
        except BaseException as exc:
            error_dir = args.output / f"layer-{args.layer:03d}" / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(error_dir / f"expert-{expert:03d}.json", {"schema": "glm53-full-exl3-tp3.encoder-error.v1", "layer": args.layer, "expert": expert, "exception": repr(exc), "time_unix": time.time()})
            raise
    if experts == list(range(EXPERTS)):
        path = seal_layer(args.output, args.layer, receipts)
        print(json.dumps({"event": "layer_complete", "layer": args.layer, "receipt": str(path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
