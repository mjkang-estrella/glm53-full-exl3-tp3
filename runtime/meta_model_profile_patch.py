"""Opt-in zero-weight allocation profile for the full TP3 model.

This patch constructs the exact vLLM module tree on the ``meta`` device and
records every final parameter/buffer shape.  It never opens checkpoint tensor
payloads and deliberately terminates before vLLM can allocate KV cache or run
inference.  The profile is a diagnostic, not a serving mode.
"""

from __future__ import annotations

from collections import defaultdict
from functools import wraps
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import torch


_LAYER = re.compile(r"(?:^|\.)layers\.(?P<layer>[0-9]+)(?:\.|$)")


def _enabled() -> bool:
    return os.environ.get("VLLM_GLM53_META_PROFILE", "0") == "1"


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _category(name: str) -> str:
    if "routed_experts" in name and any(
        token in name
        for token in ("w13_trellis", "w13_suh", "w13_svh", "w13_mcg", "w2_trellis", "w2_suh", "w2_svh", "w2_mcg")
    ):
        return "k3_routed_experts"
    if ".shared_experts." in name:
        return "bf16_shared_experts"
    if ".self_attn." in name:
        return "bf16_attention"
    if ".mlp.gate." in name or name.endswith("e_score_correction_bias"):
        return "bf16_router"
    if "embed_tokens" in name:
        return "bf16_embedding"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "bf16_lm_head"
    if "norm" in name:
        return "bf16_norm"
    if ".mlp." in name:
        return "bf16_dense_mlp"
    if "index" in name:
        return "bf16_indexer"
    return "other"


def summarize_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[dict], dict[str, int], dict[str, int], int]:
    rows: list[dict] = []
    by_category: defaultdict[str, int] = defaultdict(int)
    by_layer: defaultdict[str, int] = defaultdict(int)
    total = 0
    for name, tensor in named_tensors:
        size = _tensor_bytes(tensor)
        category = _category(name)
        match = _LAYER.search(name)
        layer = match.group("layer") if match is not None else "non_layer"
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": size,
                "category": category,
                "layer": layer,
            }
        )
        by_category[category] += size
        by_layer[layer] += size
        total += size
    rows.sort(key=lambda row: row["name"])
    return rows, dict(sorted(by_category.items())), dict(sorted(by_layer.items())), total


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # The profiler runs as root inside the isolated container while the
        # durable collector connects as the unprivileged operator account.
        # Make only this immutable diagnostic readable after the atomic seal.
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_patches() -> None:
    if not _enabled() or getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True

    from vllm.model_executor.model_loader import base_loader

    original = base_loader.BaseModelLoader.load_model

    @wraps(original)
    def meta_load_model(self, vllm_config, model_config, prefix=""):
        output = Path(os.environ["GLM53_META_PROFILE_OUTPUT"])
        original_empty = torch.empty

        # A few model constructors pass ``device='cuda'`` explicitly instead
        # of inheriting the surrounding default device.  Redirect only during
        # this diagnostic construction, then restore the factory immediately.
        def meta_empty(*args, **kwargs):
            kwargs["device"] = "meta"
            return original_empty(*args, **kwargs)

        try:
            with base_loader.set_default_torch_dtype(model_config.dtype):
                torch.empty = meta_empty
                with torch.device("meta"):
                    model = base_loader.initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
        finally:
            torch.empty = original_empty

        parameters, parameter_categories, parameter_layers, parameter_bytes = summarize_tensors(
            model.named_parameters()
        )
        buffers, buffer_categories, buffer_layers, buffer_bytes = summarize_tensors(
            model.named_buffers()
        )
        non_meta_parameters = [row for row in parameters if row["device"] != "meta"]
        non_meta_buffers = [row for row in buffers if row["device"] != "meta"]
        model_body = getattr(model, "model", None)
        value = {
            "schema": "glm53-full-exl3-tp3.meta-model-profile.v1",
            "passed": not non_meta_parameters and not non_meta_buffers,
            "rank": int(os.environ.get("RANK", "-1")),
            "world_size": int(os.environ.get("WORLD_SIZE", "-1")),
            "architecture": list(getattr(model_config, "architectures", []) or []),
            "geometry_id": os.environ.get("VLLM_GLM53_EXL3_TP3_GEOMETRY_ID"),
            "placement": os.environ.get("GLM53_META_PROFILE_PLACEMENT", "tp3"),
            "tensor_parallel_size": int(vllm_config.parallel_config.tensor_parallel_size),
            "pipeline_parallel_size": int(vllm_config.parallel_config.pipeline_parallel_size),
            "start_layer": getattr(model_body, "start_layer", None),
            "end_layer": getattr(model_body, "end_layer", None),
            "parameter_count": len(parameters),
            "parameter_bytes": parameter_bytes,
            "parameter_bytes_by_category": parameter_categories,
            "parameter_bytes_by_layer": parameter_layers,
            "buffer_count": len(buffers),
            "buffer_bytes": buffer_bytes,
            "buffer_bytes_by_category": buffer_categories,
            "buffer_bytes_by_layer": buffer_layers,
            "non_meta_parameters": non_meta_parameters,
            "non_meta_buffers": non_meta_buffers,
            "parameters": parameters,
            "buffers": buffers,
        }
        _atomic_json(output, value)
        print(
            "GLM53_META_MODEL_PROFILE_OK "
            f"rank={value['rank']} parameters={len(parameters)} "
            f"parameter_bytes={parameter_bytes} buffers={len(buffers)} "
            f"buffer_bytes={buffer_bytes}",
            flush=True,
        )
        raise RuntimeError("GLM53_META_PROFILE_COMPLETE")

    base_loader.BaseModelLoader.load_model = meta_load_model
    print("GLM53_META_MODEL_PROFILE_PATCH_INSTALLED", flush=True)


__all__ = ["apply_patches", "summarize_tensors"]
