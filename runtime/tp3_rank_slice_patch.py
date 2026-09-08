"""Rotating uneven TP3 EXL3 rank-slice loader and allocation patch.

Apply after the sealed ``exl3.py`` overlay is imported and before constructing
``GlmMoeDsaForCausalLM``. Checkpoint names contain terminal ``.rank0`` through
``.rank2`` components. Each process consumes only its rank's already-sharded,
layer-specific 640- or 768-channel payload. The patch changes only the routed
``FusedMoEFactory`` allocation and does not mutate public model configuration.
"""

from __future__ import annotations

from functools import wraps
import os
import re


from tp3k3.geometry import (
    GEOMETRY_ID,
    PHYSICAL_INTERMEDIATE,
    SEMANTIC_INTERMEDIATE,
    layer_geometry,
    rank_geometry,
)


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(?P<layer>[0-9]+)(?:\.|$)")
_RANK_PATTERN = re.compile(
    r"(?:^|\.)(?P<expert>[0-9]+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>[0-2])\.(?P<suffix>trellis|suh|svh|mcg)$"
)


def _enabled() -> bool:
    return os.environ.get("VLLM_GLM53_EXL3_TP3_RANK_SLICES", "0") == "1"


def _prefix_layer(prefix: str) -> int:
    match = _LAYER_PATTERN.search(prefix)
    if match is None:
        raise RuntimeError(f"cannot resolve rotating TP3 layer from prefix {prefix!r}")
    layer = int(match.group("layer"))
    # rank_geometry validates the routed-layer range.
    rank_geometry(layer, 0)
    return layer


def apply_patches() -> None:
    if not _enabled() or getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True

    # The synthetic qualification imports the sealed overlay as top-level
    # ``exl3``. Full vLLM imports the same file through its package namespace.
    # Use the already-loaded object in either case so registration and method
    # monkey-patching are applied exactly once.
    import sys
    exl3 = sys.modules.get("exl3")
    if exl3 is None:
        from vllm.model_executor.layers.quantization import exl3
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.models import deepseek_v2

    declared = os.environ.get("VLLM_GLM53_EXL3_TP3_GEOMETRY_ID", GEOMETRY_ID)
    if declared != GEOMETRY_ID:
        raise RuntimeError(f"checkpoint geometry {declared!r} != runtime {GEOMETRY_ID!r}")

    original_factory = deepseek_v2.FusedMoEFactory

    @wraps(original_factory)
    def fused_moe_factory(*args, **kwargs):
        if get_tensor_model_parallel_world_size() != 3:
            raise RuntimeError("TP3 rank-slice patch requires tensor_parallel_size=3")
        prefix = str(kwargs.get("prefix", ""))
        layer = _prefix_layer(prefix)
        live_rank = get_tensor_model_parallel_rank()
        offset, local_width = rank_geometry(layer, live_rank)
        semantic = int(kwargs.get("intermediate_size", -1))
        if semantic != SEMANTIC_INTERMEDIATE:
            raise RuntimeError(
                f"GLM-5.3 semantic routed width must be {SEMANTIC_INTERMEDIATE}, got {semantic}"
            )
        # FusedMoEConfig divides this value by TP. Supplying rank-local*TP to
        # each process allocates exactly that process's uneven local buffers.
        # Non-routed tensors and the public semantic config are not changed.
        kwargs["intermediate_size"] = local_width * 3
        runner = original_factory(*args, **kwargs)
        routed = runner.routed_experts
        routed._exl3_geometry_id = GEOMETRY_ID
        routed._exl3_layer = layer
        routed._exl3_semantic_intermediate = SEMANTIC_INTERMEDIATE
        routed._exl3_physical_intermediate = PHYSICAL_INTERMEDIATE
        routed._exl3_intermediate_offset = offset
        routed._exl3_expected_intermediate_local = local_width
        routed._exl3_wide_rank = layer_geometry(layer)["wide_rank"]
        return runner

    deepseek_v2.FusedMoEFactory = fused_moe_factory

    original_load_weights = RoutedExperts.load_weights

    @wraps(original_load_weights)
    def load_weights(self, weights):
        live_rank = get_tensor_model_parallel_rank()
        for name, value in weights:
            match = _RANK_PATTERN.search(name)
            if match is None:
                yield from original_load_weights(self, [(name, value)])
                continue
            if int(match.group("rank")) != live_rank:
                continue
            expert_id = int(match.group("expert"))
            projection = match.group("projection")
            suffix = match.group("suffix")
            if projection == "gate_proj":
                parameter_name, shard_id = f"w13_{suffix}", "w1"
            elif projection == "up_proj":
                parameter_name, shard_id = f"w13_{suffix}", "w3"
            else:
                parameter_name, shard_id = f"w2_{suffix}", "w2"
            parameter = getattr(self, parameter_name)
            success = parameter.weight_loader(
                param=parameter,
                loaded_weight=value,
                weight_name=f"{self.layer_name}.{parameter_name}",
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
            if success:
                yield parameter_name

    RoutedExperts.load_weights = load_weights

    original_load_exl3 = exl3.Exl3MoEMethod._load_exl3

    @wraps(original_load_exl3)
    def load_exl3(self, param, loaded_weight, weight_name, shard_id="w1", expert_id=0, return_success=False):
        owner = getattr(param, "_exl3_owner", None)
        local_id = expert_id
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
        suffix = exl3._suffix_from_mapped_name(weight_name)
        if shard_id in ("w1", "w3"):
            shard_index = 0 if shard_id == "w1" else 1
            destination = param.data[local_id, shard_index]
        elif shard_id == "w2":
            destination = param.data[local_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")
        loaded = loaded_weight.detach().contiguous()
        if tuple(loaded.shape) == tuple(destination.shape):
            destination.copy_(loaded)
            return True if return_success else None
        return original_load_exl3(
            self,
            param,
            loaded_weight,
            weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=return_success,
        )

    exl3.Exl3MoEMethod._load_exl3 = load_exl3

    original_process = exl3.Exl3MoEMethod.process_weights_after_loading

    @wraps(original_process)
    def process_weights(self, layer):
        original_process(self, layer)
        live_rank = get_tensor_model_parallel_rank()
        layer_number = int(getattr(layer, "_exl3_layer", -1))
        offset, expected = rank_geometry(layer_number, live_rank)
        local = int(getattr(layer, "_exl3_intermediate_local", -1))
        if local != expected:
            raise RuntimeError(
                f"EXL3 TP3 layer {layer_number} rank {live_rank} width {local} != {expected}"
            )
        if int(getattr(layer, "_exl3_intermediate_offset", -1)) != offset:
            raise RuntimeError("EXL3 TP3 rank offset metadata differs")
        if layer.w13_svh.shape[-1] != expected or layer.w2_suh.shape[-1] != expected:
            raise RuntimeError("EXL3 rank-specific scale buffer geometry differs")
        # Fused pointer tables and temporary buffers are constructed by the
        # overlay from _exl3_intermediate_local. Validate those plans here.
        if not getattr(layer, "_exl3_inners", None) or not getattr(layer, "_exl3_ptrs", None):
            raise RuntimeError("EXL3 fused rank-specific plans were not constructed")
        layer._exl3_rank_geometry = {
            "schema": "glm53-full-exl3-tp3.runtime-rank-geometry.v2",
            "geometry_id": GEOMETRY_ID,
            "layer": layer_number,
            "rank": live_rank,
            "wide_rank": layer_geometry(layer_number)["wide_rank"],
            "semantic_global": SEMANTIC_INTERMEDIATE,
            "physical_global": PHYSICAL_INTERMEDIATE,
            "offset": offset,
            "local_width": expected,
            "padding_channels": 0,
            "gate_up_buffer_width": int(layer.w13_svh.shape[-1]),
            "down_buffer_width": int(layer.w2_suh.shape[-1]),
            "fused_pointer_experts": len(layer._exl3_ptrs),
        }
        print(
            "GLM53_EXL3_TP3_LAYER_GEOMETRY_OK "
            f"layer={layer_number} rank={live_rank} offset={offset} "
            f"width={expected} padding=0",
            flush=True,
        )

    exl3.Exl3MoEMethod.process_weights_after_loading = process_weights
    print(
        "GLM53_EXL3_TP3_UNEVEN_RANK_SLICES_PATCH_INSTALLED "
        f"geometry={GEOMETRY_ID}",
        flush=True,
    )


__all__ = ["apply_patches"]
