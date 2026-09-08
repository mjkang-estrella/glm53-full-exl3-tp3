"""Lossless PP3 execution of the sealed rotating TP3 K3 slice payloads.

The checkpoint stores three independently encoded intermediate-channel slices
for every routed expert.  Under pipeline parallelism each stage owns complete
transformer layers, so it must evaluate all three slices locally:

    down_0(act(gate_0, up_0)) + down_1(...) + down_2(...)

This is the same quantized function as the qualified TP3 path; only placement
of the three partial sums changes.  No slice is re-encoded or expanded.
"""

from __future__ import annotations

from functools import wraps
import os
import re
import sys

import torch
from torch import nn

from tp3k3.geometry import (
    GEOMETRY_ID,
    SEMANTIC_INTERMEDIATE,
    layer_geometry,
    rank_offsets,
    rank_widths,
)


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(?P<layer>[0-9]+)(?:\.|$)")
_K3_WEIGHT = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>[0-2])\.(?P<suffix>trellis|suh|svh|mcg)$"
)


def _enabled() -> bool:
    return os.environ.get("VLLM_GLM53_EXL3_PP3_ALL_SLICES", "0") == "1"


def pipeline_layer_bounds(total_layers: int, pipeline_size: int, pipeline_rank: int) -> tuple[int, int]:
    """Match vLLM's balanced contiguous pipeline partition for divisible GLM."""

    if total_layers <= 0 or pipeline_size <= 0 or not 0 <= pipeline_rank < pipeline_size:
        raise ValueError("invalid pipeline partition")
    base, extra = divmod(total_layers, pipeline_size)
    start = pipeline_rank * base + min(pipeline_rank, extra)
    end = start + base + (1 if pipeline_rank < extra else 0)
    return start, end


class AllSlicesMoE(nn.Module):
    """Run the three serialized slice runners and sum their expert outputs."""

    def __init__(self, rank_slices: list[nn.Module], layer: int):
        super().__init__()
        if len(rank_slices) != 3:
            raise ValueError("the rotating checkpoint requires exactly three slices")
        self.rank_slices = nn.ModuleList(rank_slices)
        self.layer = int(layer)
        self.widths = rank_widths(layer)
        self.offsets = rank_offsets(layer)
        if sum(self.widths) != SEMANTIC_INTERMEDIATE:
            raise RuntimeError("slice widths do not cover the semantic intermediate dimension")

    @property
    def routed_experts_by_rank(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return tuple(runner.routed_experts for runner in self.rank_slices)  # type: ignore[return-value]

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The first runner owns the one shared-expert reference; the other two
        # contain routed slices only.  Thus summing the three outputs adds the
        # shared expert exactly once.
        result = self.rank_slices[0](
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
            shared_experts_input=shared_experts_input,
        )
        for runner in self.rank_slices[1:]:
            result = result + runner(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
                shared_experts_input=shared_experts_input,
            )
        return result


def _load_one_slice(routed, *, expert: int, projection: str, suffix: str, value: torch.Tensor) -> str:
    if projection == "gate_proj":
        parameter_name, shard_id = f"w13_{suffix}", "w1"
    elif projection == "up_proj":
        parameter_name, shard_id = f"w13_{suffix}", "w3"
    elif projection == "down_proj":
        parameter_name, shard_id = f"w2_{suffix}", "w2"
    else:  # pragma: no cover - guarded by the checkpoint-name regex
        raise ValueError(f"unknown projection {projection}")
    parameter = getattr(routed, parameter_name)
    success = parameter.weight_loader(
        param=parameter,
        loaded_weight=value,
        weight_name=f"{routed.layer_name}.{parameter_name}",
        shard_id=shard_id,
        expert_id=expert,
        return_success=True,
    )
    if not success:
        raise RuntimeError(
            f"PP3 all-slice load rejected expert={expert} projection={projection} suffix={suffix}"
        )
    return parameter_name


def apply_patches() -> None:
    if not _enabled() or getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True

    from vllm.distributed import (
        get_pp_group,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.models import deepseek_v2

    exl3 = sys.modules.get("exl3")
    if exl3 is None:
        from vllm.model_executor.layers.quantization import exl3 as package_exl3

        exl3 = package_exl3
        sys.modules["exl3"] = exl3

    declared = os.environ.get("VLLM_GLM53_EXL3_TP3_GEOMETRY_ID", GEOMETRY_ID)
    if declared != GEOMETRY_ID:
        raise RuntimeError(f"checkpoint geometry {declared!r} != runtime {GEOMETRY_ID!r}")

    original_factory = deepseek_v2.FusedMoEFactory

    @wraps(original_factory)
    def all_slices_factory(*args, **kwargs):
        if args:
            raise RuntimeError("PP3 all-slices expects the named FusedMoEFactory API")
        if get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError("PP3 all-slices requires tensor_parallel_size=1")
        if get_pp_group().world_size != 3:
            raise RuntimeError("PP3 all-slices requires pipeline_parallel_size=3")
        prefix = str(kwargs.get("prefix", ""))
        match = _LAYER_PATTERN.search(prefix)
        if match is None:
            raise RuntimeError(f"cannot resolve pipeline layer from prefix {prefix!r}")
        layer = int(match.group("layer"))
        geometry = layer_geometry(layer)
        widths = rank_widths(layer)
        offsets = rank_offsets(layer)
        if int(kwargs.get("intermediate_size", -1)) != SEMANTIC_INTERMEDIATE:
            raise RuntimeError("GLM-5.3 routed expert semantic width differs")
        shared_experts = kwargs.get("shared_experts")
        runners: list[nn.Module] = []
        for serialized_rank, (offset, width) in enumerate(
            zip(offsets, widths, strict=True)
        ):
            slice_kwargs = dict(kwargs)
            slice_kwargs["intermediate_size"] = int(width)
            slice_kwargs["prefix"] = f"{prefix}.rank_slices.{serialized_rank}"
            # Preserve the shared-expert contribution exactly once.
            slice_kwargs["shared_experts"] = shared_experts if serialized_rank == 0 else None
            runner = original_factory(**slice_kwargs)
            routed = runner.routed_experts
            routed._exl3_geometry_id = GEOMETRY_ID
            routed._exl3_layer = layer
            routed._exl3_slice_rank = serialized_rank
            routed._exl3_semantic_intermediate = SEMANTIC_INTERMEDIATE
            routed._exl3_intermediate_offset = int(offset)
            routed._exl3_expected_intermediate_local = int(width)
            runners.append(runner)
        return AllSlicesMoE(runners, layer)

    deepseek_v2.FusedMoEFactory = all_slices_factory

    original_process = exl3.Exl3MoEMethod.process_weights_after_loading

    @wraps(original_process)
    def process_weights(self, routed):
        original_process(self, routed)
        if not hasattr(routed, "_exl3_slice_rank"):
            return
        layer = int(routed._exl3_layer)
        serialized_rank = int(routed._exl3_slice_rank)
        geometry = layer_geometry(layer)
        expected_width = rank_widths(layer)[serialized_rank]
        expected_offset = rank_offsets(layer)[serialized_rank]
        if int(getattr(routed, "_exl3_intermediate_local", -1)) != expected_width:
            raise RuntimeError("PP3 EXL3 slice width differs after loading")
        if int(routed._exl3_intermediate_offset) != expected_offset:
            raise RuntimeError("PP3 EXL3 slice offset differs after loading")
        if not getattr(routed, "_exl3_inners", None) or not getattr(routed, "_exl3_ptrs", None):
            raise RuntimeError("PP3 EXL3 fused slice plan was not constructed")
        print(
            "GLM53_EXL3_PP3_SLICE_GEOMETRY_OK "
            f"layer={layer} serialized_rank={serialized_rank} "
            f"offset={expected_offset} width={expected_width} padding=0",
            flush=True,
        )

    exl3.Exl3MoEMethod.process_weights_after_loading = process_weights

    original_model_load = deepseek_v2.DeepseekV2Model.load_weights

    @wraps(original_model_load)
    def model_load(self, weights):
        loaded_k3: set[str] = set()
        base_layers = int(self.config.num_hidden_layers)
        per_layer_rank_counts: dict[tuple[int, int], int] = {}
        k3_seen = 0
        k3_local = 0
        k3_nonlocal_skipped = 0
        k3_mtp_skipped = 0

        def remaining_weights():
            nonlocal k3_seen, k3_local, k3_nonlocal_skipped, k3_mtp_skipped
            for name, value in weights:
                match = _K3_WEIGHT.fullmatch(name)
                if match is None:
                    yield name, value
                    continue
                k3_seen += 1
                layer = int(match.group("layer"))
                if layer >= base_layers:
                    k3_mtp_skipped += 1
                    continue
                if not self.start_layer <= layer < self.end_layer:
                    k3_nonlocal_skipped += 1
                    continue
                if layer < 3:
                    raise RuntimeError(f"unexpected K3 tensor below routed layer 3: {name}")
                serialized_rank = int(match.group("rank"))
                wrapper = self.layers[layer].mlp.experts
                if not isinstance(wrapper, AllSlicesMoE):
                    raise RuntimeError(f"layer {layer} is not the PP3 all-slices runtime")
                routed = wrapper.rank_slices[serialized_rank].routed_experts
                parameter_name = _load_one_slice(
                    routed,
                    expert=int(match.group("expert")),
                    projection=match.group("projection"),
                    suffix=match.group("suffix"),
                    value=value,
                )
                per_layer_rank_counts[(layer, serialized_rank)] = (
                    per_layer_rank_counts.get((layer, serialized_rank), 0) + 1
                )
                k3_local += 1
                loaded_k3.add(
                    f"layers.{layer}.mlp.experts.rank_slices.{serialized_rank}."
                    f"routed_experts.{parameter_name}"
                )

        loaded = set(original_model_load(self, remaining_weights()))
        local_moe_layers = list(range(max(3, self.start_layer), self.end_layer))
        expected_per_slice = 256 * 3 * 4
        for layer in local_moe_layers:
            for serialized_rank in range(3):
                actual = per_layer_rank_counts.get((layer, serialized_rank), 0)
                if actual != expected_per_slice:
                    raise RuntimeError(
                        f"PP3 K3 layer {layer} slice {serialized_rank} count "
                        f"{actual}/{expected_per_slice}"
                    )
        expected_local = len(local_moe_layers) * expected_per_slice * 3
        if k3_local != expected_local:
            raise RuntimeError(f"PP3 K3 local tensor count {k3_local}/{expected_local}")
        loaded.update(loaded_k3)
        print(
            "GLM53_FULL_K3_PP3_ALL_SLICES_LOAD_OK "
            f"layers={self.start_layer}:{self.end_layer} seen={k3_seen} "
            f"local={k3_local} nonlocal_skipped={k3_nonlocal_skipped} "
            f"mtp_skipped={k3_mtp_skipped}",
            flush=True,
        )
        return loaded

    deepseek_v2.DeepseekV2Model.load_weights = model_load
    print(
        "GLM53_EXL3_PP3_ALL_SLICES_PATCH_INSTALLED "
        f"geometry={GEOMETRY_ID} placement=PP3_TP1",
        flush=True,
    )


__all__ = ["AllSlicesMoE", "apply_patches", "pipeline_layer_bounds"]
