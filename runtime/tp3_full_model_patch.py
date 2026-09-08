"""Runtime-only TP3 padding plus rotating-uneven EXL3 integration.

The public checkpoint config remains semantic: 64 attention heads, vocabulary
154880, and expert width 2048. Runtime allocation pads only generic TP
boundaries: attention to 66 heads, vocabulary to 154944 rows, and the BF16
shared-expert MLP to 2304 channels. Routed experts use the unpadded per-layer
768/640/640 geometry from :mod:`tp3_rank_slice_patch`.
"""

from __future__ import annotations

from functools import wraps
import math
import os
import re
import sys

import torch


SEMANTIC_HEADS = 64
PHYSICAL_HEADS = 66
SEMANTIC_VOCAB = 154880
PHYSICAL_VOCAB = 154944
SEMANTIC_SHARED_INTERMEDIATE = 2048
PHYSICAL_SHARED_INTERMEDIATE = 2304

_K3_WEIGHT = re.compile(
    r"^(?:model\.)?layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"rank(?P<rank>[0-2])\.(?P<suffix>trellis|suh|svh|mcg)$"
)


def _enabled() -> bool:
    return os.environ.get("VLLM_GLM53_EXL3_TP3_FULL_MODEL", "0") == "1"


def _pad_dim(source: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    if source.shape[dim] == size:
        return source
    if source.shape[dim] > size:
        raise ValueError(f"refusing to truncate {tuple(source.shape)} dim {dim} to {size}")
    shape = list(source.shape)
    shape[dim] = size
    output = torch.zeros(shape, dtype=source.dtype, device=source.device)
    selection = [slice(None)] * source.ndim
    selection[dim] = slice(0, source.shape[dim])
    output[tuple(selection)].copy_(source)
    return output


def _parse_k3_weight(name: str) -> tuple[int, str] | None:
    """Return the model layer and RoutedExperts-relative serialized name."""

    match = _K3_WEIGHT.fullmatch(name)
    if match is None:
        return None
    layer = int(match.group("layer"))
    relative = (
        f"{match.group('expert')}.{match.group('projection')}."
        f"rank{match.group('rank')}.{match.group('suffix')}"
    )
    return layer, relative


def apply_patches() -> None:
    if not _enabled() or getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True

    from vllm.config.model import ModelConfig
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers import linear as linear_mod
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from vllm.model_executor.models import deepseek_v2

    exl3 = sys.modules.get("exl3")
    if exl3 is None:
        from vllm.model_executor.layers.quantization import exl3 as package_exl3
        exl3 = package_exl3
        sys.modules["exl3"] = exl3

    original_verify = ModelConfig.verify_with_parallel_config

    @wraps(original_verify)
    def verify_parallel(self, parallel_config):
        architecture = list(getattr(self, "architectures", []) or [])
        heads = self.model_arch_config.total_num_attention_heads
        is_target = "GlmMoeDsaForCausalLM" in architecture
        if parallel_config.tensor_parallel_size == 3 and is_target and heads == SEMANTIC_HEADS:
            config = self.hf_config
            saved = getattr(config, "num_attention_heads", None)
            config.num_attention_heads = PHYSICAL_HEADS
            object.__setattr__(self.model_arch_config, "total_num_attention_heads", PHYSICAL_HEADS)
            try:
                return original_verify(self, parallel_config)
            finally:
                object.__setattr__(self.model_arch_config, "total_num_attention_heads", heads)
                config.num_attention_heads = saved
        return original_verify(self, parallel_config)

    ModelConfig.verify_with_parallel_config = verify_parallel

    original_column = linear_mod.ColumnParallelLinear.weight_loader

    @wraps(original_column)
    def column_loader(self, param, loaded_weight):
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None and not getattr(param, "is_sharded_weight", False):
            expected = param.data.shape[output_dim] * self.tp_size
            if loaded_weight.ndim > output_dim and loaded_weight.shape[output_dim] < expected:
                loaded_weight = _pad_dim(loaded_weight, output_dim, expected)
        return original_column(self, param, loaded_weight)

    linear_mod.ColumnParallelLinear.weight_loader = column_loader
    original_column_v2 = linear_mod.ColumnParallelLinear.weight_loader_v2

    @wraps(original_column_v2)
    def column_loader_v2(self, param, loaded_weight):
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None:
            expected = param.data.shape[output_dim] * self.tp_size
            if loaded_weight.ndim > output_dim and loaded_weight.shape[output_dim] < expected:
                loaded_weight = _pad_dim(loaded_weight, output_dim, expected)
        return original_column_v2(self, param, loaded_weight)

    linear_mod.ColumnParallelLinear.weight_loader_v2 = column_loader_v2

    original_row = linear_mod.RowParallelLinear.weight_loader

    @wraps(original_row)
    def row_loader(self, param, loaded_weight):
        input_dim = getattr(param, "input_dim", None)
        if input_dim is not None and not getattr(param, "is_sharded_weight", False):
            expected = param.data.shape[input_dim] * self.tp_size
            if loaded_weight.ndim > input_dim and loaded_weight.shape[input_dim] < expected:
                loaded_weight = _pad_dim(loaded_weight, input_dim, expected)
        return original_row(self, param, loaded_weight)

    linear_mod.RowParallelLinear.weight_loader = row_loader
    original_row_v2 = linear_mod.RowParallelLinear.weight_loader_v2

    @wraps(original_row_v2)
    def row_loader_v2(self, param, loaded_weight):
        input_dim = getattr(param, "input_dim", None)
        if input_dim is not None:
            expected = param.data.shape[input_dim] * self.tp_size
            if loaded_weight.ndim > input_dim and loaded_weight.shape[input_dim] < expected:
                loaded_weight = _pad_dim(loaded_weight, input_dim, expected)
        return original_row_v2(self, param, loaded_weight)

    linear_mod.RowParallelLinear.weight_loader_v2 = row_loader_v2

    original_merged = linear_mod.MergedColumnParallelLinear.weight_loader

    @wraps(original_merged)
    def merged_loader(self, param, loaded_weight, loaded_shard_id=None):
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None and isinstance(loaded_shard_id, int):
            expected = self.output_sizes[loaded_shard_id]
            if loaded_weight.ndim > output_dim and loaded_weight.shape[output_dim] < expected:
                loaded_weight = _pad_dim(loaded_weight, output_dim, expected)
        return original_merged(self, param, loaded_weight, loaded_shard_id)

    linear_mod.MergedColumnParallelLinear.weight_loader = merged_loader
    original_merged_v2 = linear_mod.MergedColumnParallelLinear.weight_loader_v2

    @wraps(original_merged_v2)
    def merged_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None and isinstance(loaded_shard_id, int):
            expected = self.output_sizes[loaded_shard_id]
            if loaded_weight.ndim > output_dim and loaded_weight.shape[output_dim] < expected:
                loaded_weight = _pad_dim(loaded_weight, output_dim, expected)
        return original_merged_v2(self, param, loaded_weight, loaded_shard_id)

    linear_mod.MergedColumnParallelLinear.weight_loader_v2 = merged_loader_v2

    original_vocab = VocabParallelEmbedding.__init__

    @wraps(original_vocab)
    def vocab_init(self, *args, **kwargs):
        if get_tensor_model_parallel_world_size() == 3 and not kwargs.get("disable_tp", False):
            args = list(args)
            if "padding_size" in kwargs:
                kwargs["padding_size"] = math.lcm(int(kwargs["padding_size"]), 3)
            elif len(args) >= 5:
                args[4] = math.lcm(int(args[4]), 3)
            else:
                kwargs["padding_size"] = 192
        return original_vocab(self, *args, **kwargs)

    VocabParallelEmbedding.__init__ = vocab_init

    # Only the source-precision shared expert needs an equal TP3 padding.
    original_mlp_init = deepseek_v2.DeepseekV2MLP.__init__

    @wraps(original_mlp_init)
    def mlp_init(self, hidden_size, intermediate_size, *args, **kwargs):
        prefix = str(kwargs.get("prefix", ""))
        if (
            get_tensor_model_parallel_world_size() == 3
            and prefix.endswith(".shared_experts")
            and intermediate_size == SEMANTIC_SHARED_INTERMEDIATE
        ):
            intermediate_size = PHYSICAL_SHARED_INTERMEDIATE
        return original_mlp_init(self, hidden_size, intermediate_size, *args, **kwargs)

    deepseek_v2.DeepseekV2MLP.__init__ = mlp_init

    # Give only layer construction a 66-head physical view. The public HF
    # config is restored immediately after each layer is built.
    original_decoder_init = deepseek_v2.DeepseekV2DecoderLayer.__init__

    @wraps(original_decoder_init)
    def decoder_init(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None:
            vllm_config = kwargs.get("vllm_config") or args[0]
            config = vllm_config.model_config.hf_config
        saved = getattr(config, "num_attention_heads", None)
        if get_tensor_model_parallel_world_size() == 3 and saved == SEMANTIC_HEADS:
            config.num_attention_heads = PHYSICAL_HEADS
            try:
                return original_decoder_init(self, *args, **kwargs)
            finally:
                config.num_attention_heads = saved
        return original_decoder_init(self, *args, **kwargs)

    deepseek_v2.DeepseekV2DecoderLayer.__init__ = decoder_init

    lazy_k3 = os.environ.get("VLLM_GLM53_EXL3_LAZY_K3", "0") == "1"
    pp3_all_slices = os.environ.get("VLLM_GLM53_EXL3_PP3_ALL_SLICES", "0") == "1"
    if lazy_k3 and pp3_all_slices:
        raise RuntimeError("lazy TP3 and PP3 all-slices modes are mutually exclusive")
    if lazy_k3:
        mixed_bits = os.environ.get("GLM53_MIXED_EXPERT_BITS", "0") == "1"
        if not mixed_bits:
            model_dir = os.environ.get("MODEL_DIR", "/model")
            mixed_bits = os.path.isfile(os.path.join(model_dir, "expert-bits.json"))
        if mixed_bits:
            from lazy_k275_patch import apply_patches as apply_lazy_k3_patches
        else:
            from lazy_k3_patch import apply_patches as apply_lazy_k3_patches

        apply_lazy_k3_patches()
    elif pp3_all_slices:
        from pp3_all_slices_patch import apply_patches as apply_pp3_all_slices_patches

        apply_pp3_all_slices_patches()
    else:
        from tp3_rank_slice_patch import apply_patches as apply_rank_slice_patches

        apply_rank_slice_patches()

    # DeepseekV2Model has a custom loader which handles stock expert names
    # before AutoWeightsLoader can descend into RoutedExperts. The rotating
    # checkpoint adds a terminal rank component, so intercept only those K3
    # tensors and feed them to the already-qualified RoutedExperts loader.
    # Everything else continues through the stock model loader unchanged and
    # the iterator remains streaming. Layer 78 is an MTP layer and is retained
    # in the checkpoint but intentionally skipped by the target-only runtime.
    original_model_load = deepseek_v2.DeepseekV2Model.load_weights

    @wraps(original_model_load)
    def model_load(self, weights):
        loaded_k3: set[str] = set()
        base_layers = int(self.config.num_hidden_layers)
        k3_seen = 0
        k3_selected = 0
        k3_mtp_skipped = 0

        def remaining_weights():
            nonlocal k3_seen, k3_selected, k3_mtp_skipped
            for name, value in weights:
                parsed = _parse_k3_weight(name)
                if parsed is None:
                    yield name, value
                    continue
                layer_number, relative = parsed
                k3_seen += 1
                if layer_number >= base_layers:
                    k3_mtp_skipped += 1
                    continue
                if layer_number < 3:
                    raise RuntimeError(f"unexpected K3 tensor below routed layer 3: {name}")
                layer = self.layers[layer_number]
                routed = layer.mlp.experts.routed_experts
                for parameter_name in routed.load_weights([(relative, value)]):
                    k3_selected += 1
                    loaded_k3.add(
                        f"layers.{layer_number}.mlp.experts.routed_experts."
                        f"{parameter_name}"
                    )

        loaded = set(original_model_load(self, remaining_weights()))
        expected_seen = 76 * 256 * 36
        expected_mtp = 256 * 36
        expected_selected = 75 * 256 * 3 * 4
        if (k3_seen, k3_mtp_skipped, k3_selected) != (
            expected_seen,
            expected_mtp,
            expected_selected,
        ):
            raise RuntimeError(
                "full K3 streaming load count differs: "
                f"seen={k3_seen}/{expected_seen} "
                f"mtp_skipped={k3_mtp_skipped}/{expected_mtp} "
                f"rank_selected={k3_selected}/{expected_selected}"
            )
        loaded.update(loaded_k3)
        print(
            "GLM53_FULL_K3_STREAM_LOAD_OK "
            f"seen={k3_seen} mtp_skipped={k3_mtp_skipped} "
            f"rank_selected={k3_selected}",
            flush=True,
        )
        return loaded

    if not pp3_all_slices and not lazy_k3:
        deepseek_v2.DeepseekV2Model.load_weights = model_load
    print(
        "GLM53_FULL_TP3_PADDING_APPLIED "
        f"attention={SEMANTIC_HEADS}->{PHYSICAL_HEADS} "
        f"vocab={SEMANTIC_VOCAB}->{PHYSICAL_VOCAB} "
        f"shared_expert={SEMANTIC_SHARED_INTERMEDIATE}->{PHYSICAL_SHARED_INTERMEDIATE}",
        flush=True,
    )


__all__ = ["apply_patches"]
