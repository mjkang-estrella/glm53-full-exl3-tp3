#!/usr/bin/env python3
"""Actual three-rank GLM MoE gate for bounded lazy K3 expert residency."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from types import SimpleNamespace


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--runtime-exl3", type=Path, required=True)
    parser.add_argument("--lazy-patch", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.rank not in range(3) or args.layer not in (3, 4, 5):
        parser.error("lazy gate requires rank 0..2 and rotation layer 3, 4, or 5")

    os.environ.update(
        MASTER_ADDR=args.master_addr,
        MASTER_PORT=str(args.master_port),
        WORLD_SIZE="3",
        RANK=str(args.rank),
        LOCAL_RANK="0",
        VLLM_GLM53_EXL3_LAZY_K3="1",
        VLLM_GLM53_EXL3_TP3_GEOMETRY_ID="rotating-uneven-768-640-640-v1",
        GLM53_LAZY_K3_MODEL_DIR=str(args.model_dir),
        GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER=os.environ.get(
            "GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER", "8"
        ),
        GLM53_LAZY_K3_EXECUTION=os.environ.get(
            "GLM53_LAZY_K3_EXECUTION", "chunked_fused"
        ),
        EXL3_FUSED_MOE="1",
        VLLM_USE_V1="1",
    )
    import torch
    import torch.distributed as dist
    import vllm

    from tp3k3.geometry import GEOMETRY_ID, layer_geometry, rank_geometry

    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    runtime = import_path("exl3", args.runtime_exl3)
    lazy_patch = import_path("lazy_k3_patch", args.lazy_patch)

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models import deepseek_v2

    parallel = ParallelConfig(
        tensor_parallel_size=3,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        distributed_executor_backend="external_launcher",
        disable_custom_all_reduce=True,
    )
    vllm_config = VllmConfig(parallel_config=parallel)
    started = time.monotonic()
    result = {
        "schema": "glm53-full-exl3-tp3.lazy-distributed-runtime-gate.v2",
        "geometry_id": GEOMETRY_ID,
        "layer": args.layer,
        "rank": args.rank,
        "world_size": 3,
        "host": socket.gethostname(),
        "architecture_class": "vllm.model_executor.models.deepseek_v2.GlmMoeDsaForCausalLM",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": getattr(vllm, "__version__", None),
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=3,
                rank=args.rank,
                local_rank=0,
                distributed_init_method=f"tcp://{args.master_addr}:{args.master_port}",
                backend="nccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=3, pipeline_model_parallel_size=1
            )
            lazy_patch.apply_patches()
            if not issubclass(
                deepseek_v2.GlmMoeDsaForCausalLM, deepseek_v2.DeepseekV2ForCausalLM
            ):
                raise AssertionError("GlmMoeDsaForCausalLM registry path differs")
            config = SimpleNamespace(
                model_type="glm_moe_dsa",
                hidden_size=6144,
                moe_intermediate_size=2048,
                n_routed_experts=256,
                n_shared_experts=None,
                num_experts_per_tok=8,
                norm_topk_prob=True,
                topk_method="noaux_tc",
                n_group=1,
                topk_group=1,
                scoring_func="sigmoid",
                routed_scaling_factor=2.5,
                hidden_act="silu",
            )
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device("cuda:0"):
                    moe = deepseek_v2.DeepseekV2MoE(
                        config,
                        parallel,
                        quant_config=runtime.Exl3Config(bits=3, codebook="mcg"),
                        reduce_results=True,
                        prefix=f"model.layers.{args.layer}.mlp",
                    )
            finally:
                torch.set_default_dtype(old_dtype)
            routed = moe.experts.routed_experts
            routed.quant_method.process_weights_after_loading(routed)
            store = routed._exl3_lazy_experts
            offset, width = rank_geometry(args.layer, args.rank)
            routed_parameters = [
                (name, list(parameter.shape), parameter.numel() * parameter.element_size())
                for name, parameter in routed.named_parameters()
            ]
            expected_router_reference = [("e_score_correction_bias", [256], 1024)]
            if (
                routed._exl3_intermediate_local != width
                or routed._exl3_intermediate_offset != offset
                or routed_parameters != expected_router_reference
            ):
                raise AssertionError(
                    "lazy routed allocation/geometry differs: "
                    f"local={routed._exl3_intermediate_local}/{width} "
                    f"offset={routed._exl3_intermediate_offset}/{offset} "
                    f"parameters={routed_parameters}"
                )

            generator = torch.Generator(device="cpu").manual_seed(909090)
            hidden = torch.randn(
                (4, 6144), generator=generator, dtype=torch.float32
            ).bfloat16().cuda()
            ids = torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [8, 9, 10, 11, 12, 13, 14, 15],
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [8, 9, 10, 11, 12, 13, 14, 15],
                ],
                dtype=torch.long,
                device="cuda",
            )
            weights = torch.full(
                (4, 8), 2.5 / 8.0, dtype=torch.float32, device="cuda"
            )
            reference_local = runtime.apply_exl3_python_loop(
                hidden, ids, weights, store, None, float(runtime.SWIGLU_LIMIT_DEFAULT)
            ).to(dtype=hidden.dtype).float()
            lazy_local = routed.forward_modular(hidden, weights, ids, None, None).float()
            relative = float(
                (lazy_local - reference_local).norm().item()
                / max(reference_local.norm().item(), 1e-30)
            )
            # The fused EXL3 kernel accumulates in its qualified FP16 path,
            # while the reference loop accumulates each expert contribution
            # in FP32. Use the same 1% relative-L2 contract as the established
            # full-resident fused/reference gate.
            if not math.isfinite(relative) or relative > 0.01:
                raise AssertionError(f"lazy fused/reference local parity failed: {relative}")

            reference_global = tensor_model_parallel_all_reduce(reference_local.clone())
            lazy_global = tensor_model_parallel_all_reduce(lazy_local.clone())
            nccl_relative = float(
                (lazy_global - reference_global).norm().item()
                / max(reference_global.norm().item(), 1e-30)
            )
            if not math.isfinite(nccl_relative) or nccl_relative > 0.01:
                raise AssertionError(f"lazy fused/reference NCCL parity failed: {nccl_relative}")
            gathered = [torch.empty_like(lazy_global) for _ in range(3)]
            dist.all_gather(gathered, lazy_global)
            cross_rank_max = max(
                float((item - gathered[0]).abs().max().item()) for item in gathered
            )
            if cross_rank_max != 0.0:
                raise AssertionError("NCCL reduced lazy output differs across ranks")

            moe.gate.weight.data.zero_()
            bias = moe.gate.e_score_correction_bias
            bias.data.fill_(-8.0)
            bias.data[:8].fill_(8.0)
            model_ids = torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7]] * hidden.shape[0],
                dtype=torch.long,
                device="cuda",
            )
            model_local = routed.forward_modular(
                hidden, weights, model_ids, None, None
            ).float()
            model_global = tensor_model_parallel_all_reduce(model_local)
            with set_forward_context(None, vllm_config, num_tokens=hidden.shape[0]):
                actual_model_path = moe(hidden).float()
            model_relative = float(
                (actual_model_path - model_global).norm().item()
                / max(model_global.norm().item(), 1e-30)
            )
            if not math.isfinite(model_relative) or model_relative > 0.01:
                raise AssertionError(
                    f"actual Glm lazy MoE runner parity failed: {model_relative}"
                )

            # Fill the configured cache and then force a disjoint expert set.
            # This proves deterministic turnover for both the small diagnostic
            # caches and the larger measured serving cache. The chunked
            # implementation must retain device slots but close the rank-pack
            # mapping after every layer call, making file-backed pages
            # reclaimable.
            if os.environ["GLM53_LAZY_K3_EXECUTION"] == "chunked_fused":
                cache_limit = int(
                    os.environ["GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER"]
                )
                for start in range(16, cache_limit, 8):
                    fill = list(range(start, min(start + 8, cache_limit)))
                    fill += [fill[-1]] * (8 - len(fill))
                    fill_ids = torch.tensor(
                        [fill] * hidden.shape[0],
                        dtype=torch.long,
                        device="cuda",
                    )
                    routed.forward_modular(hidden, weights, fill_ids, None, None)
                turnover_start = cache_limit
                turnover_ids = torch.tensor(
                    [list(range(turnover_start, turnover_start + 8))]
                    * hidden.shape[0],
                    dtype=torch.long,
                    device="cuda",
                )
                routed.forward_modular(hidden, weights, turnover_ids, None, None)
                if store.packed_handle is not None or store.packed_keys is not None:
                    raise AssertionError("chunked rank-pack mapping remained live")

            handoff = None
            if os.environ["GLM53_LAZY_K3_EXECUTION"] == "shared_layer_fused":
                sibling_layer = args.layer + 3
                old_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    with torch.device("cuda:0"):
                        sibling_moe = deepseek_v2.DeepseekV2MoE(
                            config,
                            parallel,
                            quant_config=runtime.Exl3Config(bits=3, codebook="mcg"),
                            reduce_results=True,
                            prefix=f"model.layers.{sibling_layer}.mlp",
                        )
                finally:
                    torch.set_default_dtype(old_dtype)
                sibling_routed = sibling_moe.experts.routed_experts
                sibling_routed.quant_method.process_weights_after_loading(sibling_routed)
                sibling_local = sibling_routed.forward_modular(
                    hidden, weights, ids, None, None
                ).float()
                first_again = routed.forward_modular(
                    hidden, weights, ids, None, None
                ).float()
                reload_relative = float(
                    (first_again - lazy_local).norm().item()
                    / max(lazy_local.norm().item(), 1e-30)
                )
                sibling_difference = float(
                    (sibling_local - lazy_local).norm().item()
                    / max(lazy_local.norm().item(), 1e-30)
                )
                # Reloading the same payload into the same fixed arena may
                # change the fused FP16 reduction by a few ULPs. This return
                # gate is still 100x tighter than the qualified 1% fused
                # reference contract.
                if not math.isfinite(reload_relative) or reload_relative > 1e-4:
                    raise AssertionError(
                        f"shared arena layer return parity failed: {reload_relative}"
                    )
                if not math.isfinite(sibling_difference) or sibling_difference < 1e-3:
                    raise AssertionError(
                        f"shared arena sibling payload did not change: {sibling_difference}"
                    )
                handoff = {
                    "first_layer": args.layer,
                    "sibling_layer": sibling_layer,
                    "return_relative_l2": reload_relative,
                    "sibling_relative_difference": sibling_difference,
                    "sibling_cache": sibling_routed._exl3_lazy_experts.summary(),
                }
            stats = store.summary()
            execution = os.environ["GLM53_LAZY_K3_EXECUTION"]
            if stats["loads"] < 16 or stats["payload_bytes"] <= 0:
                raise AssertionError(f"lazy cache behavior differs: {stats}")
            if execution == "chunked_fused" and stats["evictions"] <= 0:
                raise AssertionError(f"chunked cache did not evict: {stats}")
            if execution == "resident_fused":
                if (
                    stats["loads"] != 256
                    or stats["evictions"] != 0
                    # The active geometry intentionally rotates the load order.
                    # Validate membership and cardinality, not numeric ordering.
                    or len(stats["resident_experts"]) != 256
                    or set(stats["resident_experts"]) != set(range(256))
                    or stats["fully_resident"] is not True
                    or stats["rank_local_pack"] is not True
                    or store.packed_handle is not None
                    or store.packed_keys is not None
                ):
                    raise AssertionError(f"full resident cache differs: {stats}")
                resident_loads = stats["loads"]
                resident_io = stats["payload_bytes"]
                second_resident = routed.forward_modular(
                    hidden, weights, ids, None, None
                ).float()
                repeat_relative = float(
                    (second_resident - lazy_local).norm().item()
                    / max(lazy_local.norm().item(), 1e-30)
                )
                after_repeat = store.summary()
                if (
                    after_repeat["loads"] != resident_loads
                    or after_repeat["payload_bytes"] != resident_io
                    or after_repeat["evictions"] != 0
                    or after_repeat["resident_apply_calls"] < 2
                    or repeat_relative > 1e-4
                ):
                    raise AssertionError(
                        "resident repeat touched storage or changed output: "
                        f"relative={repeat_relative} stats={after_repeat}"
                    )
                stats = after_repeat
            expected_apply = {
                "chunked_fused": "lazy_nvme_chunked_fused",
                "shared_layer_fused": "lazy_nvme_shared_layer_fused",
                "resident_fused": "fully_resident_fused",
                "resident_uva": "fully_resident_fused",
            }[execution]
            if routed._exl3_last_apply != expected_apply:
                raise AssertionError(
                    f"lazy execution path differs: {routed._exl3_last_apply}/{expected_apply}"
                )
            result.update(
                passed=True,
                elapsed_seconds=time.monotonic() - started,
                geometry={
                    "semantic_intermediate": 2048,
                    "local_intermediate": width,
                    "offset": offset,
                    "wide_rank": layer_geometry(args.layer)["wide_rank"],
                    "experts_addressable": len(store),
                    "top_k": 8,
                    "padding_channels": 0,
                },
                reference_fused_local_relative_l2=relative,
                reference_fused_nccl_relative_l2=nccl_relative,
                nccl_cross_rank_max_abs=cross_rank_max,
                actual_glm_moe_runner_relative_l2=model_relative,
                shared_layer_handoff=handoff,
                cache=stats,
                last_apply=routed._exl3_last_apply,
                cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            )
            dist.barrier()
    except BaseException as error:
        result.update(
            passed=False,
            exception=repr(error),
            elapsed_seconds=time.monotonic() - started,
        )
        raise
    finally:
        path = args.output_dir / f"rank-{args.rank}.json"
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if dist.is_initialized():
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass


if __name__ == "__main__":
    main()
