#!/usr/bin/env python3
"""Actual GLM MoE/EXL3 TP3 loader, fused path, and NCCL qualification."""

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
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--runtime-exl3", type=Path, required=True)
    parser.add_argument("--tp3-patch", type=Path, required=True)
    parser.add_argument("--numeric-core", type=Path, required=True)
    parser.add_argument("--numeric-core-sha256", required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.world_size != 3 or args.rank not in range(3):
        parser.error("this gate requires ranks 0..2 of TP=3")

    os.environ.update(
        MASTER_ADDR=args.master_addr,
        MASTER_PORT=str(args.master_port),
        WORLD_SIZE=str(args.world_size),
        RANK=str(args.rank),
        LOCAL_RANK="0",
        VLLM_GLM53_EXL3_TP3_RANK_SLICES="1",
        VLLM_GLM53_EXL3_TP3_GEOMETRY_ID="rotating-uneven-768-640-640-v1",
        EXL3_FUSED_MOE="1",
        VLLM_USE_V1="1",
    )
    import torch
    import torch.distributed as dist
    import vllm
    from r7_encoder.trellis import CodecConfig
    from tp3k3.encoder import IdentityR10Codec, encode_projection_rank
    from tp3k3.geometry import GEOMETRY_ID, layer_geometry, rank_geometry

    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    runtime = import_path("exl3", args.runtime_exl3)
    patch = import_path("tp3_rank_slice_patch", args.tp3_patch)

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )
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
        "schema": "glm53-full-exl3-tp3.distributed-runtime-gate.v2",
        "geometry_id": GEOMETRY_ID,
        "layer": args.layer,
        "rank": args.rank,
        "world_size": args.world_size,
        "host": socket.gethostname(),
        "architecture_class": "vllm.model_executor.models.deepseek_v2.GlmMoeDsaForCausalLM",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": getattr(vllm, "__version__", None),
            "gpu": torch.cuda.get_device_name(0),
        },
        "nccl_environment": {name: os.environ.get(name) for name in ("NCCL_IB_SUBNET_AWARE_ROUTING", "NCCL_NET_PLUGIN", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_SOCKET_IFNAME", "NCCL_DEBUG", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_BUFFSIZE")},
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
            initialize_model_parallel(tensor_model_parallel_size=3, pipeline_model_parallel_size=1)
            patch.apply_patches()
            if not issubclass(deepseek_v2.GlmMoeDsaForCausalLM, deepseek_v2.DeepseekV2ForCausalLM):
                raise AssertionError("GlmMoeDsaForCausalLM registry class is not the qualified implementation")
            config = SimpleNamespace(
                model_type="glm_moe_dsa",
                hidden_size=6144,
                moe_intermediate_size=2048,
                n_routed_experts=256,
                # This gate isolates the routed-expert component required by
                # Phase 1. Shared-expert BF16 TP3 padding/loading is outside
                # the K3 payload path and is not silently synthesized here.
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
            offset, local_width = rank_geometry(args.layer, args.rank)
            expected_geometry = layer_geometry(args.layer)
            if (
                routed._exl3_intermediate_local != local_width
                or routed.w13_svh.shape != (256, 2, local_width)
                or routed.w2_suh.shape != (256, local_width)
            ):
                raise AssertionError("physical rank geometry differs")
            if config.moe_intermediate_size != 2048 or moe.shared_experts is not None:
                raise AssertionError("routed-only synthetic component geometry differs")

            codec_config = CodecConfig(
                device="cuda:0",
                sigma_reg=0.025,
                numeric_core=args.numeric_core,
                numeric_core_sha256=args.numeric_core_sha256,
                extension=args.extension,
                extension_sha256=args.extension_sha256,
                verify_files=True,
            )
            codec = IdentityR10Codec(codec_config)
            generator = torch.Generator(device="cpu").manual_seed(2026090300 + args.rank)
            source_weights = {
                "gate_proj": torch.randn((2048, 6144), generator=generator, dtype=torch.float32).bfloat16(),
                "up_proj": torch.randn((2048, 6144), generator=generator, dtype=torch.float32).bfloat16(),
                "down_proj": torch.randn((6144, 2048), generator=generator, dtype=torch.float32).bfloat16(),
            }
            encoded = {}
            records = {}
            for projection, weight in source_weights.items():
                tensors, record = encode_projection_rank(codec, weight, layer=args.layer, expert=0, projection=projection, rank=args.rank)
                encoded[projection] = {key: value.cuda() for key, value in tensors.items()}
                records[projection] = record
            del source_weights, codec
            torch.cuda.empty_cache()

            def weights():
                # Include all serialized ranks so the actual loader must select
                # exactly one. Values for non-live ranks are never consumed.
                for expert in range(256):
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        for serialized_rank in range(3):
                            for suffix in ("trellis", "suh", "svh", "mcg"):
                                yield f"{expert}.{projection}.rank{serialized_rank}.{suffix}", encoded[projection][suffix]

            loaded_names = list(moe.experts.load_weights(weights()))
            expected_loads = 256 * 3 * 4
            if len(loaded_names) != expected_loads:
                raise AssertionError(f"rank-slice loader loaded {len(loaded_names)}, expected {expected_loads}")
            routed.quant_method.process_weights_after_loading(routed)
            geometry = routed._exl3_rank_geometry
            if (
                geometry["geometry_id"] != GEOMETRY_ID
                or geometry["rank"] != args.rank
                or geometry["layer"] != args.layer
                or geometry["offset"] != offset
                or geometry["local_width"] != local_width
                or geometry["padding_channels"] != 0
            ):
                raise AssertionError("runtime uneven geometry metadata differs")

            hgen = torch.Generator(device="cpu").manual_seed(909090)
            hidden = torch.randn((4, 6144), generator=hgen, dtype=torch.float32).bfloat16().cuda()
            ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]] * 4, dtype=torch.long, device="cuda")
            weights8 = torch.full((4, 8), 2.5 / 8.0, dtype=torch.float32, device="cuda")
            os.environ["EXL3_FUSED_MOE"] = "0"
            reference_local = routed.forward_modular(hidden, weights8, ids, None, None).float()
            os.environ["EXL3_FUSED_MOE"] = "1"
            fused_local = routed.forward_modular(hidden, weights8, ids, None, None).float()
            local_delta = fused_local - reference_local
            local_relative = float(local_delta.norm().item() / max(reference_local.norm().item(), 1e-30))
            if not math.isfinite(local_relative) or local_relative > 0.01:
                raise AssertionError(f"reference/fused local parity failed: {local_relative}")

            # Warmed fused execution must also be capturable with the local
            # rank's actual 640/768 buffer plan and replay identically.
            capture_hidden = hidden.clone()
            capture_ids = ids.clone()
            capture_weights = weights8.clone()
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                graphed_local = routed.forward_modular(
                    capture_hidden, capture_weights, capture_ids, None, None
                ).float()
            graph.replay()
            torch.cuda.synchronize()
            graph_relative = float(
                (graphed_local - fused_local).norm().item()
                / max(fused_local.norm().item(), 1e-30)
            )
            if not math.isfinite(graph_relative) or graph_relative > 0.01:
                raise AssertionError(f"eager/graphed fused parity failed: {graph_relative}")

            reference_global = tensor_model_parallel_all_reduce(reference_local.clone())
            fused_global = tensor_model_parallel_all_reduce(fused_local.clone())
            global_delta = fused_global - reference_global
            global_relative = float(global_delta.norm().item() / max(reference_global.norm().item(), 1e-30))
            if not math.isfinite(global_relative) or global_relative > 0.01:
                raise AssertionError(f"reference/fused NCCL parity failed: {global_relative}")
            gathered = [torch.empty_like(fused_global) for _ in range(3)]
            dist.all_gather(gathered, fused_global)
            cross_rank_max = max(float((item - gathered[0]).abs().max().item()) for item in gathered)
            if cross_rank_max != 0.0:
                raise AssertionError("NCCL-reduced output differs across TP ranks")

            # Exercise the actual GlmMoeDsa component runner and its late TP
            # all-reduce. Identical experts make routing IDs irrelevant; a zero
            # gate gives uniform top-8 weights matching weights8 above.
            moe.gate.weight.data.zero_()
            if moe.gate.e_score_correction_bias is not None:
                moe.gate.e_score_correction_bias.data.zero_()
            with set_forward_context(None, vllm_config, num_tokens=hidden.shape[0]):
                actual_model_path = moe(hidden).float()
            model_relative = float((actual_model_path - fused_global).norm().item() / max(fused_global.norm().item(), 1e-30))
            if not math.isfinite(model_relative) or model_relative > 0.01:
                raise AssertionError(f"GlmMoeDsa MoE runner differs from qualified fused reduction: {model_relative}")

            result.update(
                passed=True,
                elapsed_seconds=time.monotonic() - started,
                geometry={
                    "experts": 256,
                    "top_k": 8,
                    "hidden": 6144,
                    "semantic_intermediate": 2048,
                    "physical_intermediate": 2048,
                    "local_intermediate": local_width,
                    "offset": offset,
                    "wide_rank": expected_geometry["wide_rank"],
                    "padding_channels": 0,
                },
                loaded_parameter_count=len(loaded_names),
                runtime_rank_geometry=geometry,
                routed_only_component=True,
                reference_fused_local_relative_l2=local_relative,
                eager_graphed_fused_relative_l2=graph_relative,
                reference_fused_nccl_relative_l2=global_relative,
                nccl_cross_rank_max_abs=cross_rank_max,
                actual_glm_moe_runner_relative_l2=model_relative,
                fused_last_apply=getattr(routed, "_exl3_last_apply", None),
                encode_records=records,
                cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            )
            dist.barrier()
    except BaseException as exc:
        result.update(passed=False, exception=repr(exc), elapsed_seconds=time.monotonic() - started)
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
