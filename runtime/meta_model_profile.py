#!/usr/bin/env python3
"""Run vLLM far enough to emit the opt-in meta model allocation profile."""

from __future__ import annotations

import os
import json
from pathlib import Path


def main() -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    output = Path(os.environ["GLM53_META_PROFILE_OUTPUT"])
    placement = os.environ.get("GLM53_META_PROFILE_PLACEMENT", "tp3")
    if placement in ("tp3", "tp3-lazy"):
        tensor_parallel_size, pipeline_parallel_size = 3, 1
    elif placement == "pp3":
        tensor_parallel_size, pipeline_parallel_size = 1, 3
    else:
        raise ValueError(f"unknown meta profile placement {placement!r}")
    try:
        LLM(
            model=os.environ.get("MODEL_DIR", "/model"),
            served_model_name="GLM-5.3-K3-TP3-META-PROFILE",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            decode_context_parallel_size=1,
            distributed_executor_backend="external_launcher",
            quantization="exl3",
            load_format="safetensors",
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=0.86,
            max_num_seqs=1,
            max_num_batched_tokens=1024,
            kv_cache_dtype="fp8",
            enable_prefix_caching=False,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=False,
            seed=20260906,
            chat_template="/model/chat_template.jinja",
        )
    except BaseException as error:
        if output.is_file():
            profile = json.loads(output.read_text(encoding="utf-8"))
            if profile.get("passed"):
                print(
                    "GLM53_META_PROFILE_RUNNER_COMPLETE "
                    f"rank={profile.get('rank')} error={type(error).__name__}",
                    flush=True,
                )
                return
        raise
    raise RuntimeError("meta profile unexpectedly returned a live model")


if __name__ == "__main__":
    main()
