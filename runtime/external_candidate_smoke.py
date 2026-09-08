#!/usr/bin/env python3
"""Single-process-per-rank external-launcher full-model generation smoke."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import tempfile
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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
        os.chmod(path, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def main() -> None:
    # ExecutorWithExternalLauncher requires the engine and worker in this
    # process. Set this before importing vLLM.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    rank = int(os.environ["RANK"])
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))
    max_num_batched_tokens = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "1024"))
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.86"))
    max_output_tokens = int(os.environ.get("SMOKE_MAX_OUTPUT_TOKENS", "16"))
    kv_cache_memory_bytes = int(os.environ.get("KV_CACHE_MEMORY_BYTES", "0"))
    cpu_offload_gb = float(os.environ.get("CPU_OFFLOAD_GB", "0"))
    cpu_offload_params = {
        item.strip()
        for item in os.environ.get("CPU_OFFLOAD_PARAMS", "").split(",")
        if item.strip()
    }
    force_empty_think = os.environ.get("SMOKE_FORCE_EMPTY_THINK", "0") == "1"
    output = Path(os.environ["OUTPUT_JSON"])
    started = utc_now()
    started_monotonic = time.monotonic()
    base = {
        "schema": "glm53-full-exl3-tp3.external-candidate-smoke.v1",
        "rank": rank,
        "world_size": int(os.environ["WORLD_SIZE"]),
        "host": socket.gethostname(),
        "started_at": started,
        "profile": {
            "tp": 3,
            "dcp": 1,
            "pp": 1,
            "max_model_len": max_model_len,
            "max_num_seqs": 1,
            "max_num_batched_tokens": max_num_batched_tokens,
            "kv_cache_dtype": "fp8",
            "kv_cache_memory_bytes": kv_cache_memory_bytes,
            "cpu_offload_gb": cpu_offload_gb,
            "cpu_offload_params": sorted(cpu_offload_params),
            "graphs": False,
            "mtp": False,
            "prefix_cache": False,
            "distributed_executor_backend": "external_launcher",
            "v1_multiprocessing": False,
            "force_empty_think": force_empty_think,
        },
        "nccl": {
            name: os.environ.get(name)
            for name in ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_BUFFSIZE")
        },
    }
    try:
        llm_options = {}
        if kv_cache_memory_bytes:
            llm_options["kv_cache_memory_bytes"] = kv_cache_memory_bytes
        if cpu_offload_gb > 0:
            llm_options["cpu_offload_gb"] = cpu_offload_gb
        if cpu_offload_params:
            llm_options["cpu_offload_params"] = cpu_offload_params
        llm = LLM(
            model=os.environ.get("MODEL_DIR", "/model"),
            served_model_name="GLM-5.3-K3-TP3-EXTERNAL-SMOKE",
            tensor_parallel_size=3,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            distributed_executor_backend="external_launcher",
            quantization="exl3",
            load_format="safetensors",
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=1,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_cache_dtype="fp8",
            enable_prefix_caching=False,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=False,
            seed=20260906,
            max_logprobs=-1,
            chat_template="/model/chat_template.jinja",
            **llm_options,
        )
        loaded_at = utc_now()
        available_after_load = mem_available()
        resident_before = None
        if os.environ.get("GLM53_LAZY_K3_EXECUTION") in {"resident_fused", "resident_uva"}:
            from lazy_k3_patch import resident_census

            resident_before = resident_census()
            if not (
                resident_before["all_fully_resident"] is True
                and resident_before["stores"] == 75
                and resident_before["layers"] == list(range(3, 78))
                and resident_before["experts_resident"] == 75 * 256
                and resident_before["loads"] == 75 * 256
                and resident_before["evictions"] == 0
                and resident_before["open_rank_pack_handles"] == 0
            ):
                raise RuntimeError(
                    f"full resident model census differs: {resident_before}"
                )
        messages = [
            {
                "role": "user",
                "content": "Reply with exactly the capital city of France and nothing else.",
            }
        ]
        tokenizer = llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if force_empty_think:
            if not prompt.endswith("<think>"):
                raise RuntimeError("chat template generation prompt no longer ends in <think>")
            # The pinned source template always starts a reasoning block and
            # ignores enable_thinking. Closing the empty block is the template's
            # native non-reasoning form for deterministic correctness probes.
            prompt += "</think>\n"
        sampling = SamplingParams(
            temperature=0.0, max_tokens=max_output_tokens, seed=20260906
        )
        generated_started = time.monotonic()
        requests = llm.generate([prompt], sampling, use_tqdm=False)
        generated_seconds = time.monotonic() - generated_started
        if len(requests) != 1 or len(requests[0].outputs) != 1:
            raise RuntimeError("external smoke returned unexpected request/output count")
        completion = requests[0].outputs[0]
        text = completion.text
        token_ids = list(completion.token_ids)
        finish_reason = completion.finish_reason
        passed = bool(text.strip()) and finish_reason in {"stop", "length"}
        resident_after = None
        if resident_before is not None:
            from lazy_k3_patch import resident_census

            resident_after = resident_census()
            stable_fields = (
                "stores",
                "layers",
                "experts_resident",
                "loads",
                "evictions",
                "payload_bytes_read",
                "arena_bytes_declared",
                "unique_storage_allocations",
                "unique_storage_bytes",
                "open_rank_pack_handles",
            )
            if any(resident_after[key] != resident_before[key] for key in stable_fields):
                raise RuntimeError("resident storage changed during generation")
            if resident_after["resident_apply_calls"] <= resident_before["resident_apply_calls"]:
                raise RuntimeError("generation did not execute resident expert layers")
        result = {
            **base,
            "passed": passed,
            "loaded_at": loaded_at,
            "completed_at": utc_now(),
            "load_and_generate_seconds": time.monotonic() - started_monotonic,
            "generation_seconds": generated_seconds,
            "host_available_after_load_bytes": available_after_load,
            "host_available_after_generate_bytes": mem_available(),
            "prompt": prompt,
            "prompt_token_ids": list(requests[0].prompt_token_ids),
            "text": text,
            "token_ids": token_ids,
            "finish_reason": finish_reason,
            "resident_before_generation": resident_before,
            "resident_after_generation": resident_after,
        }
        atomic_json(output, result)
        print(
            "GLM53_EXTERNAL_SMOKE_GENERATED "
            f"rank={rank} finish={finish_reason} tokens={len(token_ids)} "
            f"text={json.dumps(text, ensure_ascii=False)}",
            flush=True,
        )
        if not passed:
            raise RuntimeError("external smoke completion was empty or abnormal")
    except BaseException as error:
        atomic_json(
            output.with_name("FAILED.json"),
            {
                **base,
                "passed": False,
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "host_available_bytes": mem_available(),
            },
        )
        raise


if __name__ == "__main__":
    main()
