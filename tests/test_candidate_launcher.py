from __future__ import annotations

import os
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_launcher(tmp_path: pathlib.Path, kv_bytes: str) -> list[str]:
    fake = tmp_path / "vllm"
    fake.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "MODEL_DIR": "/model",
        "SERVED_MODEL_NAME": "test",
        "PORT": "8893",
        "NODE_RANK": "0",
        "HEAD_IP": "192.0.2.1",
        "MASTER_PORT": "29654",
        "MAX_MODEL_LEN": "32768",
        "GPU_MEM_UTIL": "0.28",
        "MAX_NUM_SEQS": "1",
        "MAX_NUM_BATCHED_TOKENS": "1024",
        "KV_CACHE_DTYPE": "fp8",
        "KV_CACHE_MEMORY_BYTES": kv_bytes,
    }
    result = subprocess.run(
        ["bash", str(ROOT / "runtime" / "run_candidate_node.sh")],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return result.stdout.splitlines()


def test_candidate_launcher_passes_fixed_kv_bytes(tmp_path: pathlib.Path) -> None:
    args = _run_launcher(tmp_path, "3221225472")
    index = args.index("--kv-cache-memory-bytes")
    assert args[index + 1] == "3221225472"


def test_candidate_launcher_omits_unset_fixed_kv_bytes(tmp_path: pathlib.Path) -> None:
    args = _run_launcher(tmp_path, "0")
    assert "--kv-cache-memory-bytes" not in args


def test_cluster_launcher_supports_audited_cuda_malloc_async() -> None:
    launcher = (ROOT / "scripts" / "start_candidate_cluster_attempt.sh").read_text()
    assert 'cuda_allocator_conf=${13:-expandable_segments:True}' in launcher
    assert 'backend:cudaMallocAsync' in launcher
    assert 'expandable_segments:False' in launcher
    assert "-e PYTORCH_CUDA_ALLOC_CONF='$cuda_allocator_conf'" in launcher
    assert 'arena_block_slots=${14:-16}' in launcher
    assert "-e GLM53_LAZY_K3_ARENA_BLOCK_SLOTS='$arena_block_slots'" in launcher
    assert "scripts/evict_model_page_cache.py" in launcher
    assert "page-cache-eviction.json" in launcher
