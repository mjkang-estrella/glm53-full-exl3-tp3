from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from scripts.matched_kld import parse_full_logprobs, safetensor_f32
from runtime.tp3_logit_capture_patch import _capture, _write_part
from scripts.capture_matched_logits import normalize_scheduler_singletons, remote_capture_root


def test_parse_full_logprobs_uses_token_ids() -> None:
    response = {
        "choices": [{
            "token_ids": [2],
            "logprobs": {"top_logprobs": [{
                "token_id:0": -3.0,
                "token_id:1": -2.0,
                "token_id:2": -1.0,
            }]},
        }],
    }
    values, generated = parse_full_logprobs(response, 3)
    assert generated == 2
    assert values.tolist() == [-3.0, -2.0, -1.0]


def test_parse_full_logprobs_rejects_incomplete_vocab() -> None:
    response = {
        "choices": [{
            "token_ids": [0],
            "logprobs": {"top_logprobs": [{"token_id:0": -1.0}]},
        }],
    }
    with pytest.raises(ValueError, match="missing/nonfinite"):
        parse_full_logprobs(response, 2)


def test_safetensor_f32_is_bounded_memmap(tmp_path) -> None:
    array = np.arange(12, dtype="<f4").reshape(3, 4)
    header = json.dumps(
        {"logits": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]}},
        separators=(",", ":"),
    ).encode()
    path = tmp_path / "teacher.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + array.tobytes())
    loaded = safetensor_f32(path)
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, array)


def test_raw_capture_part_is_exact_and_hashed(tmp_path) -> None:
    array = np.arange(24, dtype="<f4").reshape(2, 12)
    path = tmp_path / "part.f32"
    size, digest = _write_part(path, array)
    assert size == array.nbytes
    assert digest == __import__("hashlib").sha256(array.tobytes()).hexdigest()
    np.testing.assert_array_equal(np.memmap(path, mode="r", dtype="<f4", shape=array.shape), array)


def test_armed_raw_capture_seals_without_overwrite(tmp_path) -> None:
    import torch

    arm = {
        "schema": "glm53-full-exl3-tp3.logit-capture-arm.v1",
        "capture_id": "selftest",
        "window_id": "synthetic",
        "token_sha256": "0" * 64,
        "model": "synthetic",
        "expected_rows": 2,
        "expected_vocab": 154880,
    }
    (tmp_path / "ARM.json").write_text(json.dumps(arm))
    logits = torch.arange(2 * 154880, dtype=torch.float32).reshape(2, 154880)
    _capture(tmp_path, logits)
    final = tmp_path / "selftest"
    receipt = json.loads((final / "CAPTURE_COMPLETE.json").read_text())
    assert receipt["passed"] is True
    assert receipt["expected_rows"] == 2
    assert receipt["bytes"] == logits.numel() * logits.element_size()
    assert not (tmp_path / "ARM.json").exists()
    assert (tmp_path / "ARM.consumed-selftest.json").is_file()


def test_matched_capture_uses_attempt_scoped_mount() -> None:
    assert remote_capture_root("20260906T034145Z", "attempt-v8") == (
        "/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/"
        "20260906T034145Z/candidate/attempts/attempt-v8/rank-0/capture"
    )
    with pytest.raises(ValueError, match="unsafe"):
        remote_capture_root("20260906T034145Z", "../wrong")


def test_scheduler_singleton_normalization_keeps_prompt_batches() -> None:
    parts = [
        {"path": f"part-{index:04d}.f32", "rows": rows, "bytes": rows * 4, "first_row": sum([1, 1024, 1, 1023][:index])}
        for index, rows in enumerate((1, 1024, 1, 1023))
    ]
    normalized = normalize_scheduler_singletons(
        {"expected_rows": 2049, "bytes": 2049 * 4, "parts": parts}
    )
    assert normalized["expected_rows"] == 2047
    assert [part["path"] for part in normalized["parts"]] == [
        "part-0001.f32", "part-0003.f32"
    ]
    assert [part["first_row"] for part in normalized["parts"]] == [0, 1024]
    assert normalized["scheduler_normalization"]["dropped_singleton_part_indices"] == [0, 2]
