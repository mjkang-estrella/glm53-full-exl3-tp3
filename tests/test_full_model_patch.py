from __future__ import annotations

import torch

from runtime.tp3_full_model_patch import (
    PHYSICAL_HEADS,
    PHYSICAL_SHARED_INTERMEDIATE,
    PHYSICAL_VOCAB,
    SEMANTIC_HEADS,
    SEMANTIC_SHARED_INTERMEDIATE,
    SEMANTIC_VOCAB,
    _pad_dim,
    _parse_k3_weight,
)


def test_full_model_physical_boundaries() -> None:
    assert (SEMANTIC_HEADS, PHYSICAL_HEADS) == (64, 66)
    assert PHYSICAL_HEADS % 3 == 0
    assert (SEMANTIC_VOCAB, PHYSICAL_VOCAB) == (154880, 154944)
    assert PHYSICAL_VOCAB % (3 * 64) == 0
    assert (SEMANTIC_SHARED_INTERMEDIATE, PHYSICAL_SHARED_INTERMEDIATE) == (2048, 2304)
    assert PHYSICAL_SHARED_INTERMEDIATE % (3 * 128) == 0


def test_pad_dim_preserves_semantic_prefix_and_zeros_tail() -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    padded = _pad_dim(source, 1, 6)
    assert torch.equal(padded[:, :4], source)
    assert torch.count_nonzero(padded[:, 4:]) == 0


def test_pad_dim_never_truncates() -> None:
    source = torch.ones(2, 3)
    try:
        _pad_dim(source, 1, 2)
    except ValueError as exc:
        assert "refusing to truncate" in str(exc)
    else:
        raise AssertionError("truncation was not rejected")


def test_parse_k3_weight_for_nested_and_model_relative_names() -> None:
    terminal = "17.gate_proj.rank2.trellis"
    assert _parse_k3_weight(f"model.layers.5.mlp.experts.{terminal}") == (5, terminal)
    assert _parse_k3_weight(f"layers.5.mlp.experts.{terminal}") == (5, terminal)


def test_parse_k3_weight_does_not_capture_bf16_or_malformed_names() -> None:
    assert _parse_k3_weight("layers.5.mlp.shared_experts.gate_proj.weight") is None
    assert _parse_k3_weight("layers.5.mlp.experts.17.gate_proj.rank3.trellis") is None
    assert _parse_k3_weight("layers.5.mlp.experts.17.gate_proj.rank2.weight") is None
