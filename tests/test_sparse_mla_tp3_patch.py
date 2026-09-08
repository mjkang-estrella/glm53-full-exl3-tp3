from __future__ import annotations

import pytest
import torch

from runtime.sparse_mla_tp3_patch import (
    GENERIC_MIN_TOKENS,
    KERNEL_HEADS,
    LOCAL_TP3_HEADS,
    _pad_head_axis,
    _pad_token_axis,
    _requires_head_adapter,
)


def test_head_padding_preserves_every_semantic_value_and_zeroes_only_tail():
    source = torch.arange(3 * LOCAL_TP3_HEADS * 7).reshape(
        3, LOCAL_TP3_HEADS, 7
    )
    padded = _pad_head_axis(source)
    assert padded.shape == (3, KERNEL_HEADS, 7)
    assert torch.equal(padded[:, :LOCAL_TP3_HEADS], source)
    assert torch.count_nonzero(padded[:, LOCAL_TP3_HEADS:]) == 0
    assert torch.equal(padded[:, :LOCAL_TP3_HEADS], source)


def test_token_shadow_padding_is_independent_and_typed():
    source = torch.arange(4 * 5, dtype=torch.int32).reshape(4, 5)
    padded = _pad_token_axis(source, GENERIC_MIN_TOKENS, 1)
    assert padded.shape == (GENERIC_MIN_TOKENS, 5)
    assert torch.equal(padded[:4], source)
    assert torch.all(padded[4:] == 1)
    assert padded.dtype == source.dtype


def test_padding_refuses_head_truncation():
    with pytest.raises(ValueError, match="refusing to truncate"):
        _pad_head_axis(torch.zeros(1, KERNEL_HEADS + 1, 2))


@pytest.mark.parametrize("tokens", [1, 64, 65, 96, 1024])
def test_tp3_head_adapter_covers_decode_and_prefill(tokens):
    query = torch.zeros(tokens, LOCAL_TP3_HEADS, 576)
    assert _requires_head_adapter(LOCAL_TP3_HEADS, query)
    assert not _requires_head_adapter(KERNEL_HEADS, query)
    assert not _requires_head_adapter(LOCAL_TP3_HEADS, query[0])
