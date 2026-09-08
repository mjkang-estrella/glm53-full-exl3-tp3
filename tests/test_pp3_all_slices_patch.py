from __future__ import annotations

import torch
from torch import nn

from runtime.pp3_all_slices_patch import AllSlicesMoE, pipeline_layer_bounds
from tp3k3.geometry import layer_geometry, rank_offsets, rank_widths


class _Runner(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.routed_experts = nn.Identity()

    def forward(self, *, hidden_states, router_logits, input_ids=None, shared_experts_input=None):
        del router_logits, input_ids, shared_experts_input
        return torch.full_like(hidden_states, self.value)


def test_pipeline_partition_covers_every_layer_once() -> None:
    bounds = [pipeline_layer_bounds(78, 3, rank) for rank in range(3)]
    assert bounds == [(0, 26), (26, 52), (52, 78)]
    covered = [layer for start, end in bounds for layer in range(start, end)]
    assert covered == list(range(78))


def test_all_slice_wrapper_sums_exactly_three_partial_outputs() -> None:
    wrapper = AllSlicesMoE([_Runner(1.0), _Runner(2.0), _Runner(4.0)], layer=3)
    hidden = torch.zeros((2, 7), dtype=torch.float32)
    output = wrapper(hidden, hidden)
    torch.testing.assert_close(output, torch.full_like(hidden, 7.0), rtol=0, atol=0)
    assert wrapper.widths == rank_widths(3)
    assert wrapper.offsets == rank_offsets(3)


def test_each_pipeline_stage_has_all_three_rotations() -> None:
    for pipeline_rank in range(3):
        start, end = pipeline_layer_bounds(78, 3, pipeline_rank)
        moe_layers = range(max(3, start), end)
        wide = [layer_geometry(layer)["wide_rank"] for layer in moe_layers]
        assert set(wide) == {0, 1, 2}
        for layer in moe_layers:
            widths = rank_widths(layer)
            offsets = rank_offsets(layer)
            assert sum(widths) == 2048
            assert offsets[0] == 0
            assert offsets[1] == widths[0]
            assert offsets[2] == sum(widths[:2])
