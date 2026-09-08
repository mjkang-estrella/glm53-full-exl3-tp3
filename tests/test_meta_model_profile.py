from __future__ import annotations

import torch

from runtime.meta_model_profile_patch import summarize_tensors


def test_meta_profile_accounts_shapes_categories_and_layers() -> None:
    tensors = [
        (
            "model.layers.3.mlp.experts.routed_experts.w13_trellis",
            torch.empty((2, 3), dtype=torch.int16, device="meta"),
        ),
        (
            "model.layers.3.self_attn.o_proj.weight",
            torch.empty((2, 5), dtype=torch.bfloat16, device="meta"),
        ),
        (
            "model.embed_tokens.weight",
            torch.empty((7, 11), dtype=torch.bfloat16, device="meta"),
        ),
    ]
    rows, categories, layers, total = summarize_tensors(tensors)
    assert total == 12 + 20 + 154
    assert categories == {
        "bf16_attention": 20,
        "bf16_embedding": 154,
        "k3_routed_experts": 12,
    }
    assert layers == {"3": 32, "non_layer": 154}
    assert all(row["device"] == "meta" for row in rows)


def test_meta_profile_does_not_double_count_aliases_from_named_parameters() -> None:
    module = torch.nn.Module()
    shared = torch.nn.Parameter(torch.empty(13, dtype=torch.float16, device="meta"))
    module.register_parameter("first", shared)
    module.register_parameter("second", shared)
    rows, _, _, total = summarize_tensors(module.named_parameters())
    assert len(rows) == 1
    assert total == 26
