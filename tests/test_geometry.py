from __future__ import annotations

import unittest

from scripts.tensor_plan import routed_coordinate
from tp3k3.encoder import physical_slice
from tp3k3.geometry import (
    GEOMETRY_ID,
    INDEX_FREQ,
    INDEX_TOP_K,
    MTP_LAYER,
    PAD_CHANNELS,
    PHYSICAL_INTERMEDIATE,
    ROUTED_LAYERS,
    SEMANTIC_INTERMEDIATE,
    TOP_K,
    checkpoint_geometry,
    checkpoint_key,
    layer_geometry,
    rank_offsets,
    rank_widths,
    slice_spec,
    wide_rank,
)


class GeometryTests(unittest.TestCase):
    class _SliceProbe:
        def __init__(self):
            self.keys = []

        def __getitem__(self, key):
            self.keys.append(key)
            return self

        def contiguous(self):
            return self.keys[-1]

    def test_model_contract(self):
        self.assertEqual(len(ROUTED_LAYERS), 76)
        self.assertEqual((ROUTED_LAYERS[0], ROUTED_LAYERS[-1]), (3, 78))
        self.assertEqual((TOP_K, INDEX_TOP_K, INDEX_FREQ), (8, 2048, 4))
        self.assertEqual(PHYSICAL_INTERMEDIATE, SEMANTIC_INTERMEDIATE)
        self.assertEqual(PAD_CHANNELS, 0)

    def test_all_three_normal_rotations(self):
        self.assertEqual(rank_widths(3), (768, 640, 640))
        self.assertEqual(rank_offsets(3), (0, 768, 1408))
        self.assertEqual(rank_widths(4), (640, 768, 640))
        self.assertEqual(rank_offsets(4), (0, 640, 1408))
        self.assertEqual(rank_widths(5), (640, 640, 768))
        self.assertEqual(rank_offsets(5), (0, 640, 1280))
        self.assertEqual(rank_widths(6), rank_widths(3))

    def test_every_layer_has_exact_aligned_coverage(self):
        for layer in ROUTED_LAYERS:
            widths = rank_widths(layer)
            offsets = rank_offsets(layer)
            self.assertEqual(sorted(widths), [640, 640, 768])
            self.assertEqual(sum(widths), 2048)
            self.assertEqual(offsets[0], 0)
            for rank in range(3):
                self.assertEqual(widths[rank] % 128, 0)
                self.assertEqual(offsets[rank] % 128, 0)
                if rank:
                    self.assertEqual(offsets[rank], offsets[rank - 1] + widths[rank - 1])
            self.assertEqual(offsets[2] + widths[2], 2048)

    def test_rotation_balance_and_mtp_override(self):
        normal_counts = {rank: 0 for rank in range(3)}
        for layer in range(3, 78):
            normal_counts[wide_rank(layer)] += 1
        self.assertEqual(normal_counts, {0: 25, 1: 25, 2: 25})
        self.assertEqual(wide_rank(MTP_LAYER), 2)
        total_counts = dict(normal_counts)
        total_counts[wide_rank(MTP_LAYER)] += 1
        self.assertEqual(total_counts, {0: 25, 1: 25, 2: 26})

    def test_projection_slice_shapes_offsets_and_no_padding(self):
        for layer in (3, 4, 5, 78):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                specs = [slice_spec(layer, 0, projection, rank) for rank in range(3)]
                self.assertEqual(
                    [(item.semantic_start, item.semantic_stop) for item in specs],
                    [(offset, offset + width) for offset, width in zip(rank_offsets(layer), rank_widths(layer))],
                )
                self.assertTrue(all(item.pad_channels == 0 for item in specs))
                self.assertTrue(all(item.scale_mask is None for item in specs))
                for rank, item in enumerate(specs):
                    width = rank_widths(layer)[rank]
                    expected = (6144, width) if projection == "down_proj" else (width, 6144)
                    self.assertEqual(item.physical_shape, expected)
                    self.assertEqual(item.physical_channels, item.semantic_channels)

    def test_gate_up_and_down_slice_the_declared_axis_and_range(self):
        for layer in (3, 4, 5, 78):
            for rank, (offset, width) in enumerate(zip(rank_offsets(layer), rank_widths(layer))):
                for projection in ("gate_proj", "up_proj"):
                    probe = self._SliceProbe()
                    self.assertEqual(
                        physical_slice(probe, projection, layer, rank),
                        (slice(offset, offset + width), slice(None, None, None)),
                    )
                probe = self._SliceProbe()
                self.assertEqual(
                    physical_slice(probe, "down_proj", layer, rank),
                    (slice(None, None, None), slice(offset, offset + width)),
                )

    def test_checkpoint_metadata_records_every_rank_of_every_layer(self):
        metadata = checkpoint_geometry()
        self.assertEqual(metadata["geometry_id"], GEOMETRY_ID)
        self.assertEqual(set(metadata["layers"]), {str(layer) for layer in ROUTED_LAYERS})
        for layer in ROUTED_LAYERS:
            self.assertEqual(metadata["layers"][str(layer)], layer_geometry(layer))
            self.assertEqual(len(metadata["layers"][str(layer)]["ranks"]), 3)

    def test_seed_and_checkpoint_name_are_coordinate_bound_and_deterministic(self):
        keys = set()
        seeds = set()
        for layer in (3, 4, 5, 78):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for rank in range(3):
                    spec = slice_spec(layer, 255, projection, rank)
                    keys.add(checkpoint_key(layer, 255, projection, rank, "trellis"))
                    seeds.add(spec.seed)
                    self.assertEqual(spec.seed, slice_spec(layer, 255, projection, rank).seed)
        self.assertEqual(len(keys), 36)
        self.assertEqual(len(seeds), 36)

    def test_invalid_coordinates_fail_closed(self):
        for layer in (2, 79):
            with self.assertRaises(ValueError):
                rank_widths(layer)
        with self.assertRaises(ValueError):
            slice_spec(3, 0, "gate_proj", 3)

    def test_tensor_disposition_only_selects_routed_expert_weights(self):
        self.assertEqual(
            routed_coordinate("model.layers.78.mlp.experts.255.down_proj.weight"),
            (78, 255, "down_proj"),
        )
        self.assertIsNone(routed_coordinate("model.layers.2.mlp.experts.0.gate_proj.weight"))
        self.assertIsNone(routed_coordinate("model.layers.3.mlp.shared_experts.down_proj.weight"))
        self.assertIsNone(routed_coordinate("model.layers.3.mlp.experts.0.gate_proj.bias"))


if __name__ == "__main__":
    unittest.main()
