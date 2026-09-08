"""Canonical rotating uneven GLM-5.3 routed-expert TP3 geometry."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass


SOURCE_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
GEOMETRY_ID = "rotating-uneven-768-640-640-v1"
ROUTED_LAYERS = tuple(range(3, 79))  # layer 78 is the MTP MoE layer
NORMAL_ROUTED_LAYERS = tuple(range(3, 78))
MTP_LAYER = 78
MTP_WIDE_RANK = 2
EXPERTS = 256
TOP_K = 8
INDEX_TOP_K = 2048
INDEX_FREQ = 4
HIDDEN = 6144
SEMANTIC_INTERMEDIATE = 2048
PHYSICAL_INTERMEDIATE = SEMANTIC_INTERMEDIATE
PAD_CHANNELS = 0
TP = 3
NARROW_SLICE = 640
WIDE_SLICE = 768
SUPPORTED_SLICE_WIDTHS = (NARROW_SLICE, WIDE_SLICE)
_bits_raw = os.environ.get("TP3K_BITS", "3")
try:
    BITS = int(_bits_raw)
except ValueError as exc:
    raise ValueError(f"TP3K_BITS must be an integer, got {_bits_raw!r}") from exc
if BITS not in (2, 3):
    raise ValueError(f"TP3K_BITS must be 2 or 3, got {BITS}")
SEED_BASE = 20260904


def wide_rank(layer: int) -> int:
    """Return the rank owning the 768-channel slice for a routed layer."""
    if layer not in ROUTED_LAYERS:
        raise ValueError("routed layer must be 3..78 inclusive")
    if layer == MTP_LAYER:
        return MTP_WIDE_RANK
    return (layer - NORMAL_ROUTED_LAYERS[0]) % TP


def rank_widths(layer: int) -> tuple[int, int, int]:
    widths = [NARROW_SLICE] * TP
    widths[wide_rank(layer)] = WIDE_SLICE
    return tuple(widths)


def rank_offsets(layer: int) -> tuple[int, int, int]:
    widths = rank_widths(layer)
    return (0, widths[0], widths[0] + widths[1])


def rank_geometry(layer: int, rank: int) -> tuple[int, int]:
    if rank not in range(TP):
        raise ValueError("rank must be 0..2")
    return rank_offsets(layer)[rank], rank_widths(layer)[rank]


def layer_geometry(layer: int) -> dict:
    widths = rank_widths(layer)
    offsets = rank_offsets(layer)
    return {
        "schema": "glm53-full-exl3-tp3.layer-geometry.v2",
        "geometry_id": GEOMETRY_ID,
        "layer": layer,
        "is_mtp": layer == MTP_LAYER,
        "wide_rank": wide_rank(layer),
        "semantic_intermediate": SEMANTIC_INTERMEDIATE,
        "physical_intermediate": PHYSICAL_INTERMEDIATE,
        "padding_channels": PAD_CHANNELS,
        "alignment": 128,
        "ranks": [
            {"rank": rank, "offset": offsets[rank], "width": widths[rank]}
            for rank in range(TP)
        ],
    }


def checkpoint_geometry() -> dict:
    """Return complete per-layer metadata required by assembly and loading."""
    return {
        "schema": "glm53-full-exl3-tp3.checkpoint-geometry.v2",
        "geometry_id": GEOMETRY_ID,
        "tp": TP,
        "semantic_intermediate": SEMANTIC_INTERMEDIATE,
        "physical_intermediate": PHYSICAL_INTERMEDIATE,
        "padding_channels": PAD_CHANNELS,
        "normal_rotation": [0, 1, 2],
        "mtp_layer": MTP_LAYER,
        "mtp_wide_rank": MTP_WIDE_RANK,
        "layers": {str(layer): layer_geometry(layer) for layer in ROUTED_LAYERS},
    }


def validate_layer_geometry(layer: int) -> None:
    widths = rank_widths(layer)
    offsets = rank_offsets(layer)
    if sorted(widths) != [NARROW_SLICE, NARROW_SLICE, WIDE_SLICE]:
        raise AssertionError(f"layer {layer}: invalid widths {widths}")
    if sum(widths) != SEMANTIC_INTERMEDIATE:
        raise AssertionError(f"layer {layer}: width coverage differs")
    if offsets[0] != 0:
        raise AssertionError(f"layer {layer}: first offset differs")
    for rank in range(TP):
        if widths[rank] % 128 or offsets[rank] % 128:
            raise AssertionError(f"layer {layer}: rank {rank} is not 128-aligned")
        if rank and offsets[rank] != offsets[rank - 1] + widths[rank - 1]:
            raise AssertionError(f"layer {layer}: rank {rank} gap or overlap")
    if offsets[-1] + widths[-1] != SEMANTIC_INTERMEDIATE:
        raise AssertionError(f"layer {layer}: terminal coverage differs")


@dataclass(frozen=True)
class SliceSpec:
    layer: int
    expert: int
    projection: str
    rank: int
    source_shape: tuple[int, int]
    physical_shape: tuple[int, int]
    semantic_start: int
    semantic_stop: int
    physical_channels: int
    semantic_channels: int
    pad_channels: int
    padding_axis: int
    scale_mask: str | None
    wide_rank: int
    geometry_id: str

    @property
    def key(self) -> str:
        return f"L{self.layer}.E{self.expert}.{self.projection}.rank{self.rank}"

    @property
    def seed(self) -> int:
        material = (
            f"{GEOMETRY_ID}:{SEED_BASE}:{self.layer}:{self.expert}:"
            f"{self.projection}:{self.rank}:{self.semantic_start}:{self.physical_channels}"
        )
        return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") & 0x7FFF_FFFF

    def to_dict(self) -> dict:
        value = asdict(self)
        value["key"] = self.key
        value["seed"] = self.seed
        return value


def slice_spec(layer: int, expert: int, projection: str, rank: int) -> SliceSpec:
    validate_layer_geometry(layer)
    if expert not in range(EXPERTS):
        raise ValueError("expert must be 0..255")
    if projection not in {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("unsupported projection")
    start, width = rank_geometry(layer, rank)
    stop = start + width
    if projection == "down_proj":
        source_shape = (HIDDEN, SEMANTIC_INTERMEDIATE)
        physical_shape = (HIDDEN, width)
        axis = 1
    else:
        source_shape = (SEMANTIC_INTERMEDIATE, HIDDEN)
        physical_shape = (width, HIDDEN)
        axis = 0
    return SliceSpec(
        layer=layer,
        expert=expert,
        projection=projection,
        rank=rank,
        source_shape=source_shape,
        physical_shape=physical_shape,
        semantic_start=start,
        semantic_stop=stop,
        physical_channels=width,
        semantic_channels=width,
        pad_channels=0,
        padding_axis=axis,
        scale_mask=None,
        wide_rank=wide_rank(layer),
        geometry_id=GEOMETRY_ID,
    )


def checkpoint_key(layer: int, expert: int, projection: str, rank: int, suffix: str) -> str:
    if suffix not in {"trellis", "suh", "svh", "mcg"}:
        raise ValueError("unsupported EXL3 tensor suffix")
    slice_spec(layer, expert, projection, rank)
    return f"model.layers.{layer}.mlp.experts.{expert}.{projection}.rank{rank}.{suffix}"


for _layer in ROUTED_LAYERS:
    validate_layer_geometry(_layer)
