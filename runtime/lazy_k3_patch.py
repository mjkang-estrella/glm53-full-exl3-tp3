"""Exact K3 expert storage for the sealed TP3 model.

All non-routed tensors remain resident in their checkpoint dtype. Routed K3
payloads remain on the independently verified local NVMe replica and only the
experts selected by the unchanged top-k router are materialized. Each layer
has a bounded deterministic cache of exact packed rank-slice tensors; no dense
expert weight is ever reconstructed persistently. The arena storage mode uses
small, bounded blocks of exact typed slots shared by layers of the same rank
width. This avoids both one enormous CUDA/UVM allocation and thousands of
independent per-expert allocations.

The ``resident_fused`` mode uses the same exact typed arena representation but
fills all 256 experts of every active layer once during model post-processing.
It closes and advises away each bounded rank-pack mapping window immediately.
Inference then performs no expert loads, evictions, or checkpoint reads.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

import torch

from tp3k3.geometry import (
    EXPERTS,
    GEOMETRY_ID,
    HIDDEN,
    SEMANTIC_INTERMEDIATE,
    layer_geometry,
    rank_geometry,
)


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(?P<layer>[0-9]+)(?:\.|$)")
_STORES: list["LazyExpertStore"] = []
_SHARED_ARENAS: dict[tuple[str, int], "SharedLayerArena"] = {}
_SHARED_ARENAS_LOCK = threading.Lock()


def _enabled() -> bool:
    return os.environ.get("VLLM_GLM53_EXL3_LAZY_K3", "0") == "1"


def _cache_limit() -> int:
    value = int(os.environ.get("GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER", "8"))
    # The 16-slot full-model attempt retained more than 86 GiB of
    # MemAvailable on every Spark and measured about 5.4 GiB of exact K3
    # cache per rank. Permit a larger, still-bounded per-layer cache for the
    # measured serving envelope. Keep 256 reserved for the width-shared
    # implementation: a per-layer 256-slot cache would retain the complete
    # routed payload and violate the 12 GiB host reserve.
    maximum = (
        EXPERTS
        if _execution_mode() in {"shared_layer_fused", "resident_fused", "resident_uva"}
        else 192
    )
    if not 1 <= value <= maximum:
        raise ValueError(f"lazy K3 cache experts per layer must be 1..{maximum}")
    return value


def _cache_policy() -> str:
    value = os.environ.get("GLM53_LAZY_K3_CACHE_POLICY", "lfu").lower()
    if value not in {"lru", "lfu"}:
        raise ValueError("lazy K3 cache policy must be lru or lfu")
    return value


def _cache_storage() -> str:
    value = os.environ.get("GLM53_LAZY_K3_CACHE_STORAGE", "arena").lower()
    if value not in {"legacy", "arena"}:
        raise ValueError("lazy K3 cache storage must be legacy or arena")
    return value


def _arena_block_slots() -> int:
    value = int(os.environ.get("GLM53_LAZY_K3_ARENA_BLOCK_SLOTS", "16"))
    if not 1 <= value <= 64:
        raise ValueError("lazy K3 arena block slots must be 1..64")
    return value


def _execution_mode() -> str:
    value = os.environ.get("GLM53_LAZY_K3_EXECUTION", "python_loop").lower()
    if value not in {
        "python_loop",
        "chunked_fused",
        "shared_layer_fused",
        "resident_fused",
        "resident_uva",
    }:
        raise ValueError(
            "lazy K3 execution must be python_loop, chunked_fused, "
            "shared_layer_fused, resident_fused, or resident_uva"
        )
    return value


def _model_dir() -> Path:
    value = Path(os.environ.get("GLM53_LAZY_K3_MODEL_DIR", "/model"))
    if not value.is_dir():
        raise RuntimeError(f"lazy K3 model directory missing: {value}")
    return value


def _aligned_bytes(value: int, alignment: int = 256) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _allocate_typed_block(tensors, count: int, device: torch.device):
    """Allocate one aligned byte slab and expose exact typed tensor views."""

    layout = []
    offset = 0
    payload_bytes = 0
    for projection, projection_tensors in tensors.items():
        for suffix, source in projection_tensors.items():
            offset = _aligned_bytes(offset)
            elements = count * source.numel()
            size = elements * source.element_size()
            layout.append((projection, suffix, source, offset, size))
            offset += size
            payload_bytes += size
    allocation_bytes = _aligned_bytes(offset)
    host_raw = None
    if os.environ.get("GLM53_LAZY_K3_UVA", "0") == "1":
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        # GB10 accepts CUDA UVA views over ordinary anonymous CPU memory. Keep
        # this unpinned so the complete resident arena is accounted as one
        # host-RAM copy instead of competing with the checkpoint page cache.
        host_raw = torch.empty(
            allocation_bytes, dtype=torch.uint8, device="cpu", pin_memory=False
        )
        raw = get_accelerator_view_from_cpu_tensor(host_raw)
    else:
        raw = torch.empty(allocation_bytes, dtype=torch.uint8, device=device)
    views = {projection: {} for projection in tensors}
    for projection, suffix, source, start, size in layout:
        typed = raw.narrow(0, start, size).view(source.dtype).view(
            (count,) + tuple(source.shape)
        )
        views[projection][suffix] = typed
    return raw, views, payload_bytes, allocation_bytes, host_raw


class SharedLayerArena:
    """One exact reusable arena for all layers of one physical rank width."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.block_slots = _arena_block_slots()
        self.lock = threading.RLock()
        self.owner: tuple[int, int] | None = None
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.slots: dict[int, int] = {}
        self.free_slots = list(range(self.limit))
        # Blocks are allocated only when first touched. Each bounded block is
        # one byte slab with aligned typed views, avoiding both the enormous
        # [256, ...] mappings and thousands of independent suffix allocations
        # that produced NV_ERR_NO_MEMORY on GB10 with ample MemAvailable.
        self.arena: dict[str, dict[str, list[torch.Tensor | None]]] | None = None
        self.arena_backing: dict[
            str, dict[str, list[torch.Tensor | None]]
        ] | None = None
        self.arena_raw_backing: list[torch.Tensor | None] | None = None
        self.arena_bytes = 0
        self.slot_packs: dict[int, dict] = {}
        self.fused_state: SimpleNamespace | None = None
        self.fused_slots = 0
        self.owner_switches = 0


class LazyExpertStore:
    """List-like exact expert pack provider with a per-layer bounded cache."""

    def __init__(self, runtime, *, layer: int, rank: int, device: torch.device, limit: int):
        self.runtime = runtime
        self.layer = int(layer)
        self.rank = int(rank)
        self.device = torch.device(device)
        self.limit = int(limit)
        self.policy = _cache_policy()
        self.storage = _cache_storage()
        self.width = rank_geometry(layer, rank)[1]
        self.block_slots = _arena_block_slots()
        self.root = _model_dir()
        packed_value = os.environ.get("GLM53_LAZY_K3_PACKED_ROOT", "")
        self.packed_root = Path(packed_value) if packed_value else None
        if self.packed_root is not None:
            complete = self.packed_root / "COMPLETE.json"
            if not complete.is_file():
                raise RuntimeError(f"rank-local K3 pack receipt missing: {complete}")
            packed_receipt = json.loads(complete.read_text(encoding="utf-8"))
            if not (
                packed_receipt.get("passed") is True
                and packed_receipt.get("geometry_id") == GEOMETRY_ID
                and packed_receipt.get("rank") == self.rank
                and packed_receipt.get("layers") == 76
            ):
                raise RuntimeError("rank-local K3 pack identity differs")
        self.packed_handle = None
        self.packed_keys: set[str] | None = None
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.lock = threading.RLock()
        self.loads = 0
        self.hits = 0
        self.evictions = 0
        self.payload_bytes = 0
        self.io_seconds = 0.0
        self.cache_drop_calls = 0
        self.cache_drop_bytes = 0
        self.last_expert: int | None = None
        self.frequency = [0] * EXPERTS
        self.last_used = [0] * EXPERTS
        self.access_clock = 0
        self.arena: dict[str, dict[str, list[torch.Tensor | None]]] | None = None
        self.arena_backing: dict[
            str, dict[str, list[torch.Tensor | None]]
        ] | None = None
        self.arena_raw_backing: list[torch.Tensor | None] | None = None
        self.arena_host_backing: list[torch.Tensor | None] = []
        self.slots: dict[int, int] = {}
        self.free_slots = list(range(self.limit))
        self.arena_bytes = 0
        self.slot_packs: dict[int, dict] = {}
        self.fused_state: SimpleNamespace | None = None
        self.fused_slots = 0
        self.resident_ready = False
        self.resident_apply_calls = 0
        _STORES.append(self)

    def __len__(self) -> int:
        return EXPERTS

    def _path(self, expert: int) -> Path:
        return self.root / f"k3-layer-{self.layer:03d}-expert-{expert:03d}.safetensors"

    def _ensure_arena(
        self, tensors: dict[str, dict[str, torch.Tensor]], slot: int
    ) -> None:
        block = slot // self.block_slots
        base = block * self.block_slots
        count = min(self.block_slots, self.limit - base)
        if self.arena is None:
            self.arena = {
                projection: {
                    suffix: [None] * self.limit
                    for suffix in projection_tensors
                }
                for projection, projection_tensors in tensors.items()
            }
            blocks = (self.limit + self.block_slots - 1) // self.block_slots
            self.arena_backing = {
                projection: {
                    suffix: [None] * blocks for suffix in projection_tensors
                }
                for projection, projection_tensors in tensors.items()
            }
            self.arena_raw_backing = [None] * blocks
        assert self.arena_backing is not None
        assert self.arena_raw_backing is not None
        if self.arena_raw_backing[block] is not None:
            return
        raw, typed_views, payload_bytes, block_bytes, host_raw = _allocate_typed_block(
            tensors, count, self.device
        )
        self.arena_raw_backing[block] = raw
        self.arena_host_backing.append(host_raw)
        for projection, projection_tensors in typed_views.items():
            for suffix, backing in projection_tensors.items():
                self.arena_backing[projection][suffix][block] = backing
                for relative in range(count):
                    self.arena[projection][suffix][base + relative] = backing[relative]
        self.arena_bytes += block_bytes
        cuda_allocated = None
        cuda_reserved = None
        if self.device.type == "cuda":
            cuda_allocated = torch.cuda.memory_allocated(self.device)
            cuda_reserved = torch.cuda.memory_reserved(self.device)
        print(
            "GLM53_LAZY_K3_ARENA_BLOCK_READY "
            f"layer={self.layer} rank={self.rank} width={self.width} "
            f"block={block} slots={base}:{base + count} "
            f"block_bytes={block_bytes} payload_bytes={payload_bytes} "
            f"arena_bytes={self.arena_bytes} allocation_layout=aligned-byte-slab-v1 "
            f"cuda_allocated={cuda_allocated} cuda_reserved={cuda_reserved}",
            flush=True,
        )

    def _release_packed_handle(self) -> None:
        """Release the mmap and make its clean rank-pack pages reclaimable."""

        was_open = getattr(self, "packed_handle", None) is not None
        self.packed_handle = None
        self.packed_keys = None
        packed_root = getattr(self, "packed_root", None)
        if not was_open or packed_root is None:
            return
        path = packed_root / f"k3-layer-{self.layer:03d}-rank{self.rank}.safetensors"
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
        self.cache_drop_calls += 1
        self.cache_drop_bytes += path.stat().st_size

    def _load(self, expert: int, slot: int | None = None) -> dict:
        from safetensors import safe_open

        path = (
            self.packed_root / f"k3-layer-{self.layer:03d}-rank{self.rank}.safetensors"
            if self.packed_root is not None
            else self._path(expert)
        )
        if not path.is_file():
            raise RuntimeError(f"sealed lazy K3 expert file missing: {path}")
        started = time.monotonic()
        tensors: dict[str, dict[str, torch.Tensor]] = {}
        if self.packed_root is not None:
            if self.packed_handle is None:
                self.packed_handle = safe_open(path, framework="pt", device="cpu")
                self.packed_keys = set(self.packed_handle.keys())
            handle = self.packed_handle
            available = self.packed_keys
            assert available is not None
            for projection in ("gate_proj", "up_proj", "down_proj"):
                projection_tensors = {}
                for suffix in ("trellis", "suh", "svh", "mcg"):
                    key = (
                        f"model.layers.{self.layer}.mlp.experts.{expert}."
                        f"{projection}.rank{self.rank}.{suffix}"
                    )
                    if key not in available:
                        raise RuntimeError(f"lazy K3 tensor missing from {path.name}: {key}")
                    projection_tensors[suffix] = handle.get_tensor(key)
                tensors[projection] = projection_tensors
        else:
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    projection_tensors = {}
                    for suffix in ("trellis", "suh", "svh", "mcg"):
                        key = (
                            f"model.layers.{self.layer}.mlp.experts.{expert}."
                            f"{projection}.rank{self.rank}.{suffix}"
                        )
                        if key not in available:
                            raise RuntimeError(
                                f"lazy K3 tensor missing from {path.name}: {key}"
                            )
                        projection_tensors[suffix] = handle.get_tensor(key)
                    tensors[projection] = projection_tensors
        payload = sum(
            tensor.numel() * tensor.element_size()
            for projection in tensors.values()
            for tensor in projection.values()
        )
        if self.storage == "arena":
            if slot is None:
                raise RuntimeError("arena lazy K3 load is missing a slot")
            self._ensure_arena(tensors, slot)
            assert self.arena is not None
            for projection, projection_tensors in tensors.items():
                for suffix, source in projection_tensors.items():
                    destination = self.arena[projection][suffix][slot]
                    assert destination is not None
                    destination.copy_(source)
            tensors = {
                projection: {
                    suffix: target[slot]
                    for suffix, target in projection_tensors.items()
                }
                for projection, projection_tensors in self.arena.items()
            }
        else:
            tensors = {
                projection: {
                    suffix: source.to(self.device)
                    for suffix, source in projection_tensors.items()
                }
                for projection, projection_tensors in tensors.items()
            }
        if self.storage == "arena" and slot in self.slot_packs:
            # LinearEXL3 keeps views of the fixed arena tensors. Reuse those
            # lightweight handles after replacing the exact slot payload.
            pack = self.slot_packs[slot]
        else:
            gate = tensors["gate_proj"]
            up = tensors["up_proj"]
            down = tensors["down_proj"]
            pack = {
                "gate": self.runtime.make_linear_exl3(
                    gate["trellis"], gate["suh"], gate["svh"], gate["mcg"]
                ),
                "up": self.runtime.make_linear_exl3(
                    up["trellis"], up["suh"], up["svh"], up["mcg"]
                ),
                "down": self.runtime.make_linear_exl3(
                    down["trellis"], down["suh"], down["svh"], down["mcg"]
                ),
            }
            if self.storage == "arena":
                assert slot is not None
                self.slot_packs[slot] = pack
        self.loads += 1
        self.payload_bytes += int(payload)
        self.io_seconds += time.monotonic() - started
        self.last_expert = expert
        return pack

    def __getitem__(self, expert: int) -> dict:
        expert = int(expert)
        if not 0 <= expert < EXPERTS:
            raise IndexError(expert)
        with self.lock:
            self.access_clock += 1
            self.last_used[expert] = self.access_clock
            pack = self.cache.pop(expert, None)
            if pack is not None:
                self.hits += 1
                self.cache[expert] = pack
                return pack
            slot = None
            if getattr(self, "storage", "legacy") == "arena":
                if self.free_slots:
                    slot = self.free_slots.pop(0)
                else:
                    victim = self._select_victim()
                    self.cache.pop(victim)
                    slot = self.slots.pop(victim)
                    self.evictions += 1
            pack = self._load(expert, slot)
            self.cache[expert] = pack
            if slot is not None:
                self.slots[expert] = slot
            while len(self.cache) > self.limit:
                victim = self._select_victim()
                self.cache.pop(victim)
                self.evictions += 1
            return pack

    def _select_victim(self) -> int:
        if self.policy == "lfu":
            return min(
                self.cache,
                key=lambda item: (self.frequency[item], self.last_used[item], item),
            )
        return next(iter(self.cache))

    def observe_ids(self, ids: torch.Tensor) -> None:
        """Account routed expert presence before deterministic LFU eviction.

        Count each expert at most once per routing call.  vLLM profiles the
        model with a large synthetic batch during startup; raw token counts
        would otherwise give those dummy routes a permanent 1024x advantage
        over experts selected by later single-request batches.
        """

        # A fully resident store never evicts. Frequency bookkeeping would
        # only synchronize every layer's CUDA routing IDs to the host, and
        # its dynamic boolean indexing is illegal during CUDA graph capture.
        if self.policy != "lfu" or _execution_mode() in {"resident_fused", "resident_uva"}:
            return
        flat = ids.detach().reshape(-1).to(dtype=torch.long)
        flat = flat[(flat >= 0) & (flat < EXPERTS)]
        counts = torch.bincount(flat, minlength=EXPERTS).cpu().tolist()
        with self.lock:
            for expert, count in enumerate(counts[:EXPERTS]):
                self.frequency[expert] += int(count > 0)

    def prepare_chunk(self, experts: list[int]) -> None:
        """Make one fused chunk resident without evicting its own members."""

        if self.storage != "arena":
            raise RuntimeError("lazy K3 fused chunks require arena storage")
        protected = set(int(expert) for expert in experts)
        if len(protected) != len(experts) or len(protected) > self.limit:
            raise ValueError("lazy K3 fused chunk must be unique and cache-bounded")
        if any(expert < 0 or expert >= EXPERTS for expert in protected):
            raise IndexError("lazy K3 fused chunk expert is out of range")
        with self.lock:
            for expert in experts:
                self.access_clock += 1
                self.last_used[expert] = self.access_clock
                pack = self.cache.pop(expert, None)
                if pack is not None:
                    self.hits += 1
                    self.cache[expert] = pack
                    continue
                if self.free_slots:
                    slot = self.free_slots.pop(0)
                else:
                    candidates = [value for value in self.cache if value not in protected]
                    if not candidates:
                        raise RuntimeError("no non-chunk lazy K3 arena victim is available")
                    if self.policy == "lfu":
                        victim = min(
                            candidates,
                            key=lambda item: (
                                self.frequency[item], self.last_used[item], item
                            ),
                        )
                    else:
                        victim = candidates[0]
                    self.cache.pop(victim)
                    slot = self.slots.pop(victim)
                    self.evictions += 1
                self.cache[expert] = self._load(expert, slot)
                self.slots[expert] = slot
            missing = protected.difference(self.slots)
            if missing:
                raise RuntimeError(f"lazy K3 fused chunk lost arena slots: {sorted(missing)}")

    def fill_resident(self) -> None:
        """Load one durable exact copy of every expert into final device slots."""

        if _execution_mode() not in {"resident_fused", "resident_uva"}:
            raise RuntimeError(
                "full resident fill requires resident_fused or resident_uva execution"
            )
        if self.storage != "arena" or self.limit != EXPERTS:
            raise RuntimeError("full resident K3 requires a 256-slot exact arena")
        if self.resident_ready:
            raise RuntimeError("full resident K3 layer was initialized twice")
        for start in range(0, EXPERTS, self.block_slots):
            stop = min(start + self.block_slots, EXPERTS)
            self.prepare_chunk(list(range(start, stop)))
            # The source mapping must outlive the final device copies.  Closing
            # it after synchronization bounds file-backed residency to one
            # small loading window instead of an entire 1.1--1.3 GiB layer.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._release_packed_handle()
        expected = set(range(EXPERTS))
        if (
            set(self.cache) != expected
            or set(self.slots) != expected
            or any(self.slots[expert] != expert for expert in range(EXPERTS))
            or self.free_slots
            or self.loads != EXPERTS
            or self.evictions != 0
        ):
            raise RuntimeError("full resident K3 slot identity differs")
        self.get_fused_state()
        self.resident_ready = True
        available = None
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
        minimum = int(os.environ.get("GLM53_RESIDENT_MIN_AVAILABLE_BYTES", "0"))
        if minimum and (available is None or available < minimum):
            raise RuntimeError(
                "full resident K3 host reserve failed after bounded load: "
                f"layer={self.layer} available={available} minimum={minimum}"
            )
        cuda_allocated = None
        cuda_reserved = None
        if self.device.type == "cuda":
            cuda_allocated = int(torch.cuda.memory_allocated(self.device))
            cuda_reserved = int(torch.cuda.memory_reserved(self.device))
        print(
            "GLM53_K3_LAYER_FULLY_RESIDENT "
            f"layer={self.layer} rank={self.rank} width={self.width} "
            f"experts={len(self.cache)} loads={self.loads} evictions={self.evictions} "
            f"arena_bytes={self.arena_bytes} available_bytes={available} "
            f"cuda_allocated={cuda_allocated} cuda_reserved={cuda_reserved}",
            flush=True,
        )

    def get_fused_state(self) -> tuple[SimpleNamespace, list[dict]]:
        """Build stable pointer tables over the currently initialized slots."""

        if self.storage != "arena" or self.arena is None or not self.slot_packs:
            raise RuntimeError("chunked fused lazy K3 requires initialized arena slots")
        slot_count = len(self.slot_packs)
        if set(self.slot_packs) != set(range(slot_count)):
            raise RuntimeError("lazy K3 arena slots are not contiguous")
        packs = [self.slot_packs[index] for index in range(slot_count)]
        if self.fused_state is None or self.fused_slots != slot_count:
            if self.fused_state is not None:
                # Do not retire an old pointer table before its last kernel has
                # consumed it. This path occurs only while the arena grows.
                torch.cuda.synchronize(self.device)
            first_trellis = self.arena["gate_proj"]["trellis"][0]
            if first_trellis is None:
                raise RuntimeError("lazy K3 arena slot zero is not initialized")
            state = SimpleNamespace(
                w13_trellis=first_trellis,
                _exl3_hidden_size=HIDDEN,
                _exl3_intermediate_local=self.width,
                _exl3_bits=3,
                _exl3_shared_w13_suh=all(
                    torch.equal(pack["gate"].suh, pack["up"].suh) for pack in packs
                ),
            )
            self.runtime.build_exl3_fused_state(state, packs)
            self.fused_state = state
            self.fused_slots = slot_count
        return self.fused_state, packs

    @contextmanager
    def shared_layer_session(self):
        """Bind this layer to the process-wide arena for its rank width.

        Model execution is sequential across layers and the qualified endpoint
        is single-sequence.  Reusing one 640-wide and one 768-wide arena avoids
        retaining a separate allocation for every routed layer while allowing
        all experts selected by a large prefill to run in one fused call.
        """

        if self.storage != "arena":
            raise RuntimeError("shared-layer lazy K3 requires arena storage")
        key = (str(self.device), self.width)
        with _SHARED_ARENAS_LOCK:
            shared = _SHARED_ARENAS.get(key)
            if shared is None:
                shared = SharedLayerArena(self.limit)
                _SHARED_ARENAS[key] = shared
            elif shared.limit != self.limit:
                raise RuntimeError("shared lazy K3 arena limit differs")
        owner = (self.rank, self.layer)
        with shared.lock:
            if shared.owner is None and self.arena is not None:
                # Unit/runtime gates may populate this store through the
                # reference loop before exercising the fused path.
                shared.owner = owner
                shared.cache = self.cache
                shared.slots = self.slots
                shared.free_slots = self.free_slots
                shared.arena = self.arena
                shared.arena_backing = self.arena_backing
                shared.arena_raw_backing = self.arena_raw_backing
                shared.arena_bytes = self.arena_bytes
                shared.slot_packs = self.slot_packs
                shared.fused_state = self.fused_state
                shared.fused_slots = self.fused_slots
            elif shared.owner != owner:
                if shared.owner is not None:
                    shared.owner_switches += 1
                shared.owner = owner
                shared.cache.clear()
                shared.slots.clear()
                shared.free_slots.clear()
                shared.free_slots.extend(range(shared.limit))
            self.cache = shared.cache
            self.slots = shared.slots
            self.free_slots = shared.free_slots
            self.arena = shared.arena
            self.arena_backing = shared.arena_backing
            self.arena_raw_backing = shared.arena_raw_backing
            self.arena_bytes = shared.arena_bytes
            self.slot_packs = shared.slot_packs
            self.fused_state = shared.fused_state
            self.fused_slots = shared.fused_slots
            try:
                yield shared
            finally:
                shared.arena = self.arena
                shared.arena_backing = self.arena_backing
                shared.arena_raw_backing = self.arena_raw_backing
                shared.arena_bytes = self.arena_bytes
                shared.slot_packs = self.slot_packs
                shared.fused_state = self.fused_state
                shared.fused_slots = self.fused_slots
                self._release_packed_handle()

    def summary(self) -> dict:
        shared = _SHARED_ARENAS.get((str(self.device), self.width))
        return {
            "layer": self.layer,
            "rank": self.rank,
            "width": self.width,
            "cache_limit": self.limit,
            "cache_policy": self.policy,
            "cache_storage": self.storage,
            "arena_layout": "aligned-byte-slab-v3" if self.storage == "arena" else None,
            "arena_block_slots": self.block_slots if self.storage == "arena" else None,
            "execution_mode": _execution_mode(),
            "arena_bytes": self.arena_bytes,
            "resident_experts": list(self.cache),
            "loads": self.loads,
            "hits": self.hits,
            "evictions": self.evictions,
            "payload_bytes": self.payload_bytes,
            "io_seconds": self.io_seconds,
            "page_cache_drop_calls": self.cache_drop_calls,
            "page_cache_drop_bytes": self.cache_drop_bytes,
            "last_expert": self.last_expert,
            "rank_local_pack": self.packed_root is not None,
            "fully_resident": self.resident_ready,
            "resident_apply_calls": self.resident_apply_calls,
            "shared_arena": None
            if shared is None
            else {
                "width": self.width,
                "limit": shared.limit,
                "block_slots": shared.block_slots,
                "owner": list(shared.owner) if shared.owner is not None else None,
                "owner_switches": shared.owner_switches,
                "arena_bytes": shared.arena_bytes,
                "initialized_slots": len(shared.slot_packs),
            },
            "top_frequency": sorted(
                enumerate(self.frequency), key=lambda item: (-item[1], item[0])
            )[:16],
        }


def _resident_first(experts: list[int], resident) -> list[int]:
    """Consume resident payload before misses without changing set identity.

    Loading misses first can evict selected weights that have not yet
    contributed and force an avoidable reload later in the same layer. The
    original sorted order is retained within both deterministic partitions.
    """

    resident_set = set(resident)
    return [expert for expert in experts if expert in resident_set] + [
        expert for expert in experts if expert not in resident_set
    ]


def _apply_chunked_fused(runtime, layer, store, x2d, ids, weights, limit):
    """Evaluate all routed experts in exact arena-sized fused chunks."""

    if store.storage != "arena":
        raise RuntimeError("chunked fused lazy K3 requires arena cache storage")
    ids = ids.reshape(x2d.shape[0], -1).to(dtype=torch.long)
    weights = weights.reshape(x2d.shape[0], -1)
    unique = [int(value) for value in torch.unique(ids).tolist() if 0 <= int(value) < EXPERTS]
    # With a 192-slot arena and all 256 experts selected, resident-first order
    # reduces steady-state refills from 128 to the minimum 64 experts.
    unique = _resident_first(unique, store.cache)
    output = torch.zeros_like(x2d, dtype=torch.float32)
    try:
        for start in range(0, len(unique), store.limit):
            experts = unique[start : start + store.limit]
            store.prepare_chunk(experts)
            state, packs = store.get_fused_state()
            sentinel = len(packs)
            lookup = torch.full(
                (EXPERTS,), sentinel, dtype=torch.long, device=ids.device
            )
            for expert in experts:
                lookup[expert] = store.slots[expert]
            safe = ids.clamp(min=0, max=EXPERTS - 1)
            slot_ids = lookup[safe]
            slot_ids = torch.where(
                (ids >= 0) & (ids < EXPERTS),
                slot_ids,
                slot_ids.new_full((), sentinel),
            )
            output.add_(
                runtime.apply_exl3_fused_moe(
                    x2d, slot_ids, weights, state, packs, None, float(limit)
                )
            )
    finally:
        # Each layer retains its exact device cache, but not the safetensors
        # mapping used to refill it.  Keeping all 76 mappings alive caused
        # reclaimable rank-pack pages to accumulate across prompts and was the
        # measured trigger for sustained swap in the earlier segmented path.
        store._release_packed_handle()
    layer._exl3_last_apply = "lazy_nvme_chunked_fused"
    return output


def _apply_shared_layer_fused(runtime, layer, store, x2d, ids, weights, limit):
    """Evaluate a layer in one fused call using its width-shared exact arena."""

    ids = ids.reshape(x2d.shape[0], -1).to(dtype=torch.long)
    weights = weights.reshape(x2d.shape[0], -1)
    experts = [
        int(value)
        for value in torch.unique(ids).tolist()
        if 0 <= int(value) < EXPERTS
    ]
    with store.shared_layer_session():
        store.prepare_chunk(experts)
        state, packs = store.get_fused_state()
        sentinel = len(packs)
        lookup = torch.full((EXPERTS,), sentinel, dtype=torch.long, device=ids.device)
        for expert in experts:
            lookup[expert] = store.slots[expert]
        safe = ids.clamp(min=0, max=EXPERTS - 1)
        slot_ids = lookup[safe]
        slot_ids = torch.where(
            (ids >= 0) & (ids < EXPERTS), slot_ids, slot_ids.new_full((), sentinel)
        )
        output = runtime.apply_exl3_fused_moe(
            x2d, slot_ids, weights, state, packs, None, float(limit)
        )
    layer._exl3_last_apply = "lazy_nvme_shared_layer_fused"
    return output


def _apply_resident_fused(runtime, layer, store, x2d, ids, weights, limit):
    """Run the immutable all-expert arena without any load or eviction path."""

    if not store.resident_ready or len(store.cache) != EXPERTS:
        raise RuntimeError("full resident K3 layer is incomplete")
    loads_before = store.loads
    ids = ids.reshape(x2d.shape[0], -1).to(dtype=torch.long)
    weights = weights.reshape(x2d.shape[0], -1)
    sentinel = EXPERTS
    slot_ids = torch.where(
        (ids >= 0) & (ids < EXPERTS), ids, ids.new_full((), sentinel)
    )
    state, packs = store.get_fused_state()
    if len(packs) != EXPERTS:
        raise RuntimeError("full resident K3 fused pointer table is incomplete")
    output = runtime.apply_exl3_fused_moe(
        x2d, slot_ids, weights, state, packs, None, float(limit)
    )
    if store.loads != loads_before or store.evictions != 0:
        raise RuntimeError("full resident K3 inference touched the loading path")
    store.resident_apply_calls += 1
    layer._exl3_last_apply = "fully_resident_fused"
    return output


def _emit_stats() -> None:
    if not _STORES:
        return
    value = {
        "schema": "glm53-full-exl3-tp3.lazy-k3-stats.v1",
        "stores": len(_STORES),
        "loads": sum(store.loads for store in _STORES),
        "hits": sum(store.hits for store in _STORES),
        "evictions": sum(store.evictions for store in _STORES),
        "payload_bytes": sum(store.payload_bytes for store in _STORES),
        "io_seconds": sum(store.io_seconds for store in _STORES),
        "layers": [store.summary() for store in _STORES],
    }
    print("GLM53_LAZY_K3_STATS " + json.dumps(value, sort_keys=True), flush=True)


def resident_census() -> dict:
    """Return a deduplicated live-storage census for the resident design."""

    rows = [store.summary() for store in _STORES]
    storage: dict[int, int] = {}
    for store in _STORES:
        for raw in store.arena_raw_backing or []:
            if raw is None:
                continue
            value = raw.untyped_storage()
            storage[int(value.data_ptr())] = int(value.nbytes())
    value = {
        "schema": "glm53-full-exl3-tp3.resident-k3-census.v1",
        "stores": len(_STORES),
        "layers": sorted(store.layer for store in _STORES),
        "all_fully_resident": bool(_STORES)
        and all(store.resident_ready for store in _STORES),
        "experts_resident": sum(len(store.cache) for store in _STORES),
        "loads": sum(store.loads for store in _STORES),
        "evictions": sum(store.evictions for store in _STORES),
        "payload_bytes_read": sum(store.payload_bytes for store in _STORES),
        "arena_bytes_declared": sum(store.arena_bytes for store in _STORES),
        "unique_storage_allocations": len(storage),
        "unique_storage_bytes": sum(storage.values()),
        "open_rank_pack_handles": sum(
            store.packed_handle is not None for store in _STORES
        ),
        "resident_apply_calls": sum(
            store.resident_apply_calls for store in _STORES
        ),
        "per_layer": rows,
    }
    return value


def apply_patches() -> None:
    if not _enabled() or getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.model_executor.model_loader import default_loader
    from vllm.model_executor.models import deepseek_v2

    runtime = sys.modules.get("exl3")
    if runtime is None:
        from vllm.model_executor.layers.quantization import exl3 as package_exl3

        runtime = package_exl3
        sys.modules["exl3"] = runtime

    declared = os.environ.get("VLLM_GLM53_EXL3_TP3_GEOMETRY_ID", GEOMETRY_ID)
    if declared != GEOMETRY_ID:
        raise RuntimeError(f"checkpoint geometry {declared!r} != runtime {GEOMETRY_ID!r}")
    limit = _cache_limit()
    if _execution_mode() in {"shared_layer_fused", "resident_fused", "resident_uva"} and limit != EXPERTS:
        raise RuntimeError(f"{_execution_mode()} requires a 256-expert arena")

    original_create = runtime.Exl3MoEMethod.create_weights

    @wraps(original_create)
    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        del params_dtype, extra_weight_attrs
        if num_experts != EXPERTS or hidden_size != HIDDEN:
            raise RuntimeError("lazy K3 model expert geometry differs")
        match = _LAYER_PATTERN.search(str(getattr(layer, "layer_name", "")))
        if match is None:
            raise RuntimeError("cannot resolve lazy K3 layer during weight creation")
        layer_number = int(match.group("layer"))
        layer._exl3_layer = layer_number
        live_rank = get_tensor_model_parallel_rank()
        expected = rank_geometry(layer_number, live_rank)[1]
        if intermediate_size_per_partition != expected:
            raise RuntimeError(
                f"lazy K3 layer {layer_number} rank {live_rank} width "
                f"{intermediate_size_per_partition}/{expected}"
            )
        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_bits = self.bits
        layer._exl3_lazy_k3 = True

    runtime.Exl3MoEMethod.create_weights = create_weights

    original_factory = deepseek_v2.FusedMoEFactory

    @wraps(original_factory)
    def lazy_factory(*args, **kwargs):
        if get_tensor_model_parallel_world_size() != 3:
            raise RuntimeError("lazy K3 path requires tensor_parallel_size=3")
        prefix = str(kwargs.get("prefix", ""))
        match = _LAYER_PATTERN.search(prefix)
        if match is None:
            raise RuntimeError(f"cannot resolve lazy K3 layer from prefix {prefix!r}")
        layer = int(match.group("layer"))
        if int(kwargs.get("intermediate_size", -1)) != SEMANTIC_INTERMEDIATE:
            raise RuntimeError("GLM-5.3 routed expert semantic width differs")
        live_rank = get_tensor_model_parallel_rank()
        offset, width = rank_geometry(layer, live_rank)
        kwargs["intermediate_size"] = width * 3
        runner = original_factory(*args, **kwargs)
        routed = runner.routed_experts
        routed._exl3_geometry_id = GEOMETRY_ID
        routed._exl3_layer = layer
        routed._exl3_lazy_rank = live_rank
        routed._exl3_intermediate_offset = offset
        routed._exl3_expected_intermediate_local = width
        return runner

    deepseek_v2.FusedMoEFactory = lazy_factory

    original_process = runtime.Exl3MoEMethod.process_weights_after_loading

    @wraps(original_process)
    def process_weights(self, layer):
        if not getattr(layer, "_exl3_lazy_k3", False):
            return original_process(self, layer)
        live_rank = int(layer._exl3_lazy_rank)
        layer_number = int(layer._exl3_layer)
        offset, width = rank_geometry(layer_number, live_rank)
        if int(layer._exl3_intermediate_local) != width:
            raise RuntimeError("lazy K3 final rank width differs")
        if int(layer._exl3_intermediate_offset) != offset:
            raise RuntimeError("lazy K3 final rank offset differs")
        first = _model_dir() / f"k3-layer-{layer_number:03d}-expert-000.safetensors"
        last = _model_dir() / f"k3-layer-{layer_number:03d}-expert-255.safetensors"
        if not first.is_file() or not last.is_file():
            raise RuntimeError(f"lazy K3 layer {layer_number} endpoint files missing")
        layer._exl3_lazy_experts = LazyExpertStore(
            runtime,
            layer=layer_number,
            rank=live_rank,
            device=layer.moe_config.device,
            limit=limit,
        )
        if _execution_mode() in {"resident_fused", "resident_uva"}:
            layer._exl3_lazy_experts.fill_resident()
        print(
            "GLM53_EXL3_LAZY_K3_LAYER_READY "
            f"layer={layer_number} rank={live_rank} offset={offset} "
            f"width={width} cache_experts={limit}",
            flush=True,
        )

    runtime.Exl3MoEMethod.process_weights_after_loading = process_weights

    original_apply = runtime.Exl3MoEMethod.apply

    @wraps(original_apply)
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if not getattr(layer, "_exl3_lazy_k3", False):
            return original_apply(
                self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
            )
        del shared_experts, shared_experts_input
        store = getattr(layer, "_exl3_lazy_experts", None)
        if store is None:
            raise RuntimeError("lazy K3 expert store was not initialized")
        limit_value = getattr(self.moe, "swiglu_limit", None) or runtime.SWIGLU_LIMIT_DEFAULT
        x2d = x.reshape(-1, x.shape[-1])
        store.observe_ids(topk_ids)
        if _execution_mode() in {"resident_fused", "resident_uva"}:
            output = _apply_resident_fused(
                runtime, layer, store, x2d, topk_ids, topk_weights, limit_value
            )
        elif _execution_mode() == "chunked_fused":
            output = _apply_chunked_fused(
                runtime, layer, store, x2d, topk_ids, topk_weights, limit_value
            )
        elif _execution_mode() == "shared_layer_fused":
            output = _apply_shared_layer_fused(
                runtime, layer, store, x2d, topk_ids, topk_weights, limit_value
            )
        else:
            output = runtime.apply_exl3_python_loop(
                x2d,
                topk_ids.reshape(x2d.shape[0], -1),
                topk_weights.reshape(x2d.shape[0], -1),
                store,
                None,
                float(limit_value),
            )
            layer._exl3_last_apply = "lazy_nvme_python_loop"
        return output.reshape_as(x).to(dtype=x.dtype)

    runtime.Exl3MoEMethod.apply = apply

    # The stock iterator calls get_tensor before the model loader can reject a
    # name. Remove K3 files from the iterator's file list so no routed payload
    # is materialized during startup; LazyExpertStore opens exact keys later.
    original_prepare = default_loader.DefaultModelLoader._prepare_weights

    @wraps(original_prepare)
    def prepare_weights(self, *args, **kwargs):
        folder, files, use_safetensors = original_prepare(self, *args, **kwargs)
        if not use_safetensors:
            raise RuntimeError("lazy K3 requires safetensors")
        retained = [path for path in files if Path(path).name.startswith("bf16-passthrough-")]
        omitted = len(files) - len(retained)
        if not retained or omitted <= 0:
            raise RuntimeError("lazy K3 pre-materialization file filter did not match checkpoint")
        print(
            "GLM53_LAZY_K3_PREMATERIALIZATION_FILTER_OK "
            f"retained_bf16_files={len(retained)} omitted_k3_files={omitted}",
            flush=True,
        )
        return folder, retained, use_safetensors

    default_loader.DefaultModelLoader._prepare_weights = prepare_weights
    atexit.register(_emit_stats)
    print(
        "GLM53_EXL3_LAZY_K3_PATCH_INSTALLED "
        f"geometry={GEOMETRY_ID} cache_experts_per_layer={limit} "
        f"cache_policy={_cache_policy()} cache_storage={_cache_storage()} "
        f"execution={_execution_mode()}",
        flush=True,
    )


__all__ = [
    "LazyExpertStore",
    "SharedLayerArena",
    "_cache_limit",
    "_cache_policy",
    "_cache_storage",
    "_arena_block_slots",
    "_execution_mode",
    "_apply_resident_fused",
    "resident_census",
    "apply_patches",
]
