from __future__ import annotations

from collections import OrderedDict
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from runtime.lazy_k3_patch import (
    LazyExpertStore,
    _SHARED_ARENAS,
    _apply_chunked_fused,
    _apply_resident_fused,
    _arena_block_slots,
    _cache_limit,
    _cache_policy,
    _cache_storage,
    _execution_mode,
    _resident_first,
)


class _FakeStore(LazyExpertStore):
    def __init__(self, limit: int, policy: str = "lru"):
        self.limit = limit
        self.policy = policy
        self.storage = "legacy"
        self.cache = OrderedDict()
        self.lock = __import__("threading").RLock()
        self.loads = 0
        self.hits = 0
        self.evictions = 0
        self.frequency = [0] * 256
        self.last_used = [0] * 256
        self.access_clock = 0
        self.slots = {}
        self.free_slots = list(range(limit))
        self.layer = 3
        self.rank = 0
        self.device = torch.device("cpu")
        self.width = 640
        self.block_slots = 16
        self.arena = None
        self.arena_backing = None
        self.arena_raw_backing = None
        self.arena_bytes = 0
        self.slot_packs = {}
        self.fused_state = None
        self.fused_slots = 0
        self.resident_ready = False
        self.resident_apply_calls = 0

    def _load(self, expert: int, slot=None):
        self.loads += 1
        return {"expert": expert}


def test_lazy_store_lru_is_bounded_and_deterministic() -> None:
    store = _FakeStore(limit=2)
    assert store[3] == {"expert": 3}
    assert store[4] == {"expert": 4}
    assert store[3] == {"expert": 3}
    assert store.hits == 1
    assert store[5] == {"expert": 5}
    assert list(store.cache) == [3, 5]
    assert store.loads == 3
    assert store.evictions == 1


def test_lazy_store_rejects_invalid_expert() -> None:
    store = _FakeStore(limit=1)
    for expert in (-1, 256):
        try:
            store[expert]
        except IndexError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid expert accepted")


def test_runtime_cache_limit_is_capacity_bounded(monkeypatch) -> None:
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER", "192")
    assert _cache_limit() == 192
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER", "193")
    with pytest.raises(ValueError, match="1..192"):
        _cache_limit()
    monkeypatch.setenv("GLM53_LAZY_K3_EXECUTION", "shared_layer_fused")
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_EXPERTS_PER_LAYER", "256")
    assert _cache_limit() == 256
    monkeypatch.setenv("GLM53_LAZY_K3_EXECUTION", "resident_fused")
    assert _cache_limit() == 256


def test_lfu_retains_frequent_expert_with_deterministic_ties() -> None:
    store = _FakeStore(limit=2, policy="lfu")
    store.frequency[3] = 10
    store.frequency[4] = 1
    store.frequency[5] = 1
    assert store[3] == {"expert": 3}
    assert store[4] == {"expert": 4}
    assert store[5] == {"expert": 5}
    assert list(store.cache) == [3, 5]
    assert store.evictions == 1


def test_lfu_counts_presence_once_per_routing_call() -> None:
    store = _FakeStore(limit=2, policy="lfu")
    store.observe_ids(torch.tensor([[3, 3, 3, 4], [3, 4, 4, 4]]))
    assert store.frequency[3] == 1
    assert store.frequency[4] == 1
    store.observe_ids(torch.tensor([4, 4, 5, -1, 256]))
    assert store.frequency[3] == 1
    assert store.frequency[4] == 2
    assert store.frequency[5] == 1


def test_chunk_order_consumes_resident_experts_before_misses() -> None:
    experts = list(range(8))
    resident = OrderedDict((expert, object()) for expert in (6, 2, 7))
    assert _resident_first(experts, resident) == [2, 6, 7, 0, 1, 3, 4, 5]


def test_fused_chunk_members_are_pinned_during_arena_fill() -> None:
    store = _FakeStore(limit=2, policy="lfu")
    store.storage = "arena"
    assert store[0] == {"expert": 0}
    assert store[1] == {"expert": 1}
    store.frequency[0] = 100
    store.frequency[1] = 100
    store.prepare_chunk([2, 3])
    assert set(store.cache) == {2, 3}
    assert set(store.slots) == {2, 3}
    assert store.evictions == 2


def test_shared_layer_arena_handoff_clears_identity_not_storage() -> None:
    _SHARED_ARENAS.clear()
    first = _FakeStore(limit=2, policy="lfu")
    first.storage = "arena"
    with first.shared_layer_session() as shared:
        first.prepare_chunk([0, 1])
        marker = object()
        first.arena = marker
        backing_marker = object()
        first.arena_backing = backing_marker
        raw_backing_marker = object()
        first.arena_raw_backing = raw_backing_marker
    assert shared.owner == (0, 3)
    assert set(shared.slots) == {0, 1}

    second = _FakeStore(limit=2, policy="lfu")
    second.storage = "arena"
    second.layer = 6
    with second.shared_layer_session() as reused:
        assert second.arena is marker
        assert second.arena_backing is backing_marker
        assert second.arena_raw_backing is raw_backing_marker
        assert not second.cache
        assert not second.slots
        second.prepare_chunk([7])
    assert reused is shared
    assert shared.owner == (0, 6)
    assert shared.owner_switches == 1
    assert shared.slots == {7: 0}
    _SHARED_ARENAS.clear()


def test_rank_local_layer_pack_reuses_one_verified_handle(
    monkeypatch, tmp_path
) -> None:
    model = tmp_path / "model"
    packed = tmp_path / "packed"
    model.mkdir()
    packed.mkdir()
    monkeypatch.setenv("GLM53_LAZY_K3_MODEL_DIR", str(model))
    monkeypatch.setenv("GLM53_LAZY_K3_PACKED_ROOT", str(packed))
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_STORAGE", "legacy")
    (packed / "COMPLETE.json").write_text(
        json.dumps(
            {
                "passed": True,
                "geometry_id": "rotating-uneven-768-640-640-v1",
                "rank": 0,
                "layers": 76,
            }
        )
    )
    tensors = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("trellis", "suh", "svh", "mcg"):
            key = f"model.layers.3.mlp.experts.0.{projection}.rank0.{suffix}"
            tensors[key] = torch.arange(4, dtype=torch.int32)
    save_file(tensors, packed / "k3-layer-003-rank0.safetensors")

    class Runtime:
        @staticmethod
        def make_linear_exl3(trellis, suh, svh, mcg):
            return SimpleNamespace(trellis=trellis, suh=suh, svh=svh, mcg=mcg)

    store = LazyExpertStore(
        Runtime(), layer=3, rank=0, device=torch.device("cpu"), limit=2
    )
    first = store._load(0)
    handle = store.packed_handle
    second = store._load(0)
    assert handle is not None and store.packed_handle is handle
    assert torch.equal(first["gate"].trellis, second["gate"].trellis)
    assert store.payload_bytes > 0
    pack_bytes = (packed / "k3-layer-003-rank0.safetensors").stat().st_size
    store._release_packed_handle()
    assert store.packed_handle is None and store.packed_keys is None
    assert store.cache_drop_calls == 1
    assert store.cache_drop_bytes == pack_bytes


def test_rank_local_arena_allocates_exact_blocked_slots_lazily(
    monkeypatch, tmp_path
) -> None:
    model = tmp_path / "model"
    packed = tmp_path / "packed"
    model.mkdir()
    packed.mkdir()
    monkeypatch.setenv("GLM53_LAZY_K3_MODEL_DIR", str(model))
    monkeypatch.setenv("GLM53_LAZY_K3_PACKED_ROOT", str(packed))
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_STORAGE", "arena")
    monkeypatch.setenv("GLM53_LAZY_K3_ARENA_BLOCK_SLOTS", "2")
    (packed / "COMPLETE.json").write_text(
        json.dumps(
            {
                "passed": True,
                "geometry_id": "rotating-uneven-768-640-640-v1",
                "rank": 0,
                "layers": 76,
            }
        )
    )
    tensors = {}
    for expert in (0, 1):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("trellis", "suh", "svh", "mcg"):
                key = f"model.layers.3.mlp.experts.{expert}.{projection}.rank0.{suffix}"
                tensors[key] = torch.full((4,), expert + 1, dtype=torch.int32)
    save_file(tensors, packed / "k3-layer-003-rank0.safetensors")

    class Runtime:
        @staticmethod
        def make_linear_exl3(trellis, suh, svh, mcg):
            return SimpleNamespace(trellis=trellis, suh=suh, svh=svh, mcg=mcg)

    store = LazyExpertStore(
        Runtime(), layer=3, rank=0, device=torch.device("cpu"), limit=2
    )
    first = store._load(0, slot=0)
    one_block_bytes = store.arena_bytes
    assert one_block_bytes >= 2 * 3 * 4 * 4 * 4
    assert store.arena is not None
    assert store.arena_backing is not None
    assert store.arena_raw_backing is not None
    assert len(store.arena_raw_backing) == 1
    assert store.arena_raw_backing[0] is not None
    storage_pointer = store.arena_raw_backing[0].untyped_storage().data_ptr()
    assert all(
        backing[0].untyped_storage().data_ptr() == storage_pointer
        for projection in store.arena_backing.values()
        for backing in projection.values()
    )
    assert store.arena["gate_proj"]["trellis"][1] is not None
    second = store._load(1, slot=1)
    assert store.arena_bytes == one_block_bytes
    assert first["gate"].trellis.data_ptr() != second["gate"].trellis.data_ptr()
    assert torch.equal(second["gate"].trellis, torch.full((4,), 2, dtype=torch.int32))


def test_shared_session_releases_rank_pack_mapping() -> None:
    _SHARED_ARENAS.clear()
    store = _FakeStore(limit=2)
    store.storage = "arena"
    store.packed_handle = object()
    store.packed_keys = {"key"}
    with store.shared_layer_session():
        pass
    assert store.packed_handle is None
    assert store.packed_keys is None
    _SHARED_ARENAS.clear()


def test_chunked_fused_releases_rank_pack_mapping() -> None:
    class Store:
        storage = "arena"
        limit = 1
        cache = {7: object()}
        slots = {7: 0}
        packed_handle = object()
        packed_keys = {"key"}

        def prepare_chunk(self, experts):
            assert experts == [7]

        def get_fused_state(self):
            return SimpleNamespace(), [{}]

        def _release_packed_handle(self):
            self.packed_handle = None
            self.packed_keys = None

    class Runtime:
        @staticmethod
        def apply_exl3_fused_moe(x, ids, weights, state, packs, shared, limit):
            assert ids.tolist() == [[0]]
            assert limit == 1.0
            return torch.zeros_like(x, dtype=torch.float32)

    store = Store()
    layer = SimpleNamespace()
    output = _apply_chunked_fused(
        Runtime(),
        layer,
        store,
        torch.ones((1, 4)),
        torch.tensor([[7]]),
        torch.ones((1, 1)),
        1.0,
    )
    assert torch.equal(output, torch.zeros((1, 4)))
    assert layer._exl3_last_apply == "lazy_nvme_chunked_fused"
    assert store.packed_handle is None
    assert store.packed_keys is None


def test_resident_fill_loads_every_slot_once_and_releases_each_block(monkeypatch) -> None:
    monkeypatch.setenv("GLM53_LAZY_K3_EXECUTION", "resident_fused")
    store = _FakeStore(limit=256)
    store.storage = "arena"
    store.block_slots = 64
    releases = []

    def prepare(experts):
        for expert in experts:
            slot = store.free_slots.pop(0)
            store.cache[expert] = {"expert": expert}
            store.slots[expert] = slot
            store.slot_packs[slot] = {"expert": expert}
            store.loads += 1

    store.prepare_chunk = prepare
    store._release_packed_handle = lambda: releases.append(store.loads)
    store.get_fused_state = lambda: (SimpleNamespace(), list(store.slot_packs.values()))
    store.fill_resident()
    assert store.resident_ready is True
    assert store.loads == 256 and store.evictions == 0
    assert store.slots == {expert: expert for expert in range(256)}
    assert releases == [64, 128, 192, 256]


def test_resident_fused_has_identity_slots_and_never_loads() -> None:
    class Store:
        resident_ready = True
        cache = {expert: object() for expert in range(256)}
        loads = 256
        evictions = 0
        resident_apply_calls = 0

        @staticmethod
        def get_fused_state():
            return SimpleNamespace(), [{} for _ in range(256)]

    class Runtime:
        @staticmethod
        def apply_exl3_fused_moe(x, ids, weights, state, packs, shared, limit):
            assert ids.tolist() == [[7, 255, 256]]
            assert len(packs) == 256 and shared is None and limit == 1.0
            return torch.ones_like(x)

    store = Store()
    layer = SimpleNamespace()
    output = _apply_resident_fused(
        Runtime(),
        layer,
        store,
        torch.zeros((1, 4)),
        torch.tensor([[7, 255, -1]]),
        torch.ones((1, 3)),
        1.0,
    )
    assert torch.equal(output, torch.ones((1, 4)))
    assert store.loads == 256 and store.evictions == 0
    assert store.resident_apply_calls == 1
    assert layer._exl3_last_apply == "fully_resident_fused"


def test_arena_block_slot_validation(monkeypatch) -> None:
    monkeypatch.setenv("GLM53_LAZY_K3_ARENA_BLOCK_SLOTS", "16")
    assert _arena_block_slots() == 16
    monkeypatch.setenv("GLM53_LAZY_K3_ARENA_BLOCK_SLOTS", "0")
    with pytest.raises(ValueError, match="1..64"):
        _arena_block_slots()


def test_cache_policy_validation(monkeypatch) -> None:
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_POLICY", "lfu")
    assert _cache_policy() == "lfu"
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_POLICY", "invalid")
    with pytest.raises(ValueError, match="lru or lfu"):
        _cache_policy()


def test_cache_storage_validation(monkeypatch) -> None:
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_STORAGE", "arena")
    assert _cache_storage() == "arena"
    monkeypatch.setenv("GLM53_LAZY_K3_CACHE_STORAGE", "invalid")
    with pytest.raises(ValueError, match="legacy or arena"):
        _cache_storage()


def test_execution_mode_validation(monkeypatch) -> None:
    monkeypatch.setenv("GLM53_LAZY_K3_EXECUTION", "chunked_fused")
    assert _execution_mode() == "chunked_fused"
    monkeypatch.setenv("GLM53_LAZY_K3_EXECUTION", "shared_layer_fused")
    assert _execution_mode() == "shared_layer_fused"
    monkeypatch.setenv("GLM53_LAZY_K3_EXECUTION", "resident_fused")
    assert _execution_mode() == "resident_fused"
    monkeypatch.setenv("GLM53_LAZY_K3_EXECUTION", "invalid")
    with pytest.raises(ValueError, match="resident_fused"):
        _execution_mode()
