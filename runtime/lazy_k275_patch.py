"""Mixed-bit resident EXL3 patch for the GLM-5.3 2.75bpw derivative.

The sealed K275 checkpoint keeps the same 256 global experts and router as the
K3 checkpoint, but 64 experts per layer use K2 payloads and 192 use K3.  The
existing lazy K3 patch already has the safe model-loader and host-reserve
guards.  This module reuses those guards and replaces only the resident fill
and apply operations: K2 and K3 experts are stored in separate exact arenas,
then evaluated with one fused call per bit width.
"""

from __future__ import annotations

from types import SimpleNamespace
import json
import os
from pathlib import Path

import torch

import lazy_k3_patch as base


EXPERTS = base.EXPERTS
HIDDEN = base.HIDDEN


def _read_mem_available() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _drop_file_cache(path: Path) -> None:
    """Release clean checkpoint pages after copying them into the arena."""

    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            advise(descriptor, 0, 0, dontneed)
        finally:
            os.close(descriptor)
    except OSError:
        # Cache eviction is an optimization; correctness must not depend on it.
        pass


class MixedLazyExpertStore(base.LazyExpertStore):
    """Resident store that keeps K2 and K3 payloads in separate arenas."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        bits_path = self.root / "expert-bits.json"
        if not bits_path.is_file():
            raise RuntimeError(f"mixed expert-bit map missing: {bits_path}")
        document = json.loads(bits_path.read_text(encoding="utf-8"))
        if document.get("schema") != "glm53-full-exl3-tp3.k275-expert-bits.v1":
            raise RuntimeError("mixed expert-bit map schema differs")
        row = document.get("bits", {}).get(str(self.layer))
        if not isinstance(row, dict):
            raise RuntimeError(f"mixed expert-bit map missing layer {self.layer}")
        k2 = sorted({int(value) for value in row.get("k2", [])})
        k3 = sorted({int(value) for value in row.get("k3", [])})
        if len(k2) != 64 or len(k3) != 192 or set(k2) | set(k3) != set(range(EXPERTS)):
            raise RuntimeError(f"mixed expert-bit partition differs for layer {self.layer}")
        if set(k2) & set(k3):
            raise RuntimeError(f"mixed expert-bit partition overlaps for layer {self.layer}")
        self.mixed_model = True
        self.mixed_experts = {2: k2, 3: k3}
        self.mixed_groups: dict[int, dict] = {}
        self.mixed_resident = False
        self.mixed_raw_backing: list[torch.Tensor | None] = []
        # The shared census in lazy_k3_patch reports resident backing through
        # this field.  Mixed groups own their two independent slabs.
        self.arena_raw_backing = []

    def _read_tensors(self, expert: int) -> dict[str, dict[str, torch.Tensor]]:
        from safetensors import safe_open

        tensors: dict[str, dict[str, torch.Tensor]] = {}
        if self.packed_root is not None:
            path = self.packed_root / f"k3-layer-{self.layer:03d}-rank{self.rank}.safetensors"
            if not path.is_file():
                raise RuntimeError(f"mixed rank pack missing: {path}")
            if self.packed_handle is None:
                self.packed_handle = safe_open(path, framework="pt", device="cpu")
                self.packed_keys = set(self.packed_handle.keys())
            handle = self.packed_handle
            available = self.packed_keys
            assert available is not None
            self._read_tensors_from_handle(tensors, handle, available, expert, path)
        else:
            path = self._path(expert)
            if not path.is_file():
                raise RuntimeError(f"mixed expert file missing: {path}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                self._read_tensors_from_handle(tensors, handle, set(handle.keys()), expert, path)
        return tensors

    def _read_tensors_from_handle(self, tensors, handle, available, expert, path) -> None:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            projection_tensors: dict[str, torch.Tensor] = {}
            for suffix in ("trellis", "suh", "svh", "mcg"):
                key = (
                    f"model.layers.{self.layer}.mlp.experts.{expert}."
                    f"{projection}.rank{self.rank}.{suffix}"
                )
                if key not in available:
                    raise RuntimeError(f"mixed tensor missing: {path.name}:{key}")
                projection_tensors[suffix] = handle.get_tensor(key)
            tensors[projection] = projection_tensors

    def _make_state(self, bits: int, packs: list[dict]) -> SimpleNamespace:
        first = packs[0]["gate"]
        state = SimpleNamespace(
            w13_trellis=first.trellis,
            _exl3_hidden_size=HIDDEN,
            _exl3_intermediate_local=self.width,
            _exl3_bits=bits,
            _exl3_shared_w13_suh=all(
                torch.equal(pack["gate"].suh, pack["up"].suh) for pack in packs
            ),
        )
        self.runtime.build_exl3_fused_state(state, packs)
        return state

    def _make_pack(self, tensors: dict[str, dict[str, torch.Tensor]], slot: int,
                   views: dict[str, dict[str, torch.Tensor]]) -> dict:
        for projection, projection_tensors in tensors.items():
            for suffix, source in projection_tensors.items():
                views[projection][suffix][slot].copy_(source)
        return {
            "gate": self.runtime.make_linear_exl3(
                views["gate_proj"]["trellis"][slot],
                views["gate_proj"]["suh"][slot],
                views["gate_proj"]["svh"][slot],
                views["gate_proj"]["mcg"][slot],
            ),
            "up": self.runtime.make_linear_exl3(
                views["up_proj"]["trellis"][slot],
                views["up_proj"]["suh"][slot],
                views["up_proj"]["svh"][slot],
                views["up_proj"]["mcg"][slot],
            ),
            "down": self.runtime.make_linear_exl3(
                views["down_proj"]["trellis"][slot],
                views["down_proj"]["suh"][slot],
                views["down_proj"]["svh"][slot],
                views["down_proj"]["mcg"][slot],
            ),
        }

    def fill_resident(self) -> None:
        if not self.mixed_model:
            return super().fill_resident()
        if self.resident_ready:
            raise RuntimeError("mixed resident layer was initialized twice")
        if self.storage != "arena":
            raise RuntimeError("mixed resident fill requires arena storage")
        if not getattr(base, "_k275_bf16_cache_dropped", False):
            for path in self.root.glob("bf16-passthrough-*"):
                _drop_file_cache(path)
            base._k275_bf16_cache_dropped = True
        for bits in (2, 3):
            experts = self.mixed_experts[bits]
            first_tensors = self._read_tensors(experts[0])
            raw, views, payload_bytes, block_bytes, host_raw = base._allocate_typed_block(
                first_tensors, len(experts), self.device
            )
            self.mixed_raw_backing.append(raw)
            self.arena_raw_backing.append(raw)
            self.arena_host_backing.append(host_raw)
            self.arena_bytes += block_bytes
            packs: list[dict] = []
            global_to_slot: dict[int, int] = {}
            for slot, expert in enumerate(experts):
                tensors = first_tensors if slot == 0 else self._read_tensors(expert)
                pack = self._make_pack(tensors, slot, views)
                if self.packed_root is None:
                    _drop_file_cache(self._path(expert))
                packs.append(pack)
                global_to_slot[expert] = slot
                self.cache[expert] = pack
                self.loads += 1
                self.payload_bytes += sum(
                    tensor.numel() * tensor.element_size()
                    for projection in tensors.values()
                    for tensor in projection.values()
                )
            state = self._make_state(bits, packs)
            lookup = torch.full(
                (EXPERTS,), len(packs), dtype=torch.long, device=self.device
            )
            for expert, slot in global_to_slot.items():
                lookup[expert] = slot
            self.mixed_groups[bits] = {
                "experts": experts,
                "packs": packs,
                "state": state,
                "lookup": lookup,
                "bits": bits,
            }
            print(
                "GLM53_K275_MIXED_GROUP_RESIDENT "
                f"layer={self.layer} rank={self.rank} bits={bits} "
                f"experts={len(experts)} block_bytes={block_bytes} "
                f"payload_bytes={payload_bytes}",
                flush=True,
            )
        self._release_packed_handle()
        if (
            set(self.cache) != set(range(EXPERTS))
            or self.loads != EXPERTS
            or self.evictions != 0
            or set(self.mixed_groups) != {2, 3}
        ):
            raise RuntimeError("mixed resident expert census differs")
        self.mixed_resident = True
        self.resident_ready = True
        available = _read_mem_available()
        minimum = int(__import__("os").environ.get("GLM53_RESIDENT_MIN_AVAILABLE_BYTES", "0"))
        if minimum and (available is None or available < minimum):
            raise RuntimeError(
                "mixed resident host reserve failed: "
                f"available={available} minimum={minimum}"
            )
        cuda_allocated = None
        cuda_reserved = None
        if self.device.type == "cuda":
            cuda_allocated = int(torch.cuda.memory_allocated(self.device))
            cuda_reserved = int(torch.cuda.memory_reserved(self.device))
        print(
            "GLM53_K275_MIXED_LAYER_FULLY_RESIDENT "
            f"layer={self.layer} rank={self.rank} width={self.width} "
            f"experts={len(self.cache)} loads={self.loads} evictions={self.evictions} "
            f"arena_bytes={self.arena_bytes} available_bytes={available} "
            f"cuda_allocated={cuda_allocated} cuda_reserved={cuda_reserved}",
            flush=True,
        )


def _apply_mixed_resident(runtime, layer, store, x2d, ids, weights, limit):
    if not store.mixed_resident:
        raise RuntimeError("mixed resident store is incomplete")
    ids = ids.reshape(x2d.shape[0], -1).to(dtype=torch.long)
    weights = weights.reshape(x2d.shape[0], -1)
    output = torch.zeros_like(x2d, dtype=torch.float32)
    for bits in (2, 3):
        group = store.mixed_groups[bits]
        lookup = group["lookup"]
        safe = ids.clamp(min=0, max=EXPERTS - 1)
        slot_ids = lookup[safe]
        slot_ids = torch.where(
            (ids >= 0) & (ids < EXPERTS),
            slot_ids,
            slot_ids.new_full((), len(group["packs"])),
        )
        output.add_(
            runtime.apply_exl3_fused_moe(
                x2d,
                slot_ids,
                weights,
                group["state"],
                group["packs"],
                None,
                float(limit),
            )
        )
    if store.loads != EXPERTS or store.evictions != 0:
        raise RuntimeError("mixed resident inference touched loading path")
    store.resident_apply_calls += 1
    layer._exl3_last_apply = "mixed_resident_fused"
    return output


def apply_patches() -> None:
    if getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True
    original_resident = base._apply_resident_fused
    base.LazyExpertStore = MixedLazyExpertStore

    def resident(runtime, layer, store, x2d, ids, weights, limit):
        if getattr(store, "mixed_model", False):
            return _apply_mixed_resident(runtime, layer, store, x2d, ids, weights, limit)
        return original_resident(runtime, layer, store, x2d, ids, weights, limit)

    base._apply_resident_fused = resident
    base.apply_patches()
    print("GLM53_EXL3_LAZY_K275_MIXED_PATCH_INSTALLED", flush=True)


resident_census = base.resident_census


__all__ = ["MixedLazyExpertStore", "apply_patches", "resident_census"]
