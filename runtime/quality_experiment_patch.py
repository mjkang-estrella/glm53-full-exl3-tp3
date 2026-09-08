"""Opt-in same-bitrate scale experiments; runs inside the Spark candidate.

Only /capture/QUALITY_CONTROL.json enables this patch. The Zima controller
changes it between requests on all ranks, retaining all weights and routes.
Gains scale the output of a routed expert group, not the final logits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import faulthandler
import signal
from types import SimpleNamespace

import torch

ROOT = Path("/capture")
CONTROL = ROOT / "QUALITY_CONTROL.json"
_control = None
_mtime = None
_capture_rows = {}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.chmod(temp, 0o644)
    os.replace(temp, path)


def validate_control(value):
    if value.get("schema") != "glm53.quality-control.v1":
        raise ValueError("quality control schema differs")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", value.get("id", "")):
        raise ValueError("invalid quality experiment ID")
    gains = value.get("gains", {})
    for bits in ("2", "3"):
        gain = float(gains.get(bits, 1.0))
        if not math.isfinite(gain) or not 0.9 <= gain <= 1.1:
            raise ValueError("quality gain exceeds bounded 0.9..1.1 range")
    expert_gains = value.get("expert_gains", {})
    if len(expert_gains) > 6:
        raise ValueError("expert refit exceeds six layers")
    for layer, experts in expert_gains.items():
        if not 3 <= int(layer) <= 77 or len(experts) > 256:
            raise ValueError("invalid expert refit layer")
        for expert, gain in experts.items():
            if not 0 <= int(expert) < 256 or not .9 <= float(gain) <= 1.1:
                raise ValueError("expert scale outside bounded range")
    capture = value.get("capture")
    limit = value.get("swiglu_limit", 10.0)
    if limit != "unclamped" and (not math.isfinite(float(limit)) or not 1 <= float(limit) <= 1000):
        raise ValueError("invalid experiment SwiGLU limit")
    if capture:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", capture.get("id", "")):
            raise ValueError("invalid activation capture ID")
        if not 1 <= int(capture.get("rows", 0)) <= 8192:
            raise ValueError("activation row bound exceeds 8192")
        if not capture.get("layers") or len(capture["layers"]) > 6:
            raise ValueError("capture requires 1..6 layers")
        if any(not 3 <= int(layer) <= 77 for layer in capture["layers"]):
            raise ValueError("capture layer outside routed backbone")
    return value


def refresh(store):
    global _control, _mtime
    if store.layer != 3 and _control is not None:
        return _control
    stamp = CONTROL.stat().st_mtime_ns
    if stamp != _mtime:
        raw = CONTROL.read_bytes()
        _control = validate_control(json.loads(raw))
        _mtime = stamp
        atomic_json(ROOT / f"QUALITY_ACK-rank{store.rank}.json", {
            "schema": "glm53.quality-ack.v1", "id": _control["id"],
            "rank": store.rank, "control_sha256": hashlib.sha256(raw).hexdigest(),
            "gains": _control.get("gains", {}),
            "swiglu_limit": _control.get("swiglu_limit", 10.0),
        })
        print(f"GLM53_QUALITY_CONTROL rank={store.rank} id={_control['id']} "
              f"gains={_control.get('gains', {})}", flush=True)
    return _control


def capture_inputs(control, store, x, ids, weights):
    config = control.get("capture")
    if not config or store.rank != 0 or store.layer not in config["layers"]:
        return
    key = (config["id"], store.layer)
    done = _capture_rows.get(key, 0)
    take = min(int(config["rows"]) - done, x.shape[0])
    if take <= 0:
        return
    folder = ROOT / "quality-activations" / config["id"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"layer-{store.layer:03d}-rows-{done:05d}.pt"
    if path.exists():
        raise RuntimeError("refusing to overwrite activation trace")
    torch.save({"x": x[:take].detach().cpu().clone(),
                "ids": ids[:take].detach().cpu().clone(),
                "weights": weights[:take].detach().float().cpu().clone(),
                "layer": store.layer, "first_row": done}, path)
    os.chmod(path, 0o644)
    with path.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    _capture_rows[key] = done + take
    atomic_json(folder / f"layer-{store.layer:03d}-STATUS.json", {
        "rows": done + take, "target": config["rows"], "layer": store.layer,
        "control_id": control["id"], "capture_id": config["id"],
        "passed": done + take == config["rows"],
    })


def scaled_runtime(runtime, gains):
    def apply(*args, **kwargs):
        state = args[3]
        bits = str(state._exl3_bits)
        output = runtime.apply_exl3_fused_moe(*args, **kwargs)
        gain = float(gains.get(bits, 1.0))
        if gain != 1.0:
            output.mul_(gain)
        return output
    return SimpleNamespace(apply_exl3_fused_moe=apply)


def apply_stored_scales(store, control):
    """Update the actual FP16 down.svh vectors addressed by fused pointers.

    Original vectors are copied to CPU only for changed groups. Every variant
    starts from those exact originals, so rounding errors never compound.
    The small snapshots stay outside the immutable checkpoint and are bounded
    by the existing down.svh payload, about 236 MiB across all groups per rank.
    """
    if getattr(store, "_quality_scale_version", None) == control["id"]:
        return
    if not hasattr(store, "_quality_original_svh"):
        store._quality_original_svh = {}
        store._quality_applied_gains = {}
    by_expert = control.get("expert_gains", {}).get(str(store.layer), {})
    for bits in (2, 3):
        group = store.mixed_groups[bits]
        global_gain = float(control.get("gains", {}).get(str(bits), 1.0))
        desired = tuple(global_gain * float(by_expert.get(str(expert), 1.0))
                        for expert in group["experts"])
        old = store._quality_applied_gains.get(bits, (1.0,) * len(desired))
        if desired == old:
            continue
        if bits not in store._quality_original_svh:
            store._quality_original_svh[bits] = [pack["down"].svh.detach().cpu().clone()
                                                for pack in group["packs"]]
        for index, (pack, gain, previous) in enumerate(zip(group["packs"], desired, old)):
            if gain != previous:
                original = store._quality_original_svh[bits][index]
                value = (original.float() * gain).to(dtype=original.dtype)
                if not torch.isfinite(value).all():
                    raise ValueError("refitted FP16 scale overflow")
                pack["down"].svh.copy_(value)
        store._quality_applied_gains[bits] = desired
    store._quality_scale_version = control["id"]


def apply_patches():
    if not CONTROL.is_file() or getattr(apply_patches, "_done", False):
        return
    validate_control(json.loads(CONTROL.read_text()))
    # Permit a scoped stack snapshot on an experiment worker if a kernel call
    # stops progressing; no ptrace or system permission changes are needed.
    faulthandler.register(signal.SIGUSR2, all_threads=True)
    import lazy_k275_patch as mixed
    original = mixed._apply_mixed_resident

    def apply(runtime, layer, store, x, ids, weights, limit):
        control = refresh(store)
        capture_inputs(control, store, x, ids, weights)
        apply_stored_scales(store, control)
        selected = control.get("swiglu_limit", limit)
        effective_limit = math.inf if selected == "unclamped" else float(selected)
        return original(runtime, layer, store, x, ids, weights, effective_limit)

    mixed._apply_mixed_resident = apply
    apply_patches._done = True
    print("GLM53_QUALITY_EXPERIMENT_PATCH_ENABLED", flush=True)
