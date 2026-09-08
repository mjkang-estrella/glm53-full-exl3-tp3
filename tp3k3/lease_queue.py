"""Durable, atomic layer leases for the rotating uneven K3 bulk encoder."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
import uuid


SCHEMA = "glm53-full-exl3-tp3.layer-lease.v1"
NODES = ("mj-spark-1", "mj-spark-2", "mj-spark-3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def queue_directories(root: Path) -> dict[str, Path]:
    return {
        name: root / name
        for name in ("pending", "leased", "completed", "failed", "results")
    }


def initialize(root: Path, layers: Iterable[int]) -> None:
    directories = queue_directories(root)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    for layer in sorted(set(layers)):
        if layer < 3 or layer > 78:
            raise ValueError(f"invalid routed layer {layer}")
        name = f"layer-{layer:03d}.json"
        if any((path / name).exists() for path in directories.values() if path.name != "results"):
            continue
        atomic_json(
            directories["pending"] / name,
            {
                "schema": SCHEMA,
                "layer": layer,
                "attempts": 0,
                # Reuse the complete measured layer without moving its evidence.
                "preferred_node": "mj-spark-3" if layer == 3 else None,
                "created_at": utc_now(),
                "history": [],
            },
        )


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError(f"unexpected lease schema in {path}")
    return value


def active_node(root: Path, node: str) -> bool:
    for path in queue_directories(root)["leased"].glob("layer-*.json"):
        if read(path).get("node") == node:
            return True
    return False


def lease_next(root: Path, node: str) -> dict[str, Any] | None:
    if node not in NODES:
        raise ValueError(f"invalid node {node}")
    if active_node(root, node):
        return None
    pending = queue_directories(root)["pending"]
    candidates = sorted(pending.glob("layer-*.json"))
    preferred = [path for path in candidates if read(path).get("preferred_node") == node]
    unreserved = [path for path in candidates if read(path).get("preferred_node") is None]
    reserved_elsewhere = [
        path
        for path in candidates
        if path not in preferred and path not in unreserved
    ]
    # A preferred node is an optimization, not a fixed assignment: consume
    # ordinary work first, but allow takeover if it is the only work left.
    candidates = preferred + unreserved + reserved_elsewhere
    for source in candidates:
        try:
            value = read(source)
            lease_id = uuid.uuid4().hex
            value.update(
                {
                    "lease_id": lease_id,
                    "node": node,
                    "leased_at": utc_now(),
                    "attempts": int(value.get("attempts", 0)) + 1,
                }
            )
            value.setdefault("history", []).append(
                {
                    "event": "leased",
                    "at": value["leased_at"],
                    "node": node,
                    "lease_id": lease_id,
                }
            )
            atomic_json(source, value)
            target = queue_directories(root)["leased"] / source.name
            os.replace(source, target)
            return value
        except FileNotFoundError:
            # Supports a future second coordinator without duplicate leasing.
            continue
    return None


def finish(root: Path, layer: int, lease_id: str, result: dict[str, Any]) -> dict[str, Any]:
    directories = queue_directories(root)
    source = directories["leased"] / f"layer-{layer:03d}.json"
    value = read(source)
    if value.get("lease_id") != lease_id:
        raise ValueError(f"stale lease completion for layer {layer}")
    completed_at = utc_now()
    value.update({"completed_at": completed_at, "result": result})
    value.setdefault("history", []).append(
        {"event": "completed", "at": completed_at, "lease_id": lease_id}
    )
    atomic_json(source, value)
    target = directories["completed"] / source.name
    os.replace(source, target)
    return value


def requeue(root: Path, layer: int, lease_id: str, reason: str) -> dict[str, Any]:
    directories = queue_directories(root)
    source = directories["leased"] / f"layer-{layer:03d}.json"
    value = read(source)
    if value.get("lease_id") != lease_id:
        raise ValueError(f"stale lease failure for layer {layer}")
    failed_at = utc_now()
    failure = {
        "schema": SCHEMA,
        "layer": layer,
        "node": value.get("node"),
        "lease_id": lease_id,
        "attempt": value.get("attempts"),
        "failed_at": failed_at,
        "reason": reason,
    }
    atomic_json(
        directories["failed"]
        / f"layer-{layer:03d}-attempt-{int(value['attempts']):02d}-{lease_id}.json",
        failure,
    )
    value.setdefault("history", []).append(
        {
            "event": "requeued",
            "at": failed_at,
            "node": value.get("node"),
            "lease_id": lease_id,
            "reason": reason,
        }
    )
    for key in ("node", "lease_id", "leased_at"):
        value.pop(key, None)
    atomic_json(source, value)
    target = directories["pending"] / source.name
    os.replace(source, target)
    return value


def prefer_pending_node(
    root: Path, layer: int, node: str, *, evidence: str
) -> dict[str, Any]:
    """Prefer the node holding verified partial work without pinning the lease."""
    if node not in NODES:
        raise ValueError(f"invalid node {node}")
    source = queue_directories(root)["pending"] / f"layer-{layer:03d}.json"
    value = read(source)
    if value.get("preferred_node") == node:
        return value
    changed_at = utc_now()
    value["preferred_node"] = node
    value.setdefault("history", []).append(
        {
            "event": "recovery_preference",
            "at": changed_at,
            "node": node,
            "evidence": evidence,
        }
    )
    atomic_json(source, value)
    return value


def counts(root: Path) -> dict[str, int]:
    directories = queue_directories(root)
    return {
        name: len(list(directories[name].glob("layer-*.json")))
        for name in ("pending", "leased", "completed", "failed")
    }
