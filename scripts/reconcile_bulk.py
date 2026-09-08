#!/usr/bin/env python3
"""Reconcile queue records, sealed publications, and resumable partial layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tp3k3.geometry import GEOMETRY_ID, ROUTED_LAYERS
from tp3k3.lease_queue import (
    NODES,
    counts,
    finish,
    prefer_pending_node,
    read,
    requeue,
)


PROJECT = Path("/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3")
STATE_ROOT = Path("/home/mj-kang/Dev/state/glm53-full-exl3-tp3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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


def publication_check(publish_root: Path, layer: int, result: dict[str, Any]) -> dict[str, Any]:
    target = publish_root / f"layer-{layer:03d}"
    errors: list[str] = []
    if result.get("passed") is not True:
        errors.append("completed queue result is not passing")
    if result.get("layer") != layer:
        errors.append("completed queue result layer differs")
    if result.get("publish_path") != str(target):
        errors.append("completed queue publish path differs")
    receipt = target / "LAYER_RECEIPT.json"
    ledger = target / "ARTIFACT_SHA256SUMS"
    verification = target / "VERIFICATION.json"
    for required in (receipt, ledger, verification):
        if not required.is_file():
            errors.append(f"missing {required.name}")
    if not errors:
        try:
            if sha256_file(receipt) != result.get("layer_receipt_sha256"):
                errors.append("layer receipt SHA-256 differs")
            if sha256_file(ledger) != result.get("artifact_ledger_sha256"):
                errors.append("artifact ledger SHA-256 differs")
            verified = json.loads(verification.read_text(encoding="utf-8"))
            if not (
                verified.get("passed") is True
                and verified.get("geometry_id") == GEOMETRY_ID
                and verified.get("layer") == layer
                and verified.get("experts") == 256
            ):
                errors.append("independent layer verification identity differs")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    return {
        "layer": layer,
        "path": str(target),
        "key_receipts_checksum_verified": not errors,
        "errors": errors,
    }


def last_lease_node(value: dict[str, Any]) -> str | None:
    for event in reversed(value.get("history", [])):
        node = event.get("node")
        if event.get("event") in {"leased", "requeued"} and node in NODES:
            return str(node)
    return None


def remote_partial(node: str, layer: int, ssh_config: Path) -> dict[str, Any]:
    layer_dir = (
        "/home/mj-kang/Dev/cache/glm53-full-exl3-tp3/encoded/"
        f"{GEOMETRY_ID}/layer-{layer:03d}"
    )
    command = [
        "ssh",
        "-F",
        str(ssh_config),
        node,
        "python3",
        str(PROJECT / "scripts/verify_partial_layer.py"),
        "--layer-dir",
        layer_dir,
        "--layer",
        str(layer),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        value = {
            "schema": "glm53-full-exl3-tp3.partial-layer-verification.v1",
            "layer": layer,
            "verified_experts": 0,
            "passed": False,
            "errors": [f"remote verifier exit {result.returncode}: {result.stderr.strip()}"],
        }
    value["node"] = node
    value["remote_exit_code"] = result.returncode
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--publish-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssh-config", type=Path, default=Path("zima-ssh-config"))
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()

    state = STATE_ROOT / "bulk" / args.run_stamp
    queue = state / "queue"
    errors: list[str] = []
    actions: list[dict[str, Any]] = []

    # Resolve stale leases only from their exact atomic result. A failed or
    # absent result is requeued after experimental processes have been stopped.
    if args.repair:
        for path in sorted((queue / "leased").glob("layer-*.json")):
            lease = read(path)
            layer = int(lease["layer"])
            lease_id = str(lease["lease_id"])
            result_path = queue / "results" / f"layer-{layer:03d}-{lease_id}.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
            if result and result.get("passed") is True:
                check = publication_check(args.publish_root, layer, result)
                if check["key_receipts_checksum_verified"]:
                    finish(queue, layer, lease_id, result)
                    actions.append({"action": "completed_stale_lease", "layer": layer, "result": str(result_path)})
                    continue
            reason = "recovery_reconcile_failed_result" if result else "recovery_reconcile_interrupted_lease"
            requeue(queue, layer, lease_id, reason)
            actions.append({"action": "requeued_stale_lease", "layer": layer, "reason": reason})

    primary: dict[int, str] = {}
    for directory in ("pending", "leased", "completed"):
        for path in sorted((queue / directory).glob("layer-*.json")):
            value = read(path)
            layer = int(value["layer"])
            if layer in primary:
                errors.append(f"layer {layer} exists in both {primary[layer]} and {directory}")
            primary[layer] = directory
    expected = set(ROUTED_LAYERS)
    if set(primary) != expected:
        errors.append(f"queue layer set differs: missing={sorted(expected-set(primary))} extra={sorted(set(primary)-expected)}")

    publications = []
    completed_layers: set[int] = set()
    for path in sorted((queue / "completed").glob("layer-*.json")):
        value = read(path)
        layer = int(value["layer"])
        completed_layers.add(layer)
        check = publication_check(args.publish_root, layer, value.get("result", {}))
        publications.append(check)
        errors.extend(f"layer {layer}: {item}" for item in check["errors"])
    published_layers = {
        int(path.name.removeprefix("layer-"))
        for path in args.publish_root.glob("layer-[0-9][0-9][0-9]")
        if path.is_dir()
    }
    if published_layers != completed_layers:
        errors.append(
            f"published/completed set differs: published_only={sorted(published_layers-completed_layers)} "
            f"completed_only={sorted(completed_layers-published_layers)}"
        )

    partials = []
    for path in sorted((queue / "pending").glob("layer-*.json")):
        value = read(path)
        node = last_lease_node(value)
        if node is None:
            continue
        layer = int(value["layer"])
        partial = remote_partial(node, layer, args.ssh_config)
        partials.append(partial)
        if partial.get("passed") is not True:
            errors.append(f"layer {layer} partial verifier failed on {node}: {partial.get('errors')}")
        elif args.repair and int(partial.get("verified_experts", 0)) > 0:
            evidence = f"{args.output}#partial-layer-{layer:03d}-{node}"
            prefer_pending_node(queue, layer, node, evidence=evidence)
            actions.append(
                {
                    "action": "preferred_checksum_verified_partial",
                    "layer": layer,
                    "node": node,
                    "verified_experts": partial["verified_experts"],
                }
            )

    value = {
        "schema": "glm53-full-exl3-tp3.bulk-reconciliation.v1",
        "run_stamp": args.run_stamp,
        "geometry_id": GEOMETRY_ID,
        "repair_enabled": args.repair,
        "counts": counts(queue),
        "completed_layers": sorted(completed_layers),
        "published_layers": sorted(published_layers),
        "publications": publications,
        "partials": partials,
        "actions": actions,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(args.output, value)
    print(json.dumps({key: value[key] for key in ("passed", "counts", "actions", "errors")}, indent=2))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
