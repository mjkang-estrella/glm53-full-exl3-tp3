#!/usr/bin/env python3
"""Crash-resumable K2-subset coordinator for the 2.75bpw candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tp3k3.geometry import GEOMETRY_ID, ROUTED_LAYERS
from tp3k3.lease_queue import (
    NODES,
    atomic_json,
    counts,
    finish,
    initialize,
    lease_next,
    prefer_pending_node,
    queue_directories,
    read,
    requeue,
)


PROJECT = Path("/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3")
STATE_ROOT = Path("/home/mj-kang/Dev/state/glm53-full-exl3-tp3")
SELECTION = STATE_ROOT / "k275/selection/k2-experts.json"
PUBLISH_PREFIX = "/mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-2.75bpw-TP3-K2-rotating-uneven-v1-"
SERVICES = dict(zip(NODES, ("glm53-exl3-head", "glm53-exl3-worker", "minimax-h3-comfy")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def run(command: list[str], *, check: bool = True, capture: bool = False):
    return subprocess.run(command, cwd=PROJECT, check=check, text=True, capture_output=capture)


def ssh(node: str, command: str, *, check: bool = True):
    return run(["ssh", "-F", "zima-ssh-config", node, command], check=check, capture=True)


def session_name(run_stamp: str, node: str) -> str:
    return f"glm53-k275-{run_stamp[9:15]}-{node[-1]}"


def worker_result(queue_root: Path, layer: int, lease_id: str) -> Path:
    return queue_root / "results" / f"layer-{layer:03d}-{lease_id}.json"


def selection_for(layer: int) -> list[int]:
    value = json.loads(SELECTION.read_text(encoding="utf-8"))
    row = value["layers"][str(layer)]
    selected = [int(x) for x in row["k2_experts"]]
    if len(selected) != 64 or selected != sorted(set(selected)):
        raise ValueError(f"invalid K2 selection for layer {layer}")
    return selected


def all_protected_stopped() -> bool:
    for node, service in SERVICES.items():
        result = ssh(node, f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(service)} 2>/dev/null || echo missing", check=False)
        if result.stdout.strip() != "false":
            return False
    return True


def stop_workers(run_stamp: str) -> None:
    for node in NODES:
        prefix = f"glm53-k275-{run_stamp[9:15]}-"
        ssh(node, "for name in $(docker ps --format '{{.Names}}' | awk '/^glm53-tp3-k2-uneven-L[0-9][0-9][0-9]$/ {print}'); do docker stop --time 30 \"$name\" >/dev/null 2>&1 || true; done", check=False)
        ssh(node, f"tmux list-sessions -F '#{{session_name}}' 2>/dev/null | awk '/^{prefix}/ {{print}}' | xargs -r -n1 tmux kill-session -t", check=False)
        run(["tmux", "kill-session", "-t", session_name(run_stamp, node)], check=False)


def strict_audit(start: str, state: Path) -> tuple[bool, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = state / "audits" / f"kernel-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(["python3", "scripts/kernel_audit.py", "--since", start, "--until", utc_now(), "--output", str(output)], check=False)
    return result.returncode == 0, output


def validate_published(publish_root: Path, layer: int, result: dict) -> bool:
    target = publish_root / f"layer-{layer:03d}"
    verification = target / "VERIFICATION.json"
    receipt = target / "LAYER_K2_RECEIPT.json"
    ledger = target / "ARTIFACT_SHA256SUMS"
    if result.get("publish_path") != str(target) or not all(path.is_file() for path in (verification, receipt, ledger)):
        return False
    try:
        verified = json.loads(verification.read_text(encoding="utf-8"))
        layer_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        return (
            verified.get("passed") is True
            and verified.get("geometry_id") == GEOMETRY_ID
            and verified.get("layer") == layer
            and verified.get("bits") == 2
            and verified.get("verified_experts") == 64
            and verified.get("expected_experts") == selection_for(layer)
            and layer_receipt.get("passed") is True
            and layer_receipt.get("layer") == layer
            and layer_receipt.get("bits") == 2
            and layer_receipt.get("k2_experts") == selection_for(layer)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def estimate(queue_root: Path, started: datetime) -> dict:
    durations = []
    for path in (queue_root / "completed").glob("layer-*.json"):
        try:
            completed = read(path)
            # Layer 3 is a precomputed qualification artifact and does not
            # represent a full 64-expert K2 bulk wall time.
            if int(completed.get("layer", -1)) == 3:
                continue
            value = completed["result"]
            durations.append((datetime.fromisoformat(value["ended_at"].replace("Z", "+00:00")) - datetime.fromisoformat(value["started_at"].replace("Z", "+00:00"))).total_seconds())
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            pass
    seconds_per_layer = sum(durations) / len(durations) if durations else 1800.0
    current = counts(queue_root)
    remaining = current["pending"] + current["leased"]
    seconds = remaining * seconds_per_layer / len(NODES)
    return {
        "basis": "completed_wall_seconds" if durations else "initial_30_minute_projection",
        "seconds_per_layer": seconds_per_layer,
        "remaining_seconds": seconds,
        "estimated_completion_utc": (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def write_status(state: Path, queue_root: Path, publish_root: Path, run_stamp: str, start: str) -> None:
    active = []
    for path in sorted((queue_root / "leased").glob("layer-*.json")):
        lease = read(path)
        progress_path = state / "worker-progress" / f"layer-{int(lease['layer']):03d}-{lease['lease_id']}.json"
        progress = {}
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        active.append({**lease, "session": session_name(run_stamp, lease["node"]), "progress": progress})
    current = counts(queue_root)
    estimate_value = estimate(queue_root, datetime.fromisoformat(start.replace("Z", "+00:00")))
    status = {
        "schema": "glm53-full-exl3-tp3.k275-bulk-status.v1",
        "geometry_id": GEOMETRY_ID,
        "target_bpw": "2.75",
        "run_stamp": run_stamp,
        "updated_at": utc_now(),
        "state": "RUNNING",
        "counts": current,
        "active_leases": active,
        "publish_root": str(publish_root),
        "selection": str(SELECTION),
        "resume_command": f"bash {PROJECT}/scripts/resume_k275_bulk.sh {run_stamp}",
        "estimate": estimate_value,
        "service_policy": "leave_3bpw_candidate_and_Flash_H3_stopped",
    }
    atomic_json(state / "STATUS.json", status)
    markdown = f"""# GLM-5.3 TP3 2.75bpw K2/K3 build running\n\nUpdated: {status['updated_at']}\n\nThe 2.75bpw candidate encodes 64 low-error experts per layer at K2 and reuses 192 K3 experts from the sealed 3.0bpw checkpoint.\n\n- Run: `{run_stamp}`\n- Pending: {current['pending']}\n- Leased: {current['leased']}\n- Completed K2 layers: {current['completed']} / {len(ROUTED_LAYERS)}\n- Failed attempts: {current['failed']}\n- ETA: `{estimate_value['estimated_completion_utc']}`\n- Publish root: `{publish_root}`\n- Selection map: `{SELECTION}`\n\n## Active workers\n\n""" + ("\n".join(f"- `{item['session']}`: {item['node']}, layer {item['layer']}, phase `{item['progress'].get('phase', 'starting')}`, receipts {item['progress'].get('expert_receipts', 0)}" for item in active) or "- No active lease at this polling instant.") + f"""\n\n## Recovery\n\n```bash\nbash {PROJECT}/scripts/resume_k275_bulk.sh {run_stamp}\n```\n\nExperimental encoders stop on kernel, OOM, swap-thrash, or corrupt-receipt evidence. The existing 3.0bpw candidate and Flash/H3 services remain stopped during this build.\n"""
    atomic_text(PROJECT / "state/STATUS.md", markdown)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
        handle.write(value)
        handle.flush()
    Path(temporary).replace(path)


def block(state: Path, queue_root: Path, run_stamp: str, reason: str, evidence: str | None) -> int:
    stop_workers(run_stamp)
    for path in sorted((queue_root / "leased").glob("layer-*.json")):
        lease = read(path)
        requeue(queue_root, int(lease["layer"]), lease["lease_id"], reason)
    payload = {"schema": "glm53-full-exl3-tp3.k275-blocked.v1", "blocked_at": utc_now(), "run_stamp": run_stamp, "reason": reason, "evidence": evidence, "counts": counts(queue_root), "service_policy": "leave_3bpw_candidate_and_Flash_H3_stopped"}
    atomic_json(state / "BLOCKED.json", payload)
    atomic_text(state / "BLOCKED.md", f"# 2.75bpw build blocked\n\n- Reason: `{reason}`\n- Evidence: `{evidence}`\n- Recovery: `bash {PROJECT}/scripts/resume_k275_bulk.sh {run_stamp}`\n")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--lease-timeout", type=float, default=4 * 3600)
    args = parser.parse_args()
    if not __import__("re").fullmatch(r"\d{8}T\d{6}Z", args.run_stamp):
        parser.error("invalid run stamp")
    if not SELECTION.is_file():
        parser.error(f"selection map missing: {SELECTION}")
    state = STATE_ROOT / "k275/bulk" / args.run_stamp
    queue_root = state / "queue"
    publish_root = Path(PUBLISH_PREFIX + args.run_stamp)
    state.mkdir(parents=True, exist_ok=True)
    initialize(queue_root, ROUTED_LAYERS)
    pending_layer3 = queue_root / "pending/layer-003.json"
    if pending_layer3.is_file():
        prefer_pending_node(queue_root, 3, "mj-spark-3", evidence="precomputed K2 layer-003 qualification")
    lock_handle = (state / "manager.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another K2 manager holds the run lock")
    start_path = state / "bulk-window-start.txt"
    if not start_path.is_file():
        atomic_text(start_path, utc_now() + "\n")
    start = start_path.read_text(encoding="utf-8").strip()
    last_audit = 0.0
    try:
        while True:
            if not all_protected_stopped():
                return block(state, queue_root, args.run_stamp, "protected_service_running", None)
            directories = queue_directories(queue_root)
            for lease_path in sorted(directories["leased"].glob("layer-*.json")):
                lease = read(lease_path)
                layer = int(lease["layer"])
                result_path = worker_result(queue_root, layer, lease["lease_id"])
                if result_path.exists():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result.get("passed") is True:
                        if not validate_published(publish_root, layer, result):
                            return block(state, queue_root, args.run_stamp, f"invalid_published_layer_{layer:03d}", str(result_path))
                        finish(queue_root, layer, lease["lease_id"], result)
                        run(["tmux", "kill-session", "-t", session_name(args.run_stamp, lease["node"])], check=False)
                    else:
                        attempts = int(lease.get("attempts", 1))
                        requeue(queue_root, layer, lease["lease_id"], result.get("reason", "worker_failed"))
                        run(["tmux", "kill-session", "-t", session_name(args.run_stamp, lease["node"])], check=False)
                        if attempts >= 3:
                            return block(state, queue_root, args.run_stamp, f"repeated_worker_failure_layer_{layer:03d}", str(result_path))
                else:
                    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(lease["leased_at"].replace("Z", "+00:00"))).total_seconds()
                    if elapsed > args.lease_timeout:
                        return block(state, queue_root, args.run_stamp, f"lease_timeout_layer_{layer:03d}", str(lease_path))
            if time.monotonic() - last_audit >= 60:
                passed, audit_path = strict_audit(start, state)
                last_audit = time.monotonic()
                if not passed:
                    return block(state, queue_root, args.run_stamp, "strict_kernel_audit_failed", str(audit_path))
            for node in NODES:
                lease = lease_next(queue_root, node)
                if lease is None:
                    continue
                layer = int(lease["layer"])
                experts = ",".join(map(str, selection_for(layer)))
                command = shlex.join(["bash", str(PROJECT / "scripts/k275_bulk_worker.sh"), args.run_stamp, str(layer), node, lease["lease_id"], str(publish_root), experts])
                worker_log = state / "worker-logs" / f"layer-{layer:03d}-{node}-{lease['lease_id']}.log"
                worker_log.parent.mkdir(parents=True, exist_ok=True)
                run(["tmux", "new-session", "-d", "-s", session_name(args.run_stamp, node), f"{command} >> {shlex.quote(str(worker_log))} 2>&1"])
            write_status(state, queue_root, publish_root, args.run_stamp, start)
            current = counts(queue_root)
            if current["pending"] == 0 and current["leased"] == 0 and current["completed"] == len(ROUTED_LAYERS):
                atomic_json(state / "COMPLETE.json", {"schema": "glm53-full-exl3-tp3.k275-complete.v1", "completed_at": utc_now(), "run_stamp": args.run_stamp, "layers": len(ROUTED_LAYERS), "publish_root": str(publish_root), "target_bpw": "2.75"})
                atomic_text(PROJECT / "state/STATUS.md", f"# GLM-5.3 TP3 2.75bpw K2 subset bulk complete\n\nRun `{args.run_stamp}` published 64 verified K2 experts for every one of 76 routed layers. Assemble the mixed checkpoint from `{publish_root}` and the sealed 3.0bpw checkpoint.\n")
                return 0
            time.sleep(10)
    except KeyboardInterrupt:
        return block(state, queue_root, args.run_stamp, "manager_interrupted", None)


if __name__ == "__main__":
    raise SystemExit(main())
