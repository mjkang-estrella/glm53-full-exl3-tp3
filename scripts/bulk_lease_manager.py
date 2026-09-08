#!/usr/bin/env python3
"""Persistent Zima coordinator for atomic, recoverable K3 layer leases."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tp3k3.geometry import GEOMETRY_ID, ROUTED_LAYERS
from tp3k3.lease_queue import NODES, atomic_json, counts, finish, initialize, lease_next, read, requeue


PROJECT = Path("/home/mj-kang/Dev/experiment/glm53-full-exl3-tp3")
STATE_ROOT = Path("/home/mj-kang/Dev/state/glm53-full-exl3-tp3")
SERVICES = dict(
    zip(NODES, ("glm53-exl3-head", "glm53-exl3-worker", "minimax-h3-comfy"))
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(command: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT,
        check=check,
        text=True,
        capture_output=capture,
    )


def ssh(node: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        ["ssh", "-F", "zima-ssh-config", node, command],
        check=check,
        capture=True,
    )


def tmux_session(run_stamp: str, node: str) -> str:
    return f"glm53-k3-lease-{run_stamp[9:15]}-{node[-1]}"


def worker_result(queue_root: Path, layer: int, lease_id: str) -> Path:
    return queue_root / "results" / f"layer-{layer:03d}-{lease_id}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_published_result(
    publish_root: Path, layer: int, result: dict[str, Any]
) -> bool:
    target = publish_root / f"layer-{layer:03d}"
    receipt = target / "LAYER_RECEIPT.json"
    ledger = target / "ARTIFACT_SHA256SUMS"
    verification = target / "VERIFICATION.json"
    if result.get("publish_path") != str(target):
        return False
    if not all(path.is_file() for path in (receipt, ledger, verification)):
        return False
    try:
        verified = json.loads(verification.read_text(encoding="utf-8"))
        return (
            sha256_file(receipt) == result.get("layer_receipt_sha256")
            and sha256_file(ledger) == result.get("artifact_ledger_sha256")
            and verified.get("passed") is True
            and verified.get("geometry_id") == GEOMETRY_ID
            and verified.get("layer") == layer
            and verified.get("experts") == 256
        )
    except (OSError, json.JSONDecodeError):
        return False


def worker_alive(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session], check=False).returncode == 0


def launch_worker(
    *, run_stamp: str, queue_root: Path, publish_root: Path, lease: dict[str, Any], log_dir: Path
) -> None:
    node = lease["node"]
    layer = int(lease["layer"])
    lease_id = lease["lease_id"]
    session = tmux_session(run_stamp, node)
    if worker_result(queue_root, layer, lease_id).exists() or worker_alive(session):
        return
    log = log_dir / f"layer-{layer:03d}-{node}-{lease_id}.log"
    command = shlex.join(
        [
            "bash",
            str(PROJECT / "scripts/bulk_lease_worker.sh"),
            run_stamp,
            str(layer),
            node,
            lease_id,
            str(publish_root),
        ]
    )
    shell_command = f"{command} >> {shlex.quote(str(log))} 2>&1"
    run(["tmux", "new-session", "-d", "-s", session, shell_command])


def all_protected_stopped() -> bool:
    for node, service in SERVICES.items():
        result = ssh(
            node,
            f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(service)} 2>/dev/null || echo missing",
            check=False,
        )
        if result.stdout.strip() != "false":
            return False
    return True


def stop_active_encoders(run_stamp: str) -> None:
    for node in NODES:
        # Names are fixed by run_encoder.sh and constrained to layers 003..078.
        command = (
            "for name in $(docker ps --format '{{.Names}}' | "
            "awk '/^glm53-tp3-k3-uneven-L[0-9][0-9][0-9]$/ {print}'); do "
            "docker stop --time 30 \"$name\" >/dev/null 2>&1 || true; done"
        )
        ssh(node, command, check=False)
        ssh(
            node,
            f"tmux list-sessions -F '#{{session_name}}' 2>/dev/null | "
            f"awk '/^glm53-k3-{run_stamp[9:15]}-L/ {{print}}' | "
            "xargs -r -n1 tmux kill-session -t",
            check=False,
        )


def terminate_local_workers(run_stamp: str) -> None:
    for node in NODES:
        run(["tmux", "kill-session", "-t", tmux_session(run_stamp, node)], check=False)


def strict_audit(start: str, state: Path) -> tuple[bool, Path]:
    now = utc_now()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = state / "audits" / f"kernel-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        [
            "python3",
            "scripts/kernel_audit.py",
            "--since",
            start,
            "--until",
            now,
            "--output",
            str(output),
        ],
        check=False,
    )
    return result.returncode == 0, output


def estimate(queue_root: Path, qualification_seconds: float) -> dict[str, Any]:
    durations = []
    for path in (queue_root / "completed").glob("layer-*.json"):
        value = read(path)
        # Layer 3 reused the complete qualification payload, so its first bulk
        # wall time measures only validation/publication and is not an encode
        # throughput sample.
        if int(value.get("layer", -1)) == 3:
            continue
        result = value.get("result", {})
        try:
            durations.append(
                (parse_utc(result["ended_at"]) - parse_utc(result["started_at"])).total_seconds()
            )
        except (KeyError, TypeError, ValueError):
            pass
    seconds_per_layer = sum(durations) / len(durations) if durations else qualification_seconds
    current = counts(queue_root)
    remaining = current["pending"] + current["leased"]
    seconds = remaining * seconds_per_layer / len(NODES)
    return {
        "basis": "completed_wall_seconds" if durations else "qualified_layer_expert_seconds",
        "seconds_per_layer": seconds_per_layer,
        "remaining_seconds": seconds,
        "estimated_completion_utc": (
            datetime.now(timezone.utc) + timedelta(seconds=seconds)
        ).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def write_status(
    state: Path,
    queue_root: Path,
    publish_root: Path,
    run_stamp: str,
    rollback_stamp: str,
    qualification_seconds: float,
) -> None:
    leases = []
    for path in sorted((queue_root / "leased").glob("layer-*.json")):
        value = read(path)
        value["local_tmux_session"] = tmux_session(run_stamp, value["node"])
        progress_path = (
            state
            / "worker-progress"
            / f"layer-{int(value['layer']):03d}-{value['lease_id']}.json"
        )
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if progress.get("lease_id") == value["lease_id"]:
                    value["progress"] = progress
            except (OSError, json.JSONDecodeError):
                # A worker writes this file atomically; a missing/unreadable
                # advisory heartbeat never changes the authoritative lease.
                pass
        leases.append(value)
    estimate_value = estimate(queue_root, qualification_seconds)
    status = {
            "schema": "glm53-full-exl3-tp3.bulk-status.v1",
            "geometry_id": GEOMETRY_ID,
            "run_stamp": run_stamp,
            "updated_at": utc_now(),
            "state": "RUNNING",
            "counts": counts(queue_root),
            "active_leases": leases,
            "publish_root": str(publish_root),
            "queue_root": str(queue_root),
            "resume_command": f"bash {PROJECT}/scripts/resume_bulk.sh {run_stamp}",
            "optional_manual_restore_command": (
                f"bash {PROJECT}/scripts/restore_protected_services.sh {run_stamp} "
                f"$(cat {state}/bulk-window-start.txt) {state}/service-baseline.json "
                f"{state}/manual-restore"
            ),
            "failure_service_policy": "leave_flash_and_h3_stopped",
            "rollback_inventory": str(STATE_ROOT / "rollback" / rollback_stamp),
            "estimate": estimate_value,
        }
    atomic_json(state / "STATUS.json", status)
    active = "\n".join(
        (
            f"- `{item['local_tmux_session']}`: {item['node']}, layer {item['layer']}, "
            f"lease `{item['lease_id']}`, phase "
            f"`{item.get('progress', {}).get('phase', 'starting')}`, verified expert receipts "
            f"{item.get('progress', {}).get('expert_receipts', 0)}, last progress "
            f"`{item.get('progress', {}).get('updated_at', item['leased_at'])}`"
        )
        for item in leases
    ) or "- No active lease during this polling instant; verification, publication, or the next polling cycle may be in progress."
    markdown = f"""# Rotating uneven K3 bulk encode running

Updated: {status['updated_at']}

The Phase 0 and Phase 1 gates passed. Bulk uniform-K3 encoding is active under
geometry `{GEOMETRY_ID}`. No K4 promotion, checkpoint assembly, client route,
or public route change is authorized by this run.

## Progress

- Run: `{run_stamp}`
- Pending: {status['counts']['pending']}
- Leased: {status['counts']['leased']}
- Completed and atomically published: {status['counts']['completed']} / {len(ROUTED_LAYERS)}
- Failed-attempt receipts: {status['counts']['failed']}
- Estimated completion: {estimate_value['estimated_completion_utc']}
- Estimate basis: {estimate_value['basis']} at {estimate_value['seconds_per_layer']:.1f} seconds/layer

## Worker sessions

{active}

## Durable paths

- Queue and receipts: `{queue_root}`
- Manager status: `{state / 'STATUS.json'}`
- Worker logs: `{state / 'worker-logs'}`
- Strict kernel audits: `{state / 'audits'}`
- Sealed UNAS layers: `{publish_root}`
- Rollback inventory: `{STATE_ROOT / 'rollback' / rollback_stamp}`

## Exact commands

Resume the manager after a Zima control-process interruption, only while the
protected services remain stopped:

```bash
bash {PROJECT}/scripts/resume_bulk.sh {run_stamp}
```

Failures leave Flash and H3 stopped. Restore them only as a separate, optional
operator action:

```bash
bash {PROJECT}/scripts/restore_protected_services.sh {run_stamp} \\
  "$(cat {state}/bulk-window-start.txt)" \\
  {state}/service-baseline.json \\
  {state}/manual-restore
```

After a recorded hard stop, establish a new strict encoder window, reconcile
unfinished work, and resume without starting Flash/H3:

```bash
bash {PROJECT}/scripts/restart_bulk_after_block.sh {run_stamp}
```
"""
    atomic_text(state / "STATUS.md", markdown)
    atomic_text(STATE_ROOT / "STATUS.md", markdown)
    atomic_text(PROJECT / "state/STATUS.md", markdown)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--rollback-stamp", required=True)
    parser.add_argument("--publish-root", type=Path, required=True)
    parser.add_argument("--qualification-seconds", type=float, required=True)
    parser.add_argument("--audit-interval", type=float, default=60.0)
    parser.add_argument("--lease-timeout", type=float, default=14400.0)
    args = parser.parse_args()
    if not __import__("re").fullmatch(r"\d{8}T\d{6}Z", args.run_stamp):
        parser.error("invalid run stamp")
    if args.qualification_seconds <= 0:
        parser.error("qualification seconds must be positive")
    required_prefix = "/mnt/unas-models/ZAI/GLM-5.3-EXL3-TR3-3.0bpw-TP3-K3-rotating-uneven-v1-"
    if not str(args.publish_root).startswith(required_prefix):
        parser.error("publish root is outside the K3 uneven namespace")

    state = STATE_ROOT / "bulk" / args.run_stamp
    queue_root = state / "queue"
    log_dir = state / "worker-logs"
    bulk_start = (state / "bulk-window-start.txt").read_text(encoding="utf-8").strip()
    log_dir.mkdir(parents=True, exist_ok=True)
    initialize(queue_root, ROUTED_LAYERS)

    lock_handle = (state / "manager.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another bulk lease manager holds the run lock")

    stopping = False

    def requested_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, requested_stop)
    signal.signal(signal.SIGINT, requested_stop)
    last_audit = 0.0
    failure_reason: str | None = None
    failure_evidence: str | None = None

    while True:
        if stopping:
            failure_reason = "manager_received_termination"
            break
        if not all_protected_stopped():
            failure_reason = "protected_service_running_during_bulk"
            break

        for lease_path in sorted((queue_root / "leased").glob("layer-*.json")):
            lease = read(lease_path)
            layer = int(lease["layer"])
            lease_id = lease["lease_id"]
            result_path = worker_result(queue_root, layer, lease_id)
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    result.get("lease_id") != lease_id
                    or result.get("layer") != layer
                    or result.get("node") != lease["node"]
                ):
                    failure_reason = f"mismatched_atomic_result_layer_{layer:03d}"
                    failure_evidence = str(result_path)
                    break
                if result.get("passed") is not True:
                    requeue(queue_root, layer, lease_id, result.get("reason", "worker_failed"))
                    failure_reason = f"worker_failed_layer_{layer:03d}"
                    failure_evidence = str(result_path)
                    break
                if not validate_published_result(args.publish_root, layer, result):
                    failure_reason = f"invalid_published_result_layer_{layer:03d}"
                    failure_evidence = str(result_path)
                    break
                finish(queue_root, layer, lease_id, result)
                run(["tmux", "kill-session", "-t", tmux_session(args.run_stamp, lease["node"])], check=False)
                continue
            elapsed = (datetime.now(timezone.utc) - parse_utc(lease["leased_at"])).total_seconds()
            if elapsed > args.lease_timeout:
                requeue(queue_root, layer, lease_id, "lease_timeout")
                failure_reason = f"lease_timeout_layer_{layer:03d}"
                failure_evidence = str(lease_path)
                break
            launch_worker(
                run_stamp=args.run_stamp,
                queue_root=queue_root,
                publish_root=args.publish_root,
                lease=lease,
                log_dir=log_dir,
            )
        if failure_reason:
            break

        if time.monotonic() - last_audit >= args.audit_interval:
            passed, audit_path = strict_audit(bulk_start, state)
            last_audit = time.monotonic()
            if not passed:
                failure_reason = "strict_kernel_audit_failed"
                failure_evidence = str(audit_path)
                break

        for node in NODES:
            lease = lease_next(queue_root, node)
            if lease is not None:
                launch_worker(
                    run_stamp=args.run_stamp,
                    queue_root=queue_root,
                    publish_root=args.publish_root,
                    lease=lease,
                    log_dir=log_dir,
                )
        write_status(
            state,
            queue_root,
            args.publish_root,
            args.run_stamp,
            args.rollback_stamp,
            args.qualification_seconds,
        )
        current = counts(queue_root)
        if current["pending"] == 0 and current["leased"] == 0 and current["completed"] == len(ROUTED_LAYERS):
            break
        time.sleep(10)

    if failure_reason:
        terminate_local_workers(args.run_stamp)
        stop_active_encoders(args.run_stamp)
        for path in sorted((queue_root / "leased").glob("layer-*.json")):
            lease = read(path)
            requeue(queue_root, lease["layer"], lease["lease_id"], failure_reason)
        atomic_json(
            state / "BLOCKED.json",
            {
                "schema": "glm53-full-exl3-tp3.bulk-blocked.v2",
                "blocked_at": utc_now(),
                "reason": failure_reason,
                "evidence": failure_evidence,
                "experimental_workers_stopped": True,
                "automatic_service_restore_attempted": False,
                "service_policy": "leave_flash_and_h3_stopped",
                "counts": counts(queue_root),
                "exact_next_action": f"run bash {PROJECT}/scripts/restart_bulk_after_block.sh {args.run_stamp}",
            },
        )
        blocked_markdown = f"""# Rotating uneven K3 bulk encode blocked

Blocked: {utc_now()}

- Reason: `{failure_reason}`
- Evidence: `{failure_evidence}`
- Experimental workers stopped: `true`
- Automatic Flash/H3 restore attempted: `false`
- Service policy: leave Flash/H3 stopped and preserve their configuration
- Queue: `{queue_root}`

Exact recovery command (it creates a fresh strict encoder window and does not
start Flash/H3):

```bash
bash {PROJECT}/scripts/restart_bulk_after_block.sh {args.run_stamp}
```
"""
        atomic_text(state / "BLOCKED.md", blocked_markdown)
        atomic_text(STATE_ROOT / "BLOCKED.md", blocked_markdown)
        atomic_text(PROJECT / "state/BLOCKED.md", blocked_markdown)
        return 1

    atomic_json(
        state / "COMPLETE.json",
        {
            "schema": "glm53-full-exl3-tp3.bulk-complete.v2",
            "completed_at": utc_now(),
            "layers": len(ROUTED_LAYERS),
            "publish_root": str(args.publish_root),
            "automatic_service_restore_attempted": False,
            "service_policy": "leave_flash_and_h3_stopped",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
