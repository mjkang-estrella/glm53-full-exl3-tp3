"""Pure lifecycle-window policy for GPU kernel and rollback evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping, Sequence


FLASH_HOSTS = frozenset(("mj-spark-1", "mj-spark-2"))
FAULT = re.compile(
    r"(?:NVRM|Xid|out of memory|oom-kill|kernel OOM|I/O error|"
    r"filesystem error|EXT4-fs error|BTRFS.*error)",
    re.I,
)
RECOVERABLE_FLASH_ALLOCATION = re.compile(
    r"NVRM:.*Out of memory\s*\[NV_ERR_NO_MEMORY\]", re.I
)
XID = re.compile(r"\bXid\b", re.I)


def policy_document() -> dict[str, Any]:
    return {
        "schema": "glm53-full-exl3-tp3.lifecycle-policy.v1",
        "candidate_runtime": {
            "allowed_kernel_faults": 0,
            "allowed_nvrm": 0,
            "allowed_xid": 0,
        },
        "encoder": {
            "allowed_kernel_faults": 0,
            "allowed_nvrm": 0,
            "allowed_xid": 0,
        },
        "protected_flash_cold_start": {
            "hosts": sorted(FLASH_HOSTS),
            "only_recoverable_pattern": RECOVERABLE_FLASH_ALLOCATION.pattern,
            "maximum_retry_span_seconds": 120.0,
            "startup_timeout_seconds": 900.0,
            "requires_both_ranks_alive": True,
            "requires_no_oom_kill": True,
            "requires_health_200": True,
            "requires_generation_finish_reason": "stop",
        },
        "post_ready": {
            "allowed_kernel_faults": 0,
            "minimum_observation_seconds": 10.0,
        },
    }


def parse_timestamp(raw: str) -> datetime:
    value = str(raw).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event(host: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    line = str(raw.get("line", ""))
    return {
        "host": host,
        "timestamp": str(raw.get("timestamp", "")),
        "line": line,
        "fault": bool(FAULT.search(line)),
        "recoverable_flash_allocation": bool(
            RECOVERABLE_FLASH_ALLOCATION.search(line)
        )
        and not bool(XID.search(line)),
    }


def _normal_generation(generation: Mapping[str, Any]) -> bool:
    choices = generation.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, Mapping) or first.get("finish_reason") != "stop":
        return False
    message = first.get("message")
    return isinstance(message, Mapping) and isinstance(message.get("content"), str) and bool(
        message["content"].strip()
    )


def evaluate_lifecycle(
    *,
    events_by_host: Mapping[str, Sequence[Mapping[str, Any]]],
    timeline: Mapping[str, Any],
    service_state: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify kernel events and enforce the complete rollback contract."""

    reasons: list[str] = []
    candidate = timeline.get("candidate_runtime", {})
    flash = timeline.get("protected_flash_rollback", {})
    post = timeline.get("post_ready", {})
    try:
        candidate_start = parse_timestamp(candidate["start"])
        candidate_end = parse_timestamp(candidate["end"])
        cold_start = parse_timestamp(flash["start"])
        ready_raw = flash.get("ready")
        ready = parse_timestamp(ready_raw) if ready_raw else None
        post_end = parse_timestamp(post["end"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema": "glm53-full-exl3-tp3.lifecycle-audit.v1",
            "policy": policy_document(),
            "passed": False,
            "failure_reasons": [f"invalid_timeline:{exc!r}"],
            "timeline": dict(timeline),
            "windows": {},
            "service_checks": {},
        }

    if not candidate_start < candidate_end <= cold_start < post_end:
        reasons.append("lifecycle_window_order")
    if ready is None:
        reasons.append("flash_startup_not_ready")
    elif not cold_start < ready <= post_end:
        reasons.append("flash_ready_window_order")

    policy = policy_document()
    cold_limit = float(
        flash.get(
            "timeout_seconds",
            policy["protected_flash_cold_start"]["startup_timeout_seconds"],
        )
    )
    if ready is not None and (ready - cold_start).total_seconds() > cold_limit:
        reasons.append("flash_startup_timeout")
    post_seconds = (post_end - ready).total_seconds() if ready is not None else 0.0
    if post_seconds < policy["post_ready"]["minimum_observation_seconds"]:
        reasons.append("post_ready_observation_too_short")

    windows: dict[str, list[dict[str, Any]]] = {
        "candidate_runtime": [],
        "protected_flash_cold_start": [],
        "post_ready": [],
        "unclassified_gap": [],
    }
    for host, rows in events_by_host.items():
        for raw in rows:
            record = _event(host, raw)
            if not record["fault"]:
                continue
            try:
                occurred = parse_timestamp(record["timestamp"])
            except ValueError:
                record["disposition"] = "hard_stop"
                record["reason"] = "unparseable_fault_timestamp"
                windows["unclassified_gap"].append(record)
                continue
            if candidate_start <= occurred <= candidate_end:
                name = "candidate_runtime"
            elif cold_start <= occurred and (ready is None or occurred < ready):
                name = "protected_flash_cold_start"
            elif ready is not None and ready <= occurred <= post_end:
                name = "post_ready"
            else:
                name = "unclassified_gap"
            if (
                name == "protected_flash_cold_start"
                and host in FLASH_HOSTS
                and record["recoverable_flash_allocation"]
            ):
                record["disposition"] = "allowed_recovered_retry"
                record["reason"] = "authorized_flash_cold_start_allocation_retry"
            else:
                record["disposition"] = "hard_stop"
                record["reason"] = f"fault_in_{name}"
            windows[name].append(record)

    for name, rows in windows.items():
        hard = [row for row in rows if row["disposition"] == "hard_stop"]
        if hard:
            reasons.append(f"{name}_hard_faults:{len(hard)}")

    retry_times = [
        parse_timestamp(row["timestamp"])
        for row in windows["protected_flash_cold_start"]
        if row["disposition"] == "allowed_recovered_retry"
    ]
    retry_span = (
        (max(retry_times) - min(retry_times)).total_seconds()
        if len(retry_times) > 1
        else 0.0
    )
    if retry_span > policy["protected_flash_cold_start"]["maximum_retry_span_seconds"]:
        reasons.append("persistent_flash_allocation_retry_loop")

    containers = service_state.get("containers", {})
    required = (
        ("mj-spark-1", "glm53-exl3-head"),
        ("mj-spark-2", "glm53-exl3-worker"),
        ("mj-spark-3", "minimax-h3-comfy"),
    )
    container_checks: dict[str, Any] = {}
    for host, name in required:
        key = f"{host}:{name}"
        state = containers.get(key, {})
        passed = (
            state.get("running") is True
            and state.get("oom_killed") is False
            and int(state.get("restart_count_after", -1))
            == int(state.get("restart_count_before", -2))
        )
        container_checks[key] = {"passed": passed, **dict(state)}
        if not passed:
            reasons.append(f"container_state:{key}")
    if service_state.get("worker_death_observed") is not False:
        reasons.append("worker_death_observed")

    health = service_state.get("health", {})
    health_ok = health.get("flash_http") == 200 and health.get("h3_http") == 200
    if not health_ok:
        reasons.append("restored_health_failed")
    generation_ok = _normal_generation(generation)
    if not generation_ok:
        reasons.append("restored_generation_failed")

    # Preserve stable ordering while removing duplicates.
    unique_reasons = list(dict.fromkeys(reasons))
    summarized = {
        name: {
            "fault_count": len(rows),
            "allowed_recovered_retry_count": sum(
                row["disposition"] == "allowed_recovered_retry" for row in rows
            ),
            "hard_stop_count": sum(row["disposition"] == "hard_stop" for row in rows),
            "events": rows,
        }
        for name, rows in windows.items()
    }
    return {
        "schema": "glm53-full-exl3-tp3.lifecycle-audit.v1",
        "policy": policy,
        "timeline": dict(timeline),
        "windows": summarized,
        "flash_retry_span_seconds": retry_span,
        "startup_seconds": (
            (ready - cold_start).total_seconds() if ready is not None else None
        ),
        "post_ready_observation_seconds": post_seconds,
        "service_checks": {
            "containers": container_checks,
            "worker_death_observed": service_state.get("worker_death_observed"),
            "health": {"passed": health_ok, **dict(health)},
            "generation": {"passed": generation_ok},
        },
        "failure_reasons": unique_reasons,
        "passed": not unique_reasons,
    }
