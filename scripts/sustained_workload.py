#!/usr/bin/env python3
"""Durable sequential mixed workload with per-request receipts and heartbeat."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from real_test_client import MODEL, completion


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://192.168.0.238:8893")
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        "Give three concise reasons atomic renames protect checkpoint publication.",
        "Compute the greatest common divisor of 123456 and 7890 and explain briefly.",
        "Write a Python function that merges two sorted integer iterators lazily.",
        "한국어로 분산 시스템의 체크섬 검증을 두 문장으로 설명하세요.",
        "Review this code and identify the bug: def total(xs): return sum(x for x in xs if x)",
        "Return JSON with keys status and layers, where status is complete and layers is 76.",
    ]
    started_wall = utc()
    started = time.monotonic()
    deadline = started + args.duration_seconds
    completed = failed = 0
    latencies: list[float] = []
    with (args.state_dir / "requests.jsonl").open("a", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            index = completed + failed
            prompt = prompts[index % len(prompts)]
            request_started_at = utc()
            atomic_json(args.state_dir / "STATUS.json", {
                "schema": "glm53-full-exl3-tp3.sustained-status.v1",
                "started_at": started_wall,
                "updated_at": request_started_at,
                "duration_target_seconds": args.duration_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "completed": completed,
                "failed": failed,
                "phase": "running",
                "current_request": index,
                "current_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "current_request_started_at": request_started_at,
            })
            try:
                row = completion(
                    args.endpoint,
                    [{"role": "user", "content": prompt}],
                    max_tokens=256,
                    chat_template_kwargs={"thinking": False},
                )
                ok = bool(row["content"].strip()) and row["finish_reason"] == "stop"
                row.update(index=index, ok=ok)
                completed += int(ok)
                failed += int(not ok)
                latencies.append(float(row["elapsed_seconds"]))
            except Exception as exc:
                row = {"index": index, "ok": False, "requested_at": utc(), "error": repr(exc)}
                failed += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            atomic_json(args.state_dir / "STATUS.json", {
                "schema": "glm53-full-exl3-tp3.sustained-status.v1",
                "started_at": started_wall,
                "updated_at": utc(),
                "duration_target_seconds": args.duration_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "completed": completed,
                "failed": failed,
                "phase": "running",
                "current_request": None,
                "current_prompt_sha256": None,
                "current_request_started_at": None,
            })
    result = {
        "schema": "glm53-full-exl3-tp3.sustained-result.v1",
        "endpoint": args.endpoint,
        "model": MODEL,
        "started_at": started_wall,
        "completed_at": utc(),
        "elapsed_seconds": time.monotonic() - started,
        "duration_target_seconds": args.duration_seconds,
        "completed_requests": completed,
        "failed_requests": failed,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "maximum_latency_seconds": max(latencies) if latencies else None,
        "passed": failed == 0 and completed > 0 and time.monotonic() - started >= args.duration_seconds,
    }
    atomic_json(args.state_dir / "RESULT.json", result)
    atomic_json(args.state_dir / "STATUS.json", {**result, "phase": "complete"})
    if not result["passed"]:
        raise SystemExit("sustained workload failed")


if __name__ == "__main__":
    main()
