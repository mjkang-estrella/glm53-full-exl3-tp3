#!/usr/bin/env python3
"""Summarize copied per-rank candidate watchdog CSVs atomically."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path: Path, value: object) -> None:
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


def numeric(row: dict[str, str], key: str, kind=float):
    value = row[key]
    try:
        return kind(value)
    except ValueError:
        return None


def summarize(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"watchdog metrics are empty: {path}")
    available = [numeric(row, "host_available_bytes", int) for row in rows]
    cgroup_memory = [numeric(row, "cgroup_memory_bytes", int) for row in rows]
    cgroup_swap = [numeric(row, "cgroup_swap_bytes", int) for row in rows]
    temperature = [numeric(row, "temperature_c") for row in rows]
    power = [numeric(row, "power_w") for row in rows]
    pswpin = [numeric(row, "pswpin", int) for row in rows]
    pswpout = [numeric(row, "pswpout", int) for row in rows]
    required = available + cgroup_memory + cgroup_swap + pswpin + pswpout
    if any(value is None for value in required):
        raise ValueError(f"watchdog metrics contain an invalid required field: {path}")
    valid_temperature = [value for value in temperature if value is not None]
    valid_power = [value for value in power if value is not None]
    minimum = min(available)
    return {
        "source": str(path),
        "samples": len(rows),
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "min_host_available_bytes": minimum,
        "min_host_available_gib": minimum / (1 << 30),
        "max_cgroup_memory_bytes": max(cgroup_memory),
        "max_cgroup_swap_bytes": max(cgroup_swap),
        "max_temperature_c": max(valid_temperature) if valid_temperature else None,
        "max_power_w": max(valid_power) if valid_power else None,
        "pswpin_delta": pswpin[-1] - pswpin[0],
        "pswpout_delta": pswpout[-1] - pswpout[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = {
        str(rank): summarize(args.attempt_dir / "ranks" / f"rank-{rank}.metrics.csv")
        for rank in range(3)
    }
    reserve = 12 * (1 << 30)
    result = {
        "schema": "glm53-full-exl3-tp3.candidate-memory-summary.v2",
        "attempt": args.attempt,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "reserve_bytes": reserve,
        "ranks": ranks,
        "passed": all(
            row["min_host_available_bytes"] >= reserve
            and row["max_cgroup_swap_bytes"] == 0
            for row in ranks.values()
        ),
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
