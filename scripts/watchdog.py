#!/usr/bin/env python3
"""Host-side encoder watchdog for memory, swap, GPU, kernel, and filesystem faults."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


GIB = 1 << 30
FAULT = re.compile(r"(?:NVRM|Xid|out of memory|oom-kill|I/O error|filesystem error|EXT4-fs error|BTRFS.*error)", re.I)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def meminfo() -> dict[str, int]:
    wanted = {"MemAvailable", "Cached", "AnonPages", "Unevictable", "Mlocked"}
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name = line.split(":", 1)[0]
        if name in wanted:
            values[name] = int(line.split()[1]) * 1024
    if "MemAvailable" not in values:
        raise RuntimeError("MemAvailable missing")
    return values


def swap_pages() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        name, raw = line.split()
        if name in ("pswpin", "pswpout"):
            values[name] = int(raw)
    return values.get("pswpin", 0), values.get("pswpout", 0)


def running(container: str) -> bool:
    result = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container], text=True, capture_output=True)
    return result.returncode == 0 and result.stdout.strip() == "true"


def cgroup_memory(container: str) -> dict[str, int]:
    """Return container aggregate memory without walking large process smaps."""

    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", container],
        text=True,
        capture_output=True,
    )
    if result.returncode or not result.stdout.strip().isdigit():
        return {}
    pid = int(result.stdout.strip())
    if pid <= 0:
        return {}
    cgroup_line = next(
        (
            line
            for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if cgroup_line is None:
        return {}
    root = Path("/sys/fs/cgroup") / cgroup_line.split("::", 1)[1].lstrip("/")

    def scalar(name: str) -> int:
        path = root / name
        return int(path.read_text().strip()) if path.is_file() else 0

    statistics = {}
    stat_path = root / "memory.stat"
    if stat_path.is_file():
        statistics = {
            name: int(value)
            for name, value in (
                line.split() for line in stat_path.read_text().splitlines()
            )
        }
    return {
        "current": scalar("memory.current"),
        "swap_current": scalar("memory.swap.current"),
        "anon": statistics.get("anon", 0),
        "file": statistics.get("file", 0),
        "pagetables": statistics.get("pagetables", 0),
    }


def kernel_faults(since: str) -> tuple[list[str], str | None]:
    result = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager", "-o", "cat"], text=True, capture_output=True)
    if result.returncode:
        return [], result.stderr.strip() or f"journalctl exit {result.returncode}"
    return [line for line in result.stdout.splitlines() if FAULT.search(line)][-20:], None


def gpu_row() -> list[str]:
    query = "timestamp,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if result.returncode:
        return [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "ERROR", result.stderr.strip(), "", "", ""]
    return [part.strip() for part in result.stdout.splitlines()[0].split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--reserve-gib", type=int, default=12)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.state_dir / "metrics.csv"
    # journalctl otherwise interprets a timezone-free UTC string in each
    # Spark's local timezone. The nodes are not configured identically.
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    initial_swap = swap_pages()
    previous_swap = initial_swap
    low_memory_samples = 0
    swap_thrash_samples = 0
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if metrics_path.stat().st_size == 0:
            writer.writerow([
                "timestamp", "temperature_c", "power_w", "graphics_clock_mhz",
                "memory_clock_mhz", "reported_memory_mib", "host_available_bytes",
                "host_cached_bytes", "host_anon_bytes", "host_unevictable_bytes",
                "host_mlocked_bytes", "cgroup_memory_bytes", "cgroup_swap_bytes",
                "cgroup_anon_bytes", "cgroup_file_bytes", "cgroup_pagetables_bytes",
                "pswpin", "pswpout",
            ])
        while running(args.container):
            memory = meminfo()
            available = memory["MemAvailable"]
            cgroup = cgroup_memory(args.container)
            swap = swap_pages()
            row = gpu_row()
            writer.writerow(row + [
                available,
                memory.get("Cached", 0),
                memory.get("AnonPages", 0),
                memory.get("Unevictable", 0),
                memory.get("Mlocked", 0),
                cgroup.get("current", 0),
                cgroup.get("swap_current", 0),
                cgroup.get("anon", 0),
                cgroup.get("file", 0),
                cgroup.get("pagetables", 0),
                swap[0],
                swap[1],
            ])
            handle.flush()
            os.fsync(handle.fileno())
            low_memory_samples = low_memory_samples + 1 if available < args.reserve_gib * GIB else 0
            interval_previous_swap = previous_swap
            delta_pages = (swap[0] - previous_swap[0]) + (swap[1] - previous_swap[1])
            # Three consecutive intervals at >=64 MiB swap I/O each is sustained thrashing.
            swap_thrash_samples = swap_thrash_samples + 1 if delta_pages * os.sysconf("SC_PAGE_SIZE") >= 64 << 20 else 0
            previous_swap = swap
            faults, journal_error = kernel_faults(started)
            reason = None
            evidence = None
            if journal_error:
                reason, evidence = "kernel_audit_unavailable", {"error": journal_error}
            elif low_memory_samples >= 3:
                reason, evidence = "host_memory_reserve", {"available_bytes": available, "reserve_bytes": args.reserve_gib * GIB}
            elif swap_thrash_samples >= 3:
                reason, evidence = "sustained_swap_thrash", {
                    "initial_swap_pages": initial_swap,
                    "previous_swap_pages": interval_previous_swap,
                    "current_swap_pages": swap,
                    "last_interval_delta_pages": delta_pages,
                    "consecutive_samples": swap_thrash_samples,
                    "page_size": os.sysconf("SC_PAGE_SIZE"),
                    "host_memory": memory,
                    "cgroup_memory": cgroup,
                }
            elif faults:
                reason, evidence = "kernel_fault", {"lines": faults}
            if reason:
                atomic_json(args.state_dir / "STOP.json", {"schema": "glm53-full-exl3-tp3.watchdog-stop.v1", "container": args.container, "reason": reason, "evidence": evidence, "time": time.time()})
                subprocess.run(["docker", "stop", "--time", "30", args.container], check=False)
                raise SystemExit(2)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
