#!/usr/bin/env python3
"""Collect strict or lifecycle-classified cross-Spark kernel evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tp3k3.lifecycle import FAULT, evaluate_lifecycle, policy_document


NODES = ("mj-spark-1", "mj-spark-2", "mj-spark-3")


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


def collect(since: str, until: str | None, ssh_config: Path) -> dict[str, list[dict[str, str]]]:
    hosts: dict[str, list[dict[str, str]]] = {}
    remote_since = since.replace(" ", "T")
    remote_until = until.replace(" ", "T") if until else None
    for node in NODES:
        journal_command = ["journalctl", "-k", f"--since={remote_since}"]
        if remote_until:
            journal_command.append(f"--until={remote_until}")
        journal_command.extend(["--no-pager", "-o", "short-iso-precise"])
        result = subprocess.run(
            ["ssh", "-F", str(ssh_config), node, *journal_command],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise SystemExit(
                f"kernel journal query failed on {node}: {result.stderr.strip()}"
            )
        rows = []
        for line in result.stdout.splitlines():
            if not FAULT.search(line):
                continue
            timestamp = line.split(maxsplit=1)[0] if line.split() else ""
            rows.append({"timestamp": timestamp, "line": line})
        hosts[node] = rows
    return hosts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--service-state", type=Path)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--write-policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ssh-config", type=Path, default=Path("zima-ssh-config"))
    args = parser.parse_args()
    if args.write_policy:
        atomic_json(args.write_policy, policy_document())
        return
    if args.output is None:
        parser.error("--output is required for an audit")

    lifecycle_paths = (args.timeline, args.service_state, args.generation)
    if any(lifecycle_paths) and not all(lifecycle_paths):
        parser.error("--timeline, --service-state, and --generation are required together")
    if all(lifecycle_paths):
        timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
        service_state = json.loads(args.service_state.read_text(encoding="utf-8"))
        generation = json.loads(args.generation.read_text(encoding="utf-8"))
        try:
            since = timeline["candidate_runtime"]["start"]
            until = timeline["post_ready"]["end"]
        except (KeyError, TypeError):
            since = args.since
            until = args.until
        if not since or not until:
            parser.error("lifecycle timeline must supply candidate start and post-ready end")
        events = collect(since, until, args.ssh_config)
        value = evaluate_lifecycle(
            events_by_host=events,
            timeline=timeline,
            service_state=service_state,
            generation=generation,
        )
        value["source"] = {
            "journal_since": since,
            "journal_until": until,
            "ssh_config": str(args.ssh_config),
        }
    else:
        if not args.since:
            parser.error("--since is required for a strict audit")
        events = collect(args.since, args.until, args.ssh_config)
        hosts = {
            host: {
                "fault_count": len(rows),
                "fault_lines": [row["line"] for row in rows],
            }
            for host, rows in events.items()
        }
        value = {
            "schema": "glm53-full-exl3-tp3.kernel-audit.v1",
            "policy": "strict-zero-fault",
            "since": args.since,
            "until": args.until,
            "hosts": hosts,
            "passed": all(row["fault_count"] == 0 for row in hosts.values()),
        }
    atomic_json(args.output, value)
    if not value["passed"]:
        raise SystemExit("kernel fault audit found events")


if __name__ == "__main__":
    main()
