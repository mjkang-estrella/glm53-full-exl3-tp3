#!/usr/bin/env python3
"""Verify a three-rank external-launcher candidate smoke and its safety data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile


GIB = 1 << 30


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    ranks = []
    for rank in range(3):
        result_path = args.state / "ranks" / f"rank-{rank}.json"
        inspect_path = args.state / "ranks" / f"rank-{rank}.inspect.json"
        metrics_path = args.state / "ranks" / f"rank-{rank}.metrics.csv"
        log_path = args.logs / f"rank-{rank}.log"
        if not all(path.is_file() for path in (result_path, inspect_path, metrics_path, log_path)):
            errors.append(f"rank {rank}: evidence file missing")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        inspected = json.loads(inspect_path.read_text(encoding="utf-8"))[0]
        metrics = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        log = log_path.read_text(encoding="utf-8", errors="replace")
        minimum = min((int(row["host_available_bytes"]) for row in metrics), default=0)
        if not result.get("passed"):
            errors.append(f"rank {rank}: result did not pass")
        if inspected["State"]["ExitCode"] != 0 or inspected["State"]["OOMKilled"]:
            errors.append(f"rank {rank}: container exit/OOM state differs")
        if minimum < 12 * GIB:
            errors.append(f"rank {rank}: minimum host reserve {minimum} is below 12 GiB")
        for marker in (
            "GLM53_FULL_K3_STREAM_LOAD_OK",
            "GLM53_EXTERNAL_SMOKE_GENERATED",
            "NCCL INFO Using network IB",
            "NCCL INFO 1 coll channels",
        ):
            if marker not in log:
                errors.append(f"rank {rank}: log marker missing: {marker}")
        ranks.append(
            {
                "rank": rank,
                "host": result.get("host"),
                "minimum_host_available_bytes": minimum,
                "text": result.get("text"),
                "token_ids": result.get("token_ids"),
                "finish_reason": result.get("finish_reason"),
                "load_and_generate_seconds": result.get("load_and_generate_seconds"),
                "generation_seconds": result.get("generation_seconds"),
            }
        )
    if len(ranks) == 3:
        reference = (ranks[0]["text"], ranks[0]["token_ids"], ranks[0]["finish_reason"])
        if any((row["text"], row["token_ids"], row["finish_reason"]) != reference for row in ranks[1:]):
            errors.append("rank outputs differ")
    audit_path = args.state / "kernel-audit.json"
    if not audit_path.is_file() or not json.loads(audit_path.read_text(encoding="utf-8")).get("passed"):
        errors.append("strict kernel audit missing or failed")
    summary = {
        "schema": "glm53-full-exl3-tp3.external-candidate-smoke-summary.v1",
        "passed": not errors,
        "errors": errors,
        "ranks": ranks,
    }
    atomic_json(args.output, summary)
    if errors:
        raise SystemExit("external candidate smoke failed: " + "; ".join(errors[:5]))


if __name__ == "__main__":
    main()
