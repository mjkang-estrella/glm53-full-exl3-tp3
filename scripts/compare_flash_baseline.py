#!/usr/bin/env python3
"""Compare sealed Flash canaries with matching K3 candidate receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--core", required=True, type=Path)
    parser.add_argument("--extended-v1", required=True, type=Path)
    parser.add_argument("--extended-v2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = load(args.baseline)
    core = load(args.core)
    v1 = load(args.extended_v1)
    v2 = load(args.extended_v2)
    baseline_cases = {row["name"]: row for row in baseline["cases"]}
    candidate = {
        "korean_unicode": (core["probes"]["korean"], core["checks"]["korean"]),
        "factual_math": (core["probes"]["arithmetic"], core["checks"]["arithmetic"]),
        "coding": (core["probes"]["code"], core["checks"]["code"]),
        "reasoning_separation": (v2["probes"]["reasoning"], v2["checks"]["reasoning_answer"] and v2["checks"]["reasoning_trace"]),
        "tool_call": (v1["probes"]["tool_auto"], v1["checks"]["tool_auto"]),
    }
    comparisons = []
    for name, (row, passed) in candidate.items():
        flash = baseline_cases[name]
        candidate_seconds = float(row["elapsed_seconds"])
        flash_seconds = float(flash["elapsed_seconds"])
        comparisons.append({
            "name": name,
            "prompt_semantics_matched": True,
            "flash_passed": bool(flash["passed"]),
            "candidate_passed": bool(passed),
            "flash_elapsed_seconds": flash_seconds,
            "candidate_elapsed_seconds": candidate_seconds,
            "candidate_to_flash_latency_ratio": candidate_seconds / flash_seconds,
            "candidate_finish_reason": row["finish_reason"],
        })
    result = {
        "schema": "glm53-full-exl3-tp3.flash-baseline-comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "baseline_model": baseline["model"],
        "candidate_model": "GLM-5.3-K3-TP3-CANDIDATE",
        "common_cases": comparisons,
        "common_case_count": len(comparisons),
        "common_correctness_passed": all(row["flash_passed"] and row["candidate_passed"] for row in comparisons),
        "unmatched_baseline_cases": [{
            "name": "multimodal_text",
            "reason": "No matching candidate image fixture was authorized or preserved; not claimed.",
        }],
        "flash_was_restarted": False,
        "sources": {
            "baseline": str(args.baseline),
            "core": str(args.core),
            "extended_v1": str(args.extended_v1),
            "extended_v2": str(args.extended_v2),
        },
    }
    atomic_json(args.output, result)
    if not result["common_correctness_passed"]:
        raise SystemExit("candidate did not match baseline correctness on common probes")


if __name__ == "__main__":
    main()
