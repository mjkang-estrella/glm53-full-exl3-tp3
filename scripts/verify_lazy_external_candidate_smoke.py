#!/usr/bin/env python3
"""Verify full-model lazy-K3 external-launcher generation and safety."""

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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.state / "CONFIG.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    expected_context = int(config.get("max_model_len", 4096))
    expected_reserve = int(config.get("resident_min_available_bytes", 12 * GIB))
    expected_execution = config.get("execution")
    errors: list[str] = []
    ranks: list[dict] = []
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
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = list(csv.DictReader(handle))
        log = log_path.read_text(encoding="utf-8", errors="replace")
        minimum = min((int(row["host_available_bytes"]) for row in metrics), default=0)
        text = str(result.get("text", ""))
        if not result.get("passed") or "paris" not in text.lower():
            errors.append(f"rank {rank}: coherent Paris completion missing")
        if result.get("profile", {}).get("max_model_len") != expected_context:
            errors.append(
                f"rank {rank}: context {result.get('profile', {}).get('max_model_len')} "
                f"differs from configured {expected_context}"
            )
        if inspected["State"]["ExitCode"] != 0 or inspected["State"]["OOMKilled"]:
            errors.append(f"rank {rank}: container exit/OOM state differs")
        if minimum < expected_reserve:
            errors.append(
                f"rank {rank}: minimum host reserve {minimum} below configured "
                f"{expected_reserve} bytes"
            )
        for marker in (
            "GLM53_EXL3_LAZY_K3_PATCH_INSTALLED",
            "GLM53_TP3_SPARSE_MLA_HEAD_ADAPTER_APPLIED",
            "GLM53_TP3_SPARSE_MLA_SHADOW_OK",
            "GLM53_LAZY_K3_PREMATERIALIZATION_FILTER_OK",
            "GLM53_EXL3_LAZY_K3_LAYER_READY layer=77",
            "GLM53_EXTERNAL_SMOKE_GENERATED",
            "GLM53_LAZY_K3_STATS",
            "NCCL INFO Using network IB",
            "NCCL INFO 1 coll channels",
        ):
            if marker not in log:
                errors.append(f"rank {rank}: log marker missing: {marker}")
        ranks.append(
            {
                "rank": rank,
                "host": result.get("host"),
                "text": text,
                "token_ids": result.get("token_ids"),
                "finish_reason": result.get("finish_reason"),
                "load_and_generate_seconds": result.get("load_and_generate_seconds"),
                "generation_seconds": result.get("generation_seconds"),
                "minimum_host_available_bytes": minimum,
                "available_after_load_bytes": result.get("host_available_after_load_bytes"),
                "available_after_generate_bytes": result.get("host_available_after_generate_bytes"),
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
        "schema": "glm53-full-exl3-tp3.lazy-external-smoke-summary.v1",
        "passed": not errors,
        "errors": errors,
        "config": {
            "execution": expected_execution,
            "max_model_len": expected_context,
            "resident_min_available_bytes": expected_reserve,
        },
        "ranks": ranks,
    }
    atomic_json(args.output, summary)
    if errors:
        raise SystemExit("lazy external candidate failed: " + "; ".join(errors[:8]))


if __name__ == "__main__":
    main()
