#!/usr/bin/env python3
"""Verify full-model K3 load counts, every rank geometry line, and fatal logs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tp3k3.geometry import GEOMETRY_ID, rank_geometry


GEOMETRY = re.compile(
    r"GLM53_EXL3_TP3_LAYER_GEOMETRY_OK layer=(?P<layer>[0-9]+) "
    r"rank=(?P<rank>[0-2]) offset=(?P<offset>[0-9]+) width=(?P<width>[0-9]+) padding=0"
)
LOAD = re.compile(
    r"GLM53_FULL_K3_STREAM_LOAD_OK seen=700416 mtp_skipped=9216 rank_selected=230400"
)
FATAL = re.compile(
    r"(?:Traceback \(most recent call last\)|EngineDeadError|CUDA out of memory|"
    r"NVRM|Xid|oom-kill|kernel OOM|I/O error|filesystem error|NCCL[^\n]*(?:abort|error))",
    re.IGNORECASE,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks = []
    errors: list[str] = []
    for rank in range(3):
        path = args.logs / f"rank-{rank}.log"
        text = path.read_text(encoding="utf-8", errors="replace")
        rows = []
        for match in GEOMETRY.finditer(text):
            row = {key: int(value) for key, value in match.groupdict().items()}
            rows.append(row)
        by_layer = {row["layer"]: row for row in rows}
        geometry_errors = []
        if len(rows) != 75 or set(by_layer) != set(range(3, 78)):
            geometry_errors.append(f"geometry rows/layers differ: rows={len(rows)} unique={len(by_layer)}")
        for layer, row in by_layer.items():
            expected_offset, expected_width = rank_geometry(layer, rank)
            if row != {"layer": layer, "rank": rank, "offset": expected_offset, "width": expected_width}:
                geometry_errors.append(f"layer {layer} geometry differs: {row}")
        load_lines = len(LOAD.findall(text))
        fatal_lines = [line for line in text.splitlines() if FATAL.search(line)][-20:]
        if load_lines != 1:
            geometry_errors.append(f"full streaming load marker count differs: {load_lines}")
        if "NCCL INFO Using network IB" not in text:
            geometry_errors.append("NCCL IB transport marker missing")
        if "GLM53_FULL_TP3_PADDING_APPLIED attention=64->66 vocab=154880->154944 shared_expert=2048->2304" not in text:
            geometry_errors.append("full-model padding marker missing")
        if fatal_lines:
            geometry_errors.append("fatal log lines present")
        if geometry_errors:
            errors.extend(f"rank {rank}: {value}" for value in geometry_errors)
        ranks.append({
            "rank": rank,
            "log": str(path),
            "geometry_lines": len(rows),
            "unique_layers": len(by_layer),
            "stream_load_markers": load_lines,
            "fatal_lines": fatal_lines,
            "errors": geometry_errors,
            "passed": not geometry_errors,
        })
    result = {
        "schema": "glm53-full-exl3-tp3.candidate-log-verification.v1",
        "geometry_id": GEOMETRY_ID,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "ranks": ranks,
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(args.output, result)
    if errors:
        raise SystemExit("candidate log verification failed: " + "; ".join(errors[:5]))


if __name__ == "__main__":
    main()
