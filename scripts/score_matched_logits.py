#!/usr/bin/env python3
"""Full-position, full-vocabulary KL(BF16 || K3) from bounded raw captures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np


VOCAB = 154880


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_memmap(path: Path) -> np.memmap:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    spec = header["logits"]
    if spec["dtype"] != "F32" or spec["shape"] != [2047, VOCAB]:
        raise ValueError(f"teacher geometry differs: {spec}")
    start, end = spec["data_offsets"]
    if end - start != 2047 * VOCAB * 4:
        raise ValueError("teacher byte span differs")
    return np.memmap(path, mode="r", dtype="<f4", offset=8 + header_len + start, shape=(2047, VOCAB))


def candidate_parts(root: Path) -> tuple[dict, list[np.memmap]]:
    receipt = json.loads((root / "CAPTURE_COMPLETE.json").read_text(encoding="utf-8"))
    arrays = []
    for part in receipt["parts"]:
        path = root / part["path"]
        if path.stat().st_size != part["bytes"] or sha256_file(path) != part["sha256"]:
            raise ValueError(f"candidate capture checksum differs: {path}")
        arrays.append(np.memmap(path, mode="r", dtype="<f4", shape=(part["rows"], VOCAB)))
    if sum(array.shape[0] for array in arrays) != 2047:
        raise ValueError("candidate capture row count differs")
    return receipt, arrays


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


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--windows", default="confirmation-0000,confirmation-0001,confirmation-0002,confirmation-0003")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads(args.manifest.read_text(encoding="utf-8"))
    logit_rows = {row["window_id"]: row for row in aggregate["logit_files"]}
    results = []
    total_kld = 0.0
    total_agree = 0
    total_top5_overlap = 0.0
    total_positions = 0
    margin_bins = {
        "lt_0.1": {"count": 0, "agree": 0, "kld": 0.0},
        "0.1_to_0.5": {"count": 0, "agree": 0, "kld": 0.0},
        "0.5_to_1.0": {"count": 0, "agree": 0, "kld": 0.0},
        "ge_1.0": {"count": 0, "agree": 0, "kld": 0.0},
    }
    first_divergence = None
    for window_id in args.windows.split(","):
        reference_record = logit_rows[window_id]
        teacher_path = args.reference_root / reference_record["path"]
        if sha256_file(teacher_path) != reference_record["sha256"]:
            raise ValueError(f"teacher checksum differs for {window_id}")
        teacher = teacher_memmap(teacher_path)
        receipt, parts = candidate_parts(args.capture_root / window_id)
        position_rows = []
        part_index = part_row = 0
        for position in range(2047):
            if part_row == parts[part_index].shape[0]:
                part_index += 1
                part_row = 0
            candidate_logits = np.asarray(parts[part_index][part_row], dtype=np.float64)
            part_row += 1
            teacher_logits = np.asarray(teacher[position], dtype=np.float64)
            teacher_logz = float(np.logaddexp.reduce(teacher_logits))
            candidate_logz = float(np.logaddexp.reduce(candidate_logits))
            teacher_logp = teacher_logits - teacher_logz
            kld = float(np.sum(np.exp(teacher_logp) * (teacher_logp - (candidate_logits - candidate_logz)), dtype=np.float64))
            teacher_argmax = int(np.argmax(teacher_logits))
            candidate_argmax = int(np.argmax(candidate_logits))
            agree = teacher_argmax == candidate_argmax
            teacher_top5_indices = np.argpartition(teacher_logits, -5)[-5:]
            candidate_top5_indices = np.argpartition(candidate_logits, -5)[-5:]
            teacher_top2 = np.partition(teacher_logits[teacher_top5_indices], -2)[-2:]
            candidate_top2 = np.partition(candidate_logits[candidate_top5_indices], -2)[-2:]
            teacher_margin = float(teacher_top2.max() - teacher_top2.min())
            candidate_margin = float(candidate_top2.max() - candidate_top2.min())
            teacher_top5 = {int(value) for value in teacher_top5_indices}
            candidate_top5 = {int(value) for value in candidate_top5_indices}
            top5_overlap = len(teacher_top5 & candidate_top5) / 5
            if teacher_margin < 0.1:
                margin_bin = "lt_0.1"
            elif teacher_margin < 0.5:
                margin_bin = "0.1_to_0.5"
            elif teacher_margin < 1.0:
                margin_bin = "0.5_to_1.0"
            else:
                margin_bin = "ge_1.0"
            margin_bins[margin_bin]["count"] += 1
            margin_bins[margin_bin]["agree"] += int(agree)
            margin_bins[margin_bin]["kld"] += kld
            if not agree and first_divergence is None:
                first_divergence = {"window_id": window_id, "position": position, "teacher_argmax": teacher_argmax, "candidate_argmax": candidate_argmax}
            position_rows.append({"position": position, "kld_bf16_to_k3": kld, "teacher_argmax": teacher_argmax, "candidate_argmax": candidate_argmax, "argmax_agreement": agree, "teacher_top1_margin": teacher_margin, "candidate_top1_margin": candidate_margin, "top5_overlap_fraction": top5_overlap, "teacher_top5": sorted(teacher_top5), "candidate_top5": sorted(candidate_top5)})
            total_kld += kld
            total_agree += int(agree)
            total_top5_overlap += top5_overlap
            total_positions += 1
        results.append({
            "window_id": window_id,
            "reference_sha256": reference_record["sha256"],
            "capture_id": receipt["capture_id"],
            "positions": position_rows,
            "mean_kld_bf16_to_k3": sum(row["kld_bf16_to_k3"] for row in position_rows) / len(position_rows),
            "argmax_agreement": sum(row["argmax_agreement"] for row in position_rows) / len(position_rows),
            "mean_top5_overlap_fraction": sum(row["top5_overlap_fraction"] for row in position_rows) / len(position_rows),
            "domain": reference_record.get("domain"),
        })
    margin_summary = {
        name: {
            "positions": row["count"],
            "argmax_agreement": row["agree"] / row["count"] if row["count"] else None,
            "mean_kld_bf16_to_k3": row["kld"] / row["count"] if row["count"] else None,
        }
        for name, row in margin_bins.items()
    }
    result = {
        "schema": "glm53-full-exl3-tp3.full-panel-kld.v1",
        "passed": total_positions == 4 * 2047 and all(math.isfinite(row["kld_bf16_to_k3"]) and row["kld_bf16_to_k3"] >= -1e-10 for window in results for row in window["positions"]),
        "reference_scope": "supplemental-public-reference-nonfinal",
        "reference_repo": aggregate["repo_id"],
        "bf16_model_revision": aggregate["model_revision"],
        "windows": results,
        "window_count": len(results),
        "prediction_positions": total_positions,
        "vocab_size": VOCAB,
        "mean_kld_bf16_to_k3": total_kld / total_positions,
        "absolute_development_gate_nats": 0.030,
        "absolute_development_gate_passed": total_kld / total_positions <= 0.030,
        "published_uniform_k3_contextual_kld": 0.03754,
        "argmax_agreement": total_agree / total_positions,
        "mean_top5_overlap_fraction": total_top5_overlap / total_positions,
        "margin_conditioned": margin_summary,
        "first_divergence": first_divergence,
        "completed_at": utc(),
    }
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("matched full-panel scoring validation failed")


if __name__ == "__main__":
    main()
