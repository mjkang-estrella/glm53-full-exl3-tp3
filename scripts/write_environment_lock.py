#!/usr/bin/env python3
"""Hash the project implementation and bind it to the inventoried runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--rollback", type=Path, required=True)
    parser.add_argument("--source-structure", type=Path, required=True)
    parser.add_argument("--tensor-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    relative_files = [Path(name) for name in ("AGENTS.md", "PLAN.md", "TASK_PROMPT.md", "README.md")]
    for directory in ("tp3k3", "runtime", "scripts", "tests"):
        relative_files.extend(
            path.relative_to(project)
            for path in sorted((project / directory).glob("*"))
            if path.is_file() and path.suffix in {".py", ".sh"}
        )
    implementation = {
        str(relative): {"bytes": (project / relative).stat().st_size, "sha256": sha256(project / relative)}
        for relative in sorted(set(relative_files))
    }
    spark_images = {}
    summary_dir = args.rollback / "mj-zima" / "spark-summaries"
    for path in sorted(summary_dir.glob("mj-spark-*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        spark_images[summary["host"]] = [
            {"name": row["name"], "image_ref": row["image_ref"], "image_id": row["image_id"]}
            for row in summary["containers"]
        ]
    numeric_core = project / "encoder-r10" / "lineage" / "encode_tr3_v31.py"
    result = {
        "schema": "glm53-full-exl3-tp3.environment-lock.v1",
        "project": str(project),
        "source_revision": "304b8051cfb2b260b61ce0cbe330e02a98e73639",
        "source_structure": {"path": str(args.source_structure), "sha256": sha256(args.source_structure)},
        "tensor_plan": {"path": str(args.tensor_plan), "sha256": sha256(args.tensor_plan)},
        "rollback_inventory": str(args.rollback),
        "spark_images": spark_images,
        "sealed_components": {
            "numeric_core": {"path": str(numeric_core), "sha256": sha256(numeric_core)},
            "r10_source_bundle": {"path": str(project / "encoder-r10"), "verification": "encoder-r10/verify_bundle.py"},
            "exl3_extension_sha256": "7bba0fe1cb7f018bc188cc1df558b6ed5d329c7178093e0d71da7a8d731907d2",
            "exl3_runtime_overlay_sha256": "c9e765e13747cde82840c7af44945b7f06a1dee176df472dcebd1d858f9a5843",
        },
        "implementation_files": implementation,
    }
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
