#!/usr/bin/env python3
"""Create a hardlink-based one-layer mixed checkpoint smoke target."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

GEOMETRY_ID = "rotating-uneven-768-640-640-v1"


def atomic_json(path: Path, value: object) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
    os.replace(temporary, path)


def clone_links(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for root, _dirs, files in os.walk(source):
        rel = Path(root).relative_to(source)
        dest = target / rel
        dest.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = Path(root) / name
            dst = dest / name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--k2-layer", type=Path, required=True)
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    selection = json.loads(args.experts.read_text(encoding="utf-8"))
    selected = [int(x) for x in selection["layers"]["3"]["k2_experts"]]
    if len(selected) != 64:
        raise SystemExit("layer-3 selection is not 64 experts")
    staging = args.output.with_name(args.output.name + ".incoming")
    if staging.exists():
        raise SystemExit(f"staging exists: {staging}")
    clone_links(args.base, staging)
    try:
        for expert in selected:
            src = args.k2_layer / "experts" / f"expert-{expert:03d}.safetensors"
            dst = staging / f"k3-layer-003-expert-{expert:03d}.safetensors"
            temporary = dst.with_name(f".{dst.name}.k2.tmp")
            shutil.copy2(src, temporary)
            os.replace(temporary, dst)
        bits = {
            "schema": "glm53-full-exl3-tp3.k275-expert-bits.v1",
            "geometry_id": GEOMETRY_ID,
            "target_bpw": "2.75-partial-layer3",
            "bits": {str(layer): {"k2": selected if layer == 3 else [], "k3": [e for e in range(256) if not (layer == 3 and e in selected)]} for layer in range(3, 79)},
        }
        atomic_json(staging / "expert-bits.json", bits)
        quant = json.loads((staging / "quantization_config.json").read_text(encoding="utf-8"))
        quant.update({"bits": 3, "target_bpw": "2.75-partial-layer3", "mixed_expert_bits": True, "k2_experts_per_layer": 64, "k3_experts_per_layer": 192, "expert_bits_file": "expert-bits.json", "serving_reader_qualified": False})
        atomic_json(staging / "quantization_config.json", quant)
        atomic_json(staging / "MIXED_PARTIAL_TEST.json", {"schema": "glm53-full-exl3-tp3.k275-partial-test.v1", "passed": True, "geometry_id": GEOMETRY_ID, "mixed_layer": 3, "k2_experts": selected, "other_layers": "sealed K3 base"})
        os.replace(staging, args.output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"output": str(args.output), "k2_experts": len(selected), "passed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
