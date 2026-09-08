#!/usr/bin/env python3
"""List the BF16 safetensor shards containing one routed-expert layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    if args.layer not in range(3, 79):
        parser.error("layer must be 3..78")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    prefix = f"model.layers.{args.layer}.mlp.experts."
    names = sorted(
        {
            shard
            for tensor, shard in index["weight_map"].items()
            if tensor.startswith(prefix)
            and tensor.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"))
        }
    )
    tensors = sum(
        tensor.startswith(prefix)
        and tensor.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"))
        for tensor in index["weight_map"]
    )
    if tensors != 256 * 3:
        raise SystemExit(f"layer {args.layer}: found {tensors} expert tensors, expected 768")
    for name in names:
        print(name)


if __name__ == "__main__":
    main()
