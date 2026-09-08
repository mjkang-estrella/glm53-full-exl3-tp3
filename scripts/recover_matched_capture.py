#!/usr/bin/env python3
"""Recover a sealed remote logit capture after a controller transfer failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.capture_matched_logits import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ssh-config", default="zima-ssh-config")
    parser.add_argument("--node", default="mj-spark-1")
    args = parser.parse_args()
    final = args.output_root / args.window_id
    incoming = args.output_root / f"{args.window_id}.incoming-{args.capture_id}"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    incoming.mkdir(parents=True, exist_ok=True)
    container = f"glm53-k3-cand-{args.attempt}-rank0"
    subprocess.run(
        [
            "ssh", "-F", args.ssh_config, args.node, "docker", "exec",
            container, "chmod", "-R", "a+rX", f"/capture/{args.capture_id}",
        ],
        check=True,
    )
    remote = f"{args.node}:{args.remote_root}/{args.capture_id}/"
    subprocess.run(
        ["rsync", "-a", "--partial", "-e", f"ssh -F {args.ssh_config}", remote, f"{incoming}/"],
        check=True,
    )
    receipt = json.loads((incoming / "CAPTURE_COMPLETE.json").read_text(encoding="utf-8"))
    errors = []
    if not (
        receipt.get("passed") is True
        and receipt.get("capture_id") == args.capture_id
        and receipt.get("window_id") == args.window_id
        and receipt.get("token_sha256") == sha256_file(args.tokens)
        and receipt.get("expected_rows") == 2047
        and receipt.get("expected_vocab") == 154880
    ):
        errors.append("capture receipt contract differs")
    for part in receipt.get("parts", []):
        path = incoming / part["path"]
        if not path.is_file() or path.stat().st_size != part["bytes"] or sha256_file(path) != part["sha256"]:
            errors.append(f"capture part differs: {part['path']}")
    if sum(int(part["rows"]) for part in receipt.get("parts", [])) != 2047:
        errors.append("capture row count differs")
    recovery = {
        "schema": "glm53-full-exl3-tp3.matched-capture-recovery.v1",
        "passed": not errors,
        "capture_id": args.capture_id,
        "window_id": args.window_id,
        "reason": "controller-rsync-json-mode-0600",
        "payload_recomputed": False,
        "checksum_validation": not errors,
        "api_row_ordering_check": "pending-response-replay",
        "errors": errors,
    }
    atomic_json(incoming / "RECOVERY.json", recovery)
    if errors:
        raise SystemExit("; ".join(errors))
    os.replace(incoming, final)
    print(json.dumps(recovery, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
