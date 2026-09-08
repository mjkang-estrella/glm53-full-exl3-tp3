#!/usr/bin/env python3
"""Arm rank-0 bounded capture, run an exact token window, and seal it to UNAS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

import numpy as np


MODEL = "GLM-5.3-K3-TP3-CANDIDATE"


def normalize_scheduler_singletons(receipt: dict) -> dict:
    """Remove vLLM sample-logit calls surrounding prompt-logprob batches."""

    parts = receipt.get("parts", [])
    rows = [int(part.get("rows", -1)) for part in parts]
    expected_rows = int(receipt.get("expected_rows", 0))
    if expected_rows < 2049 or sum(rows) != expected_rows:
        raise ValueError(f"scheduler logit call pattern differs: {rows}")
    selected = []
    selected_indices = []
    dropped_indices = []
    first_row = 0
    for index, part_value in enumerate(parts):
        rows_value = int(part_value.get("rows", -1))
        if rows_value == 1:
            dropped_indices.append(index)
            continue
        if rows_value <= 1:
            raise ValueError(f"scheduler non-singleton part has invalid rows: {rows}")
        part = dict(part_value)
        part["first_row"] = first_row
        first_row += rows_value
        selected_indices.append(index)
        selected.append(part)
    if first_row != 2047:
        raise ValueError("normalized prompt-logit row count differs")
    normalized = dict(receipt)
    normalized.update(
        schema="glm53-full-exl3-tp3.logit-capture-complete.normalized-v2",
        expected_rows=2047,
        bytes=sum(int(part["bytes"]) for part in selected),
        parts=selected,
        scheduler_normalization={
            "source_expected_rows": expected_rows,
            "source_part_rows": rows,
            "selected_part_indices": selected_indices,
            "dropped_singleton_part_indices": dropped_indices,
            "basis": "vLLM sample-logit calls precede prompt-logprob batches",
        },
    )
    return normalized


def remote_capture_root(stamp: str, attempt: str) -> str:
    safe = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
    if not safe.fullmatch(stamp) or not safe.fullmatch(attempt):
        raise ValueError("unsafe real-test stamp or candidate attempt")
    return (
        "/home/mj-kang/Dev/state/glm53-full-exl3-tp3/real-test/"
        f"{stamp}/candidate/attempts/{attempt}/rank-0/capture"
    )


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
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


def run(*command: str, input_data: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, input=input_data, check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.0.238:8893")
    parser.add_argument("--ssh-config", default="zima-ssh-config")
    parser.add_argument("--node", default="mj-spark-1")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--scheduler-singletons", action="store_true")
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    tokens = np.load(args.tokens, mmap_mode="r")
    if tokens.shape != (2048,):
        raise ValueError(f"token window geometry differs: {tokens.shape}")
    token_sha = sha256_file(args.tokens)
    capture_id = f"{args.window_id}-{int(time.time())}"
    remote_root = remote_capture_root(args.stamp, args.attempt)
    scheduler_expected_rows = (
        int(args.expected_rows)
        if args.expected_rows is not None
        else (2049 if args.scheduler_singletons else 2047)
    )
    if scheduler_expected_rows < (2049 if args.scheduler_singletons else 2047) or scheduler_expected_rows > 32768:
        raise ValueError(f"invalid expected row bound: {scheduler_expected_rows}")
    arm = {
        "schema": "glm53-full-exl3-tp3.logit-capture-arm.v1",
        "capture_id": capture_id,
        "window_id": args.window_id,
        "token_sha256": token_sha,
        "model": MODEL,
        "expected_rows": scheduler_expected_rows,
        "expected_vocab": 154880,
    }
    arm_raw = (json.dumps(arm, sort_keys=True, indent=2) + "\n").encode()
    run("ssh", "-F", args.ssh_config, args.node, "test", "!", "-e", f"{remote_root}/ARM.json")
    # stdin avoids fragile shell quoting; rename makes the arm atomic.
    run("ssh", "-F", args.ssh_config, args.node, "sh", "-c", f"'umask 022; tee {remote_root}/.ARM.tmp >/dev/null && mv {remote_root}/.ARM.tmp {remote_root}/ARM.json'", input_data=arm_raw)
    payload = {
        "model": MODEL,
        "prompt": [int(value) for value in tokens],
        "add_special_tokens": False,
        "echo": True,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 1,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
        "ignore_eos": True,
        "seed": 20260906,
    }
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        api_response = json.load(response)
    provisional_response = args.output_root / f".{args.window_id}.{capture_id}.api-response.json"
    atomic_json(
        provisional_response,
        {
            "schema": "glm53-full-exl3-tp3.matched-logit-provisional-response.v1",
            "capture_id": capture_id,
            "window_id": args.window_id,
            "started_at_unix": started,
            "completed_at_unix": time.time(),
            "parameters": payload,
            "response": api_response,
        },
    )
    deadline = time.time() + args.timeout
    remote_final = f"{remote_root}/{capture_id}"
    while time.time() < deadline:
        result = subprocess.run(
            ["ssh", "-F", args.ssh_config, args.node, "test", "-s", f"{remote_final}/CAPTURE_COMPLETE.json"],
            check=False,
        )
        if result.returncode == 0:
            break
        time.sleep(2)
    else:
        raise TimeoutError("rank-0 raw-logit receipt did not appear")
    # Candidate containers run as root while the immutable host mount belongs
    # to the operator. Make only this completed capture readable before rsync.
    container = f"glm53-k3-cand-{args.attempt}-rank0"
    run(
        "ssh", "-F", args.ssh_config, args.node,
        "docker", "exec", container, "chmod", "-R", "a+rX",
        f"/capture/{capture_id}",
    )
    final = args.output_root / args.window_id
    incoming = args.output_root / f"{args.window_id}.incoming-{capture_id}"
    if final.exists() or incoming.exists():
        raise FileExistsError(f"refusing to overwrite matched-logit evidence for {args.window_id}")
    incoming.mkdir(parents=True)
    subprocess.run(
        ["rsync", "-a", "--partial", "-e", f"ssh -F {args.ssh_config}", f"{args.node}:{remote_final}/", f"{incoming}/"],
        check=True,
    )
    receipt = json.loads((incoming / "CAPTURE_COMPLETE.json").read_text(encoding="utf-8"))
    errors = []
    raw_expected_rows = scheduler_expected_rows
    if not (
        receipt.get("passed") is True
        and receipt.get("capture_id") == capture_id
        and receipt.get("window_id") == args.window_id
        and receipt.get("token_sha256") == token_sha
        and receipt.get("expected_rows") == raw_expected_rows
        and receipt.get("expected_vocab") == 154880
    ):
        errors.append("capture receipt contract differs")
    for part in receipt.get("parts", []):
        path = incoming / part["path"]
        if not path.is_file() or path.stat().st_size != part["bytes"] or sha256_file(path) != part["sha256"]:
            errors.append(f"captured part differs: {part['path']}")
    if sum(int(part["rows"]) for part in receipt.get("parts", [])) != raw_expected_rows:
        errors.append("captured row count differs")
    if args.scheduler_singletons and not errors:
        raw_receipt = receipt
        try:
            receipt = normalize_scheduler_singletons(raw_receipt)
        except ValueError as error:
            errors.append(str(error))
        else:
            atomic_json(incoming / "SCHEDULER_CAPTURE_COMPLETE.json", raw_receipt)
            atomic_json(incoming / "CAPTURE_COMPLETE.json", receipt)
            atomic_json(
                incoming / "NORMALIZATION.json",
                {
                    "schema": "glm53-full-exl3-tp3.logit-capture-normalization.v1",
                    "passed": True,
                    "capture_id": capture_id,
                    **receipt["scheduler_normalization"],
                },
            )
    ordering_checks = []
    api_logprobs = (((api_response.get("choices") or [{}])[0].get("logprobs") or {}).get("token_logprobs") or [])
    if len(api_logprobs) < 2048:
        errors.append(f"echoed prompt logprobs missing: {len(api_logprobs)}")
    else:
        arrays = [
            np.memmap(incoming / part["path"], mode="r", dtype="<f4", shape=(part["rows"], 154880))
            for part in receipt["parts"]
        ]
        for position in (0, 1023, 2046):
            relative = position
            for array in arrays:
                if relative < array.shape[0]:
                    row = np.asarray(array[relative], dtype=np.float64)
                    break
                relative -= array.shape[0]
            else:
                raise AssertionError("captured row lookup failed")
            target = int(tokens[position + 1])
            raw_logprob = float(row[target] - np.logaddexp.reduce(row))
            api_logprob = float(api_logprobs[position + 1])
            delta = abs(raw_logprob - api_logprob)
            ordering_checks.append({"position": position, "target_token": target, "raw_logprob": raw_logprob, "api_logprob": api_logprob, "absolute_delta": delta})
            if delta > 5e-3:
                errors.append(f"captured row/API ordering differs at {position}: {delta}")
    if errors:
        atomic_json(incoming / "TRANSFER_FAILED.json", {"errors": errors, "capture_id": capture_id})
        raise RuntimeError("; ".join(errors))
    atomic_json(
        incoming / "REQUEST.json",
        {
            "schema": "glm53-full-exl3-tp3.matched-logit-request.v1",
            "capture_id": capture_id,
            "window_id": args.window_id,
            "started_at_unix": started,
            "completed_at_unix": time.time(),
            "parameters": payload,
            "response": api_response,
            "row_ordering_checks": ordering_checks,
            "scheduler_singletons_normalized": args.scheduler_singletons,
        },
    )
    os.replace(incoming, final)
    provisional_response.unlink()
    print(json.dumps({"passed": True, "window_id": args.window_id, "capture": str(final), "receipt": receipt}, indent=2))


if __name__ == "__main__":
    main()
