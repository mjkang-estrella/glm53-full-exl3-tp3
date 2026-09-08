#!/usr/bin/env python3
"""Replay one deterministic token window to verify recovered raw-logit row order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

from capture_matched_logits import MODEL, atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://192.168.0.238:8893")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    tokens = np.load(args.tokens, mmap_mode="r")
    if tokens.shape != (2048,):
        raise ValueError("token window geometry differs")
    receipt = json.loads((args.capture / "CAPTURE_COMPLETE.json").read_text(encoding="utf-8"))
    if not (
        receipt.get("passed") is True
        and receipt.get("window_id") == args.window_id
        and receipt.get("token_sha256") == sha256_file(args.tokens)
    ):
        raise ValueError("capture/token identity differs")
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
    api_logprobs = (((api_response.get("choices") or [{}])[0].get("logprobs") or {}).get("token_logprobs") or [])
    errors = []
    if len(api_logprobs) < 2048:
        errors.append(f"echoed prompt logprobs missing: {len(api_logprobs)}")
    arrays = [
        np.memmap(
            args.capture / part["path"], mode="r", dtype="<f4",
            shape=(part["rows"], 154880),
        )
        for part in receipt["parts"]
    ]
    checks = []
    if not errors:
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
            checks.append(
                {
                    "position": position,
                    "target_token": target,
                    "raw_logprob": raw_logprob,
                    "api_logprob": api_logprob,
                    "absolute_delta": delta,
                }
            )
            if delta > 5e-3:
                errors.append(f"row/API ordering differs at {position}: {delta}")
    result = {
        "schema": "glm53-full-exl3-tp3.matched-capture-ordering.v1",
        "passed": not errors,
        "window_id": args.window_id,
        "capture_id": receipt["capture_id"],
        "payload_recomputed": False,
        "request_replayed": True,
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "parameters": payload,
        "response": api_response,
        "row_ordering_checks": checks,
        "errors": errors,
    }
    atomic_json(args.output, result)
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
