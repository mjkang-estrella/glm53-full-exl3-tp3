#!/usr/bin/env python3
"""Run only the bounded long-context retrieval gate against a live candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from real_test_client import completion, make_long_prompt, post, utc


MODEL = "GLM-5.3-K3-TP3-CANDIDATE"


def raw_native_completion(base: str, user: str, *, max_tokens: int, timeout: int) -> dict:
    prompt = (
        "[gMASK]<sop><|system|>Reasoning Effort: Low"
        f"<|user|>{user}<|assistant|><think></think>\n"
    )
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0,
        "seed": 20260906,
        "max_tokens": max_tokens,
    }
    response, elapsed = post(base, "/v1/completions", payload, timeout=timeout)
    choice = (response.get("choices") or [{}])[0]
    return {
        "request": payload,
        "response": response,
        "content": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "elapsed_seconds": elapsed,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
    parser.add_argument("--endpoint", default="http://192.168.0.238:8893")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-tokens", default=27000, type=int)
    parser.add_argument("--timeout-seconds", default=10800, type=int)
    parser.add_argument("--max-tokens", default=32, type=int)
    parser.add_argument("--mode", choices=("chat", "raw-native"), default="chat")
    args = parser.parse_args()
    if not 1024 <= args.target_tokens <= 30000:
        parser.error("target-tokens must be between 1024 and 30000")

    started_at = utc()
    atomic_json(args.output, {
        "schema": "glm53-full-exl3-tp3.long-context-progress.v1",
        "phase": "building_prompt",
        "endpoint": args.endpoint,
        "target_tokens": args.target_tokens,
        "started_at": started_at,
        "updated_at": utc(),
        "passed": None,
    })
    prompt, measured = make_long_prompt(args.endpoint, args.target_tokens)
    atomic_json(args.output, {
        "schema": "glm53-full-exl3-tp3.long-context-progress.v1",
        "phase": "generating",
        "endpoint": args.endpoint,
        "target_tokens": args.target_tokens,
        "measured_prompt_tokens": measured,
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "started_at": started_at,
        "updated_at": utc(),
        "passed": None,
    })
    if args.mode == "raw-native":
        row = raw_native_completion(
            args.endpoint, prompt, max_tokens=args.max_tokens, timeout=args.timeout_seconds
        )
    else:
        row = completion(
            args.endpoint,
            [{"role": "user", "content": prompt}],
            max_tokens=args.max_tokens,
            timeout=args.timeout_seconds,
            chat_template_kwargs={"thinking": False},
        )
    passed = row["content"].strip() == "K3-NEEDLE-74291" and row["finish_reason"] == "stop"
    result = {
        "schema": "glm53-full-exl3-tp3.long-context.v1",
        "phase": "complete",
        "endpoint": args.endpoint,
        "target_tokens": args.target_tokens,
        "mode": args.mode,
        "client_timeout_seconds": args.timeout_seconds,
        "measured_prompt_tokens": measured,
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "started_at": started_at,
        "completed_at": utc(),
        "request": row["request"],
        "content": row["content"],
        "finish_reason": row["finish_reason"],
        "usage": row["response"].get("usage"),
        "elapsed_seconds": row["elapsed_seconds"],
        "passed": passed,
    }
    # Do not persist the 27K-token prompt itself; its deterministic digest,
    # target, generator implementation, and exact API parameters are enough to
    # reconstruct it while keeping the state tree bounded.
    if "messages" in result["request"]:
        result["request"]["messages"] = [{"role": "user", "content": "<see prompt_sha256 and make_long_prompt>"}]
    else:
        result["request"]["prompt"] = "<native template plus prompt_sha256/make_long_prompt>"
    atomic_json(args.output, result)
    if not passed:
        raise SystemExit("long-context retrieval gate failed")


if __name__ == "__main__":
    main()
