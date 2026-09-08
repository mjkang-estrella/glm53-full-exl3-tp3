#!/usr/bin/env python3
"""Reproducible real-generation, tool-call, long-context, and timing probes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MODEL = "GLM-5.3-K3-TP3-CANDIDATE"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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


def post(base: str, path: str, payload: dict[str, Any], timeout: int = 1800) -> tuple[dict, float]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(base + path, data=raw, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
    return json.loads(body), time.monotonic() - started


def completion(
    base: str,
    messages: list[dict],
    *,
    max_tokens: int,
    timeout: int = 1800,
    **extra: Any,
) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": 20260906,
        "max_tokens": max_tokens,
        **extra,
    }
    requested = utc()
    response, elapsed = post(base, "/v1/chat/completions", payload, timeout=timeout)
    choice = response.get("choices", [{}])[0]
    return {
        "requested_at": requested,
        "completed_at": utc(),
        "elapsed_seconds": elapsed,
        "request": payload,
        "response": response,
        "finish_reason": choice.get("finish_reason"),
        "content": (choice.get("message") or {}).get("content") or "",
        "reasoning": (choice.get("message") or {}).get("reasoning") or "",
        "tool_calls": (choice.get("message") or {}).get("tool_calls") or [],
    }


def streamed_completion(
    base: str,
    messages: list[dict],
    *,
    max_tokens: int,
    timeout: int = 1800,
    **extra: Any,
) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": 20260906,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(base + "/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"})
    started_at = utc()
    started = time.monotonic()
    first = None
    content: list[str] = []
    reasoning: list[str] = []
    usage = None
    finish = None
    with urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="strict").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            row = json.loads(line[6:])
            if row.get("usage"):
                usage = row["usage"]
            for choice in row.get("choices", []):
                delta = choice.get("delta") or {}
                piece = (delta.get("reasoning") or "") + (delta.get("content") or "")
                if piece and first is None:
                    first = time.monotonic()
                if delta.get("reasoning"):
                    reasoning.append(delta["reasoning"])
                if delta.get("content"):
                    content.append(delta["content"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    ended = time.monotonic()
    ttft = None if first is None else first - started
    completion_tokens = int((usage or {}).get("completion_tokens", 0))
    decode_span = None if first is None else max(ended - first, 1e-9)
    return {
        "requested_at": started_at,
        "completed_at": utc(),
        "request": payload,
        "finish_reason": finish,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "elapsed_seconds": ended - started,
        "ttft_seconds": ttft,
        "decode_tokens_per_second": None if decode_span is None else max(0, completion_tokens - 1) / decode_span,
    }


def token_count(base: str, text: str) -> int:
    response, _ = post(base, "/tokenize", {"model": MODEL, "prompt": text})
    if "count" in response:
        return int(response["count"])
    if "tokens" in response:
        return len(response["tokens"])
    raise RuntimeError(f"unexpected tokenize response: {response}")


def make_long_prompt(base: str, target_tokens: int = 27000) -> tuple[str, int]:
    needle = "K3-NEEDLE-74291"
    unit = "archive cedar delta ember fjord granite harbor iris juniper kinetic lunar maple. "
    best = unit
    best_count = token_count(base, best)
    # Bound the binary search in repeat-count space. Using target_tokens as a
    # repeat count can make the first tokenize request hundreds of thousands
    # of tokens and fail before the actual 32K probe starts.
    sample_repeats = 128
    sample_count = token_count(base, unit * sample_repeats)
    tokens_per_repeat = max(1.0, (sample_count - best_count) / (sample_repeats - 1))
    low, high = 1, max(1, int(target_tokens / tokens_per_repeat) + 64)
    while low <= high:
        middle = (low + high) // 2
        text = unit * middle
        count = token_count(base, text)
        if count <= target_tokens:
            best, best_count = text, count
            low = middle + 1
        else:
            high = middle - 1
    pivot = len(best) * 2 // 3
    document = best[:pivot] + f"\nThe unique retrieval code is {needle}.\n" + best[pivot:]
    prompt = (
        "Read the document. Return only the unique retrieval code, with no explanation.\n"
        "<document>\n" + document + "\n</document>"
    )
    return prompt, token_count(base, prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://192.168.0.238:8893")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-long", action="store_true")
    args = parser.parse_args()
    tests: dict[str, dict] = {}

    def progress(current: str) -> None:
        atomic_json(
            args.output,
            {
                "schema": "glm53-full-exl3-tp3.real-tests-progress.v1",
                "endpoint": args.endpoint,
                "model": MODEL,
                "updated_at": utc(),
                "phase": "running",
                "current_test": current,
                "completed_tests": list(tests),
                "tests": tests,
                "passed": None,
            },
        )

    exact_messages = [{"role": "user", "content": "Reply with exactly K3_TP3_OK and nothing else."}]
    progress("exact_1")
    tests["exact_1"] = completion(args.endpoint, exact_messages, max_tokens=16, chat_template_kwargs={"thinking": False})
    progress("exact_2")
    tests["exact_2"] = completion(args.endpoint, exact_messages, max_tokens=16, chat_template_kwargs={"thinking": False})
    progress("arithmetic")
    tests["arithmetic"] = completion(
        args.endpoint,
        [{"role": "user", "content": "What is 17 × 19? Reply with only the integer."}],
        max_tokens=32,
        chat_template_kwargs={"thinking": False},
    )
    progress("korean")
    tests["korean"] = completion(
        args.endpoint,
        [{"role": "user", "content": "다음 문장을 글자 깨짐 없이 정확히 한 번만 출력하세요: 인터페이스 층은 네트워크 효과 때문에 집중되고, 창출 층은 롱테일로 파편화된다."}],
        max_tokens=96,
        chat_template_kwargs={"thinking": False},
    )
    progress("reasoning")
    tests["reasoning"] = completion(
        args.endpoint,
        [{"role": "user", "content": "A box has 3 red and 2 blue balls. Two are drawn without replacement. What is the probability both are red? Give the reduced fraction."}],
        max_tokens=384,
        chat_template_kwargs={"thinking": True},
    )
    progress("code")
    tests["code"] = completion(
        args.endpoint,
        [{"role": "user", "content": "Write only a Python function named add(a, b) that returns their sum."}],
        max_tokens=96,
        chat_template_kwargs={"thinking": False},
    )
    progress("code_complex")
    tests["code_complex"] = completion(
        args.endpoint,
        [{"role": "user", "content": "Write a correct Python function is_prime(n: int) -> bool. Include edge cases and no prose outside one code block."}],
        max_tokens=384,
        chat_template_kwargs={"thinking": False},
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }]
    progress("tool_auto")
    tests["tool_auto"] = completion(
        args.endpoint,
        [{"role": "user", "content": "What is the weather in Seoul right now? Use the tool."}],
        max_tokens=256,
        tools=tools,
        tool_choice="auto",
        chat_template_kwargs={"thinking": True},
    )
    progress("stream_short")
    tests["stream_short"] = streamed_completion(
        args.endpoint,
        [{"role": "user", "content": "In two concise paragraphs, explain why checksums matter for model checkpoint replication."}],
        max_tokens=192,
    )
    if not args.skip_long:
        progress("long_context_build")
        long_prompt, count = make_long_prompt(args.endpoint)
        progress("long_context")
        tests["long_context"] = completion(
            args.endpoint,
            [{"role": "user", "content": long_prompt}],
            max_tokens=32,
        )
        tests["long_context"]["measured_prompt_tokens"] = count

    checks = {
        "exact_1": tests["exact_1"]["content"].strip() == "K3_TP3_OK",
        "exact_2": tests["exact_2"]["content"].strip() == "K3_TP3_OK",
        "deterministic": tests["exact_1"]["content"] == tests["exact_2"]["content"],
        "arithmetic": tests["arithmetic"]["content"].strip() == "323",
        "korean": "인터페이스 층" in tests["korean"]["content"] and "창출 층" in tests["korean"]["content"] and "�" not in tests["korean"]["content"],
        "reasoning": ("3/10" in tests["reasoning"]["content"] or r"\frac{3}{10}" in tests["reasoning"]["content"]) and bool(tests["reasoning"]["reasoning"]),
        "code": all(token in tests["code"]["content"] for token in ("def add", "return", "a + b")),
        "code_complex": "def is_prime" in tests["code_complex"]["content"] and "return" in tests["code_complex"]["content"],
        "tool_auto": bool(tests["tool_auto"]["tool_calls"]),
        "stream_short": bool(tests["stream_short"]["content"].strip()),
        "normal_finish_reasons": all(
            row["finish_reason"] == "stop"
            for name, row in tests.items()
            if name != "tool_auto" and name != "long_context"
        ) and tests["tool_auto"]["finish_reason"] in ("stop", "tool_calls"),
    }
    if "long_context" in tests:
        checks["long_context"] = tests["long_context"]["content"].strip() == "K3-NEEDLE-74291"
        checks["long_context_finish"] = tests["long_context"]["finish_reason"] == "stop"
    result = {
        "schema": "glm53-full-exl3-tp3.real-tests.v1",
        "endpoint": args.endpoint,
        "model": MODEL,
        "completed_at": utc(),
        "checks": checks,
        "passed": all(checks.values()),
        "tests": tests,
    }
    atomic_json(args.output, result)
    if not result["passed"]:
        raise SystemExit("one or more real-generation checks failed")


if __name__ == "__main__":
    main()
