#!/usr/bin/env python3
"""Fast deterministic raw-completion probes using the pinned native template."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback
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


def native_prompt(user: str) -> str:
    return (
        "[gMASK]<sop><|system|>Reasoning Effort: Low"
        f"<|user|>{user}<|assistant|><think></think>\n"
    )


def complete(endpoint: str, user: str, max_tokens: int, timeout: int = 3600) -> dict:
    payload = {
        "model": MODEL,
        "prompt": native_prompt(user),
        "temperature": 0,
        "seed": 20260906,
        "max_tokens": max_tokens,
    }
    request = Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    requested = utc()
    started = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    choice = (result.get("choices") or [{}])[0]
    return {
        "requested_at": requested,
        "completed_at": utc(),
        "elapsed_seconds": elapsed,
        "request": payload,
        "response": result,
        "text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": result.get("usage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://192.168.0.238:8893")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probes",
        default="exact,arithmetic,code,korean",
        help="comma-separated subset of exact,arithmetic,code,code_repair,korean",
    )
    args = parser.parse_args()
    specifications = {
        "exact": ("Reply with exactly K3_TP3_OK and nothing else.", 16),
        "arithmetic": ("What is 17 × 19? Reply with only the integer.", 16),
        "code": (
            "Write only a Python function named add(a, b) that returns their sum.",
            64,
        ),
        "code_repair": (
            "Repair this Python function so it sums all non-None values, including zero. Return only one Python code block.\ndef total(xs): return sum(x for x in xs if x)",
            96,
        ),
        "korean": (
            "다음 문장을 글자 깨짐 없이 정확히 한 번만 출력하세요: 인터페이스 층은 네트워크 효과 때문에 집중되고, 창출 층은 롱테일로 파편화된다.",
            96,
        ),
    }
    requested = [value.strip() for value in args.probes.split(",") if value.strip()]
    if not requested or len(requested) != len(set(requested)):
        parser.error("probe list must be nonempty and unique")
    unknown = [value for value in requested if value not in specifications]
    if unknown:
        parser.error(f"unknown probes: {','.join(unknown)}")
    started_at = utc()
    probes: dict[str, dict] = {}

    def progress(phase: str, current: str | None = None, error: str | None = None) -> None:
        atomic_json(
            args.output,
            {
                "schema": "glm53-full-exl3-tp3.quick-real-probes.v2",
                "endpoint": args.endpoint,
                "started_at": started_at,
                "updated_at": utc(),
                "phase": phase,
                "current_probe": current,
                "requested_probes": requested,
                "completed_probes": list(probes),
                "probes": probes,
                "error": error,
                "passed": None,
            },
        )

    for name in requested:
        progress("running", name)
        user, max_tokens = specifications[name]
        try:
            probes[name] = complete(args.endpoint, user, max_tokens)
        except Exception:
            progress("failed", name, traceback.format_exc())
            raise

    checks: dict[str, bool] = {}
    if "exact" in probes:
        checks["exact"] = probes["exact"]["text"].strip() == "K3_TP3_OK"
    if "arithmetic" in probes:
        checks["arithmetic"] = probes["arithmetic"]["text"].strip() == "323"
    if "code" in probes:
        checks["code"] = all(
            value in probes["code"]["text"] for value in ("def add", "return", "a + b")
        )
    if "code_repair" in probes:
        checks["code_repair_strict"] = bool(re.fullmatch(
            r"```(?:python)?\s*\ndef total\(xs\):\s*return sum\(x for x in xs if x is not None\)\s*\n```",
            probes["code_repair"]["text"].strip(),
        ))
    if "korean" in probes:
        checks["korean"] = (
            "인터페이스 층" in probes["korean"]["text"]
            and "창출 층" in probes["korean"]["text"]
            and "�" not in probes["korean"]["text"]
        )
    checks["normal_stops"] = all(
        probe["finish_reason"] == "stop" for probe in probes.values()
    )
    result = {
        "schema": "glm53-full-exl3-tp3.quick-real-probes.v2",
        "endpoint": args.endpoint,
        "started_at": started_at,
        "completed_at": utc(),
        "phase": "complete",
        "requested_probes": requested,
        "completed_probes": list(probes),
        "passed": all(checks.values()),
        "checks": checks,
        "probes": probes,
    }
    atomic_json(args.output, result)
    if not result["passed"]:
        raise SystemExit("one or more quick candidate probes failed")


if __name__ == "__main__":
    main()
