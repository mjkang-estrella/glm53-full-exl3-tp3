#!/usr/bin/env python3
"""Durable reasoning, structured-output, code, and tool-call probes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import traceback

from real_test_client import completion, utc


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
    parser.add_argument(
        "--probes",
        default="reasoning,code_repair,json_schema,tool_auto,tool_forced",
    )
    args = parser.parse_args()
    requested = [name.strip() for name in args.probes.split(",") if name.strip()]
    allowed = {"reasoning", "code_repair", "json_schema", "tool_auto", "tool_forced"}
    if not requested or len(requested) != len(set(requested)) or not set(requested) <= allowed:
        parser.error("probes must be a unique subset of " + ",".join(sorted(allowed)))

    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    specifications = {
        "reasoning": lambda: completion(
            args.endpoint,
            [{"role": "user", "content": "A box has 3 red and 2 blue balls. Two are drawn without replacement. What is the probability both are red? Give the reduced fraction."}],
            # GLM may place a concise derivation in both reasoning and visible
            # content before stopping; 128 truncated a correct first result.
            max_tokens=384,
            chat_template_kwargs={"thinking": True},
        ),
        "code_repair": lambda: completion(
            args.endpoint,
            [{"role": "user", "content": "Repair this Python function so it sums all non-None values, including zero. Return only one code block.\ndef total(xs): return sum(x for x in xs if x)"}],
            # The model explains the falsy-value bug before its requested code
            # block even with thinking disabled. Preserve a normal-stop gate.
            max_tokens=384,
            chat_template_kwargs={"thinking": False},
        ),
        "json_schema": lambda: completion(
            args.endpoint,
            [{"role": "user", "content": "Report the completed layer count."}],
            max_tokens=64,
            chat_template_kwargs={"thinking": False},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "layer_status",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["complete"]},
                            "layers": {"type": "integer", "enum": [76]},
                        },
                        "required": ["status", "layers"],
                        "additionalProperties": False,
                    },
                },
            },
        ),
        "tool_auto": lambda: completion(
            args.endpoint,
            [{"role": "user", "content": "What is the weather in Seoul right now? Use the tool."}],
            max_tokens=256,
            tools=[weather_tool],
            tool_choice="auto",
            chat_template_kwargs={"thinking": True},
        ),
        "tool_forced": lambda: completion(
            args.endpoint,
            [{"role": "user", "content": "Get the current weather for Busan."}],
            max_tokens=128,
            tools=[weather_tool],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            chat_template_kwargs={"thinking": False},
        ),
    }
    started_at = utc()
    probes: dict[str, dict] = {}

    def progress(phase: str, current: str | None = None, error: str | None = None) -> None:
        atomic_json(
            args.output,
            {
                "schema": "glm53-full-exl3-tp3.extended-probes.v1",
                "endpoint": args.endpoint,
                "started_at": started_at,
                "updated_at": utc(),
                "phase": phase,
                "requested_probes": requested,
                "completed_probes": list(probes),
                "current_probe": current,
                "probes": probes,
                "error": error,
                "passed": None,
            },
        )

    for name in requested:
        progress("running", name)
        try:
            probes[name] = specifications[name]()
        except Exception:
            progress("failed", name, traceback.format_exc())
            raise

    checks: dict[str, bool] = {}
    if "reasoning" in probes:
        checks["reasoning_answer"] = "3/10" in probes["reasoning"]["content"] or r"\frac{3}{10}" in probes["reasoning"]["content"]
        checks["reasoning_trace"] = bool(probes["reasoning"]["reasoning"].strip())
    if "code_repair" in probes:
        text = probes["code_repair"]["content"].strip()
        checks["code_repair"] = bool(re.fullmatch(
            r"```(?:python)?\s*\ndef total\(xs\):\s*return sum\(x for x in xs if x is not None\)\s*\n```",
            text,
        ))
    if "json_schema" in probes:
        try:
            value = json.loads(probes["json_schema"]["content"])
        except (TypeError, json.JSONDecodeError):
            value = None
        checks["json_schema"] = value == {"status": "complete", "layers": 76}
    for name, city in (("tool_auto", "Seoul"), ("tool_forced", "Busan")):
        if name in probes:
            calls = probes[name]["tool_calls"]
            checks[name] = bool(calls) and calls[0].get("function", {}).get("name") == "get_weather" and city.lower() in calls[0].get("function", {}).get("arguments", "").lower()
    checks["finish_reasons"] = all(
        row["finish_reason"] in ({"stop"} if name not in {"tool_auto", "tool_forced"} else {"stop", "tool_calls"})
        for name, row in probes.items()
    )
    result = {
        "schema": "glm53-full-exl3-tp3.extended-probes.v1",
        "endpoint": args.endpoint,
        "started_at": started_at,
        "completed_at": utc(),
        "phase": "complete",
        "requested_probes": requested,
        "completed_probes": list(probes),
        "checks": checks,
        "probes": probes,
        "passed": all(checks.values()),
    }
    atomic_json(args.output, result)
    if not result["passed"]:
        raise SystemExit("one or more extended candidate probes failed")


if __name__ == "__main__":
    main()
