#!/usr/bin/env python3
"""Cold-prefill and decode measurements for the isolated K3 candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import tempfile

from real_test_client import streamed_completion, token_count


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


def sized_prompt(base: str, target: int) -> tuple[str, int]:
    unit = "cedar delta ember fjord granite harbor iris juniper kinetic lunar maple. "
    best, best_count = unit, token_count(base, unit)
    sample_repeats = 128
    sample_count = token_count(base, unit * sample_repeats)
    tokens_per_repeat = max(1.0, (sample_count - best_count) / (sample_repeats - 1))
    low, high = 1, max(1, int(target / tokens_per_repeat) + 64)
    while low <= high:
        middle = (low + high) // 2
        text = unit * middle
        count = token_count(base, text)
        if count <= target:
            best, best_count = text, count
            low = middle + 1
        else:
            high = middle - 1
    return (
        "Read this archive silently, then reply only READY.\n<archive>\n"
        + best
        + "\n</archive>",
        best_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://192.168.0.238:8893")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decode-runs", default=5, type=int)
    parser.add_argument("--prefill-targets", default="8000,12000,16000,27000")
    args = parser.parse_args()
    started = utc()
    decode = []
    prefill = []

    def progress(current: str | None) -> None:
        atomic_json(args.output, {
            "schema": "glm53-full-exl3-tp3.candidate-benchmark-progress.v1",
            "endpoint": args.endpoint,
            "started_at": started,
            "updated_at": utc(),
            "phase": "running",
            "current_probe": current,
            "completed_decode_runs": len(decode),
            "completed_prefill_runs": len(prefill),
            "decode": decode,
            "prefill": prefill,
            "passed": None,
        })

    for run in range(args.decode_runs):
        progress(f"decode-{run}")
        row = streamed_completion(
            args.endpoint,
            [{"role": "user", "content": "Explain checksum verification in three concise paragraphs."}],
            max_tokens=256,
            timeout=10800,
            chat_template_kwargs={"thinking": False},
        )
        row["run"] = run
        decode.append(row)
    for target in [int(value) for value in args.prefill_targets.split(",")]:
        progress(f"prefill-build-{target}")
        prompt, approximate = sized_prompt(args.endpoint, target)
        measured = token_count(args.endpoint, prompt)
        progress(f"prefill-{target}")
        row = streamed_completion(
            args.endpoint,
            [{"role": "user", "content": prompt}],
            max_tokens=1,
            timeout=10800,
            chat_template_kwargs={"thinking": False},
        )
        row.update(target_tokens=target, approximate_document_tokens=approximate, measured_prompt_tokens=measured)
        prefill.append(row)
    rates = [float(row["decode_tokens_per_second"]) for row in decode if row["decode_tokens_per_second"] is not None]
    ttfts = [float(row["ttft_seconds"]) for row in decode if row["ttft_seconds"] is not None]
    result = {
        "schema": "glm53-full-exl3-tp3.candidate-benchmark.v1",
        "endpoint": args.endpoint,
        "configuration": {
            "tp": 3,
            "dcp": 1,
            "pp": 1,
            "max_model_len": 32768,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 1024,
            "kv_cache_dtype": "fp8",
            "graphs": False,
            "mtp": False,
            "prefix_cache": False,
        },
        "started_at": started,
        "completed_at": utc(),
        "decode": decode,
        "prefill": prefill,
        "decode_tokens_per_second_median": statistics.median(rates),
        "decode_ttft_seconds_median": statistics.median(ttfts),
        "passed": len(rates) == args.decode_runs and all(row["ttft_seconds"] is not None for row in prefill),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit("candidate benchmark did not produce all requested timings")


if __name__ == "__main__":
    main()
