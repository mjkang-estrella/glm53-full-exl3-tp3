#!/usr/bin/env python3
"""Score bounded positions against a sealed BF16 full-vocabulary logit window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np


SOURCE_REVISION = "304b8051cfb2b260b61ce0cbe330e02a98e73639"
REAL_VOCAB_SIZE = 154_880


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def safetensor_f32(path: Path, tensor: str = "logits") -> np.memmap:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    spec = header[tensor]
    if spec["dtype"] != "F32":
        raise ValueError(f"expected F32 teacher logits, got {spec['dtype']}")
    start, end = spec["data_offsets"]
    shape = tuple(spec["shape"])
    expected = math.prod(shape) * 4
    if end - start != expected:
        raise ValueError("teacher tensor byte span disagrees with shape")
    return np.memmap(path, mode="r", dtype="<f4", offset=8 + header_len + start, shape=shape)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def parse_full_logprobs(response: dict, requested_vocab: int) -> tuple[np.ndarray, int]:
    choice = response["choices"][0]
    result = np.full(requested_vocab, np.nan, dtype=np.float64)
    entries = choice["logprobs"]["top_logprobs"][0]
    for key, value in entries.items():
        if not key.startswith("token_id:"):
            raise ValueError(f"unexpected logprob key: {key!r}")
        token_id = int(key.removeprefix("token_id:"))
        if 0 <= token_id < requested_vocab:
            result[token_id] = float(value)
    missing = np.flatnonzero(~np.isfinite(result))
    if len(missing):
        raise ValueError(f"full logprobs missing/nonfinite for {len(missing)} requested tokens")
    generated = choice.get("token_ids")
    if not isinstance(generated, list) or len(generated) != 1:
        raise ValueError(f"expected one returned token id, got {generated!r}")
    return result, int(generated[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--positions", default="127,255,511,1023,1535,2046")
    parser.add_argument("--candidate-vocab", default=REAL_VOCAB_SIZE, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", default=1800, type=int)
    args = parser.parse_args()

    started = time.time()
    actual_teacher_sha = sha256_file(args.teacher)
    if actual_teacher_sha != args.teacher_sha256:
        raise ValueError(f"teacher SHA mismatch: {actual_teacher_sha}")
    tokens = np.load(args.tokens, mmap_mode="r")
    if tokens.ndim != 1:
        raise ValueError(f"expected one-dimensional token array, got {tokens.shape}")
    teacher = safetensor_f32(args.teacher)
    if teacher.shape != (len(tokens) - 1, REAL_VOCAB_SIZE):
        raise ValueError(f"teacher/token geometry mismatch: {teacher.shape}, {tokens.shape}")
    positions = [int(item) for item in args.positions.split(",")]
    if any(position < 0 or position >= teacher.shape[0] for position in positions):
        raise ValueError("requested position is outside teacher window")

    records: list[dict] = []
    for position in positions:
        prefix = [int(value) for value in tokens[: position + 1]]
        request_started = time.time()
        response = post_json(
            args.endpoint.rstrip("/") + "/v1/completions",
            {
                "model": args.model,
                "prompt": prefix,
                "add_special_tokens": False,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": args.candidate_vocab,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
                "ignore_eos": True,
                "seed": 1,
            },
            args.timeout,
        )
        candidate_logp, generated = parse_full_logprobs(response, args.candidate_vocab)
        if args.candidate_vocab != REAL_VOCAB_SIZE:
            raise ValueError(f"candidate vocab must be the semantic {REAL_VOCAB_SIZE}")
        candidate_logp -= np.logaddexp.reduce(candidate_logp)
        teacher_logits = np.asarray(teacher[position], dtype=np.float64)
        teacher_logp = teacher_logits - np.logaddexp.reduce(teacher_logits)
        teacher_p = np.exp(teacher_logp)
        kld = float(np.sum(teacher_p * (teacher_logp - candidate_logp), dtype=np.float64))
        teacher_argmax = int(np.argmax(teacher_logits))
        candidate_argmax = int(np.argmax(candidate_logp))
        records.append(
            {
                "position": position,
                "prefix_tokens": len(prefix),
                "target_token": int(tokens[position + 1]),
                "teacher_argmax": teacher_argmax,
                "candidate_argmax": candidate_argmax,
                "generated_token": generated,
                "argmax_agreement": teacher_argmax == candidate_argmax,
                "generated_matches_candidate_argmax": generated == candidate_argmax,
                "kld_bf16_to_k3": kld,
                "elapsed_seconds": time.time() - request_started,
            }
        )

    output = {
        "schema": "glm53-full-exl3-tp3.matched-kld.v1",
        "passed": all(r["generated_matches_candidate_argmax"] for r in records),
        "reference_scope": "supplemental-public-reference-nonfinal",
        "reference_repo": "brandonmusic/GLM-5.3-BF16-full-logits",
        "reference_revision": "427368f12a4bdc21668bc4171ce0dc54f8990200",
        "bf16_model_revision": SOURCE_REVISION,
        "window_id": args.window_id,
        "teacher_file": str(args.teacher),
        "teacher_sha256": actual_teacher_sha,
        "token_file": str(args.tokens),
        "token_sha256": sha256_file(args.tokens),
        "positions": records,
        "position_count": len(records),
        "mean_kld_bf16_to_k3": float(np.mean([r["kld_bf16_to_k3"] for r in records])),
        "max_kld_bf16_to_k3": max(r["kld_bf16_to_k3"] for r in records),
        "argmax_agreement": sum(r["argmax_agreement"] for r in records) / len(records),
        "started_at_unix": started,
        "completed_at_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
