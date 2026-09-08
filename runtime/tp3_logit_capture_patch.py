"""Opt-in, bounded raw-logit capture for matched BF16 scoring.

The normal serving path is unchanged unless a valid ``ARM.json`` is placed in
the rank-0 capture directory. Prompt-logprob execution then streams at most the
declared row count as float32 raw parts and atomically seals a receipt.
"""

from __future__ import annotations

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_state: dict[str, Any] | None = None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_part(path: Path, array) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    digest = hashlib.sha256()
    view = memoryview(array).cast("B")
    with temporary.open("wb", buffering=0) as handle:
        for start in range(0, len(view), 16 << 20):
            chunk = view[start : start + (16 << 20)]
            handle.write(chunk)
            digest.update(chunk)
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return len(view), digest.hexdigest()


def _capture(root: Path, logits) -> None:
    global _state
    arm_path = root / "ARM.json"
    if not arm_path.is_file():
        return
    arm = json.loads(arm_path.read_text(encoding="utf-8"))
    if arm.get("schema") != "glm53-full-exl3-tp3.logit-capture-arm.v1":
        raise RuntimeError("raw-logit ARM schema differs")
    capture_id = str(arm.get("capture_id", ""))
    if not _SAFE_ID.fullmatch(capture_id):
        raise RuntimeError("unsafe raw-logit capture id")
    expected_rows = int(arm["expected_rows"])
    expected_vocab = int(arm["expected_vocab"])
    if expected_rows <= 0 or expected_rows > 32768 or expected_vocab != 154880:
        raise RuntimeError("raw-logit capture bounds differ")
    if logits.ndim != 2 or logits.shape[1] < expected_vocab:
        raise RuntimeError(f"raw-logit tensor geometry differs: {tuple(logits.shape)}")
    if _state is None or _state["capture_id"] != capture_id:
        incoming = root / f"{capture_id}.incomplete"
        final = root / capture_id
        if incoming.exists() or final.exists():
            raise RuntimeError(f"refusing to overwrite raw-logit evidence {capture_id}")
        incoming.mkdir(parents=True)
        _state = {
            "capture_id": capture_id,
            "incoming": incoming,
            "final": final,
            "rows": 0,
            "parts": [],
            "arm": arm,
        }
    remaining = expected_rows - int(_state["rows"])
    if remaining <= 0:
        return
    take = min(remaining, int(logits.shape[0]))
    # The gathered logits contain only the semantic vocabulary; the explicit
    # slice also protects against any future internal padded-vocabulary view.
    array = logits[:take, :expected_vocab].detach().float().cpu().contiguous().numpy()
    part_index = len(_state["parts"])
    part_name = f"part-{part_index:04d}.f32"
    size, digest = _write_part(_state["incoming"] / part_name, array)
    record = {
        "path": part_name,
        "first_row": int(_state["rows"]),
        "rows": take,
        "vocab": expected_vocab,
        "dtype": "float32-little-endian",
        "bytes": size,
        "sha256": digest,
    }
    _state["parts"].append(record)
    _state["rows"] += take
    _atomic_json(
        _state["incoming"] / "STATUS.json",
        {
            "schema": "glm53-full-exl3-tp3.logit-capture-status.v1",
            "capture_id": capture_id,
            "rows": _state["rows"],
            "expected_rows": expected_rows,
            "parts": _state["parts"],
        },
    )
    if _state["rows"] == expected_rows:
        receipt = {
            "schema": "glm53-full-exl3-tp3.logit-capture-complete.v1",
            "passed": True,
            "capture_id": capture_id,
            "window_id": arm["window_id"],
            "token_sha256": arm["token_sha256"],
            "model": arm["model"],
            "expected_rows": expected_rows,
            "expected_vocab": expected_vocab,
            "dtype": "float32-little-endian",
            "bytes": sum(part["bytes"] for part in _state["parts"]),
            "parts": _state["parts"],
        }
        _atomic_json(_state["incoming"] / "CAPTURE_COMPLETE.json", receipt)
        os.replace(_state["incoming"], _state["final"])
        os.replace(arm_path, root / f"ARM.consumed-{capture_id}.json")
        print(
            f"GLM53_MATCHED_LOGITS_CAPTURE_OK id={capture_id} "
            f"rows={expected_rows} vocab={expected_vocab}",
            flush=True,
        )
        _state = None


def apply_patches() -> None:
    root_raw = os.environ.get("GLM53_K3_LOGIT_CAPTURE_ROOT")
    if not root_raw or getattr(apply_patches, "_done", False):
        return
    apply_patches._done = True
    root = Path(root_raw)
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    original = LogitsProcessor.forward

    @wraps(original)
    def forward(self, *args, **kwargs):
        logits = original(self, *args, **kwargs)
        if logits is not None and (root / "ARM.json").is_file():
            _capture(root, logits)
        return logits

    LogitsProcessor.forward = forward
    print(f"GLM53_MATCHED_LOGITS_CAPTURE_PATCH_INSTALLED root={root}", flush=True)


__all__ = ["apply_patches"]
