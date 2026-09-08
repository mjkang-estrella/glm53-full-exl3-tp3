from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from tp3k3.safetensors_stream import encoded_header, read_header, sha256_range, write_subset


def _make_source(path: Path) -> tuple[bytes, bytes, int]:
    first = bytes(range(32))
    second = b"byte-exact-payload" * 3
    header = encoded_header({
        "a": {"dtype": "U8", "shape": [len(first)], "data_offsets": [0, len(first)]},
        "b": {"dtype": "U8", "shape": [len(second)], "data_offsets": [len(first), len(first) + len(second)]},
    })
    path.write_bytes(header + first + second)
    return first, second, len(header)


def test_stream_subset_is_byte_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    first, second, data_start = _make_source(source)
    target = tmp_path / "subset.safetensors"
    result = write_subset(source, target, [{
        "name": "b",
        "dtype": "U8",
        "shape": [len(second)],
        "nbytes": len(second),
        "payload_start": data_start + len(first),
        "payload_end": data_start + len(first) + len(second),
        "payload_sha256": hashlib.sha256(second).hexdigest(),
    }], metadata={"source": "unit"})
    output_start, header = read_header(target)
    assert header["__metadata__"] == {"source": "unit"}
    assert set(header) == {"__metadata__", "b"}
    assert header["b"]["data_offsets"] == [0, len(second)]
    assert sha256_range(target, output_start, len(second)) == hashlib.sha256(second).hexdigest()
    assert result["tensor_sha256"]["b"] == hashlib.sha256(second).hexdigest()


def test_encoded_header_is_aligned_and_valid() -> None:
    raw = encoded_header({"x": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}})
    length = struct.unpack("<Q", raw[:8])[0]
    assert length % 8 == 0
    assert len(raw) == length + 8
    assert json.loads(raw[8:])["x"]["shape"] == [1]
