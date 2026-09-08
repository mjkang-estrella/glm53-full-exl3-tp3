from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_partial_layer import SOURCE_REVISION, verify_partial
from tp3k3.geometry import GEOMETRY_ID, layer_geometry


class PartialRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.layer_dir = Path(self.temporary.name) / "layer-052"
        (self.layer_dir / "experts").mkdir(parents=True)
        (self.layer_dir / "receipts").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def write_expert(self, expert: int, payload: bytes = b"resumable-k3") -> None:
        shard = self.layer_dir / "experts" / f"expert-{expert:03d}.safetensors"
        shard.write_bytes(payload)
        receipt = {
            "schema": "glm53-full-exl3-tp3.expert-receipt.v2",
            "geometry_id": GEOMETRY_ID,
            "layer": 52,
            "expert": expert,
            "bits": 3,
            "tp": 3,
            "padding_channels": 0,
            "layer_geometry": layer_geometry(52),
            "source_revision": SOURCE_REVISION,
            "output_file": shard.name,
            "output_bytes": len(payload),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "passed": True,
        }
        (self.layer_dir / "receipts" / f"expert-{expert:03d}.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    def test_checksum_verified_partial_is_resumable(self):
        self.write_expert(0)
        self.write_expert(1)
        value = verify_partial(self.layer_dir, 52)
        self.assertTrue(value["passed"])
        self.assertEqual(value["verified_experts"], 2)
        self.assertEqual(value["expert_ids"], [0, 1])

    def test_corrupt_partial_is_not_accepted(self):
        self.write_expert(0)
        (self.layer_dir / "experts" / "expert-000.safetensors").write_bytes(b"changed")
        value = verify_partial(self.layer_dir, 52)
        self.assertFalse(value["passed"])
        self.assertIn("SHA-256", value["errors"][0])

    def test_unpaired_shard_is_not_accepted(self):
        (self.layer_dir / "experts" / "expert-002.safetensors").write_bytes(b"orphan")
        value = verify_partial(self.layer_dir, 52)
        self.assertFalse(value["passed"])
        self.assertIn("unpaired expert shards", value["errors"][0])


if __name__ == "__main__":
    unittest.main()
