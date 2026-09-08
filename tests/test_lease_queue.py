from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from tp3k3.lease_queue import (
    counts,
    finish,
    initialize,
    lease_next,
    prefer_pending_node,
    read,
    requeue,
)


class LeaseQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        initialize(self.root, range(3, 79))

    def tearDown(self):
        self.temporary.cleanup()

    def test_initialization_is_idempotent_and_complete(self):
        initialize(self.root, range(3, 79))
        self.assertEqual(
            counts(self.root),
            {"pending": 76, "leased": 0, "completed": 0, "failed": 0},
        )

    def test_layer_three_prefers_spark_three(self):
        other = lease_next(self.root, "mj-spark-1")
        self.assertNotEqual(other["layer"], 3)
        value = lease_next(self.root, "mj-spark-3")
        self.assertEqual(value["layer"], 3)
        self.assertEqual(value["node"], "mj-spark-3")

    def test_one_active_lease_per_node_and_atomic_completion(self):
        value = lease_next(self.root, "mj-spark-1")
        self.assertIsNotNone(value)
        self.assertIsNone(lease_next(self.root, "mj-spark-1"))
        finished = finish(self.root, value["layer"], value["lease_id"], {"passed": True})
        self.assertTrue(finished["result"]["passed"])
        self.assertEqual(counts(self.root)["completed"], 1)
        self.assertEqual(read(self.root / "completed" / f"layer-{value['layer']:03d}.json")["lease_id"], value["lease_id"])

    def test_failed_or_slow_work_can_be_requeued_to_another_node(self):
        value = lease_next(self.root, "mj-spark-2")
        layer = value["layer"]
        requeue(self.root, layer, value["lease_id"], "stale_worker")
        replacement = lease_next(self.root, "mj-spark-1")
        self.assertEqual(replacement["layer"], layer)
        self.assertEqual(replacement["attempts"], 2)
        self.assertEqual(replacement["node"], "mj-spark-1")

    def test_stale_receipt_cannot_complete_new_lease(self):
        first = lease_next(self.root, "mj-spark-1")
        requeue(self.root, first["layer"], first["lease_id"], "lost")
        second = lease_next(self.root, "mj-spark-2")
        with self.assertRaises(ValueError):
            finish(self.root, second["layer"], first["lease_id"], {"passed": True})

    def test_verified_partial_work_can_prefer_its_original_node(self):
        prefer_pending_node(self.root, 52, "mj-spark-1", evidence="partial.json")
        first = lease_next(self.root, "mj-spark-1")
        self.assertEqual(first["layer"], 52)
        self.assertEqual(first["preferred_node"], "mj-spark-1")
        self.assertEqual(first["history"][-2]["event"], "recovery_preference")

    def test_recovery_preference_is_not_a_fixed_assignment(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        try:
            initialize(root, [52])
            prefer_pending_node(root, 52, "mj-spark-1", evidence="partial.json")
            takeover = lease_next(root, "mj-spark-2")
            self.assertEqual(takeover["layer"], 52)
            self.assertEqual(takeover["node"], "mj-spark-2")
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
