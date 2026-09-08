from __future__ import annotations

import unittest

from tp3k3.lifecycle import evaluate_lifecycle


def event(timestamp: str, line: str) -> dict[str, str]:
    return {"timestamp": timestamp, "line": line}


def timeline(*, ready: str | None = "2026-09-04T00:08:00Z") -> dict:
    return {
        "candidate_runtime": {
            "start": "2026-09-04T00:00:00Z",
            "end": "2026-09-04T00:01:00Z",
        },
        "protected_flash_rollback": {
            "start": "2026-09-04T00:01:01Z",
            "ready": ready,
            "timeout_seconds": 900,
        },
        "post_ready": {"end": "2026-09-04T00:08:15Z"},
    }


def services(**overrides) -> dict:
    containers = {
        "mj-spark-1:glm53-exl3-head": {
            "running": True,
            "oom_killed": False,
            "restart_count_before": 0,
            "restart_count_after": 0,
        },
        "mj-spark-2:glm53-exl3-worker": {
            "running": True,
            "oom_killed": False,
            "restart_count_before": 0,
            "restart_count_after": 0,
        },
        "mj-spark-3:minimax-h3-comfy": {
            "running": True,
            "oom_killed": False,
            "restart_count_before": 0,
            "restart_count_after": 0,
        },
    }
    value = {
        "containers": containers,
        "worker_death_observed": False,
        "health": {"flash_http": 200, "h3_http": 200},
    }
    value.update(overrides)
    return value


def generation(reason: str = "stop") -> dict:
    return {
        "choices": [
            {"finish_reason": reason, "message": {"content": "RESTORED"}}
        ]
    }


class LifecyclePolicyTests(unittest.TestCase):
    def evaluate(self, events=None, **kwargs):
        return evaluate_lifecycle(
            events_by_host=events or {},
            timeline=kwargs.get("timeline", timeline()),
            service_state=kwargs.get("service_state", services()),
            generation=kwargs.get("generation", generation()),
        )

    def test_clean_lifecycle_passes(self):
        self.assertTrue(self.evaluate()["passed"])

    def test_recovered_flash_cold_start_retries_pass(self):
        line = (
            "kernel: NVRM: nvCheckOkFailedNoLog: Check failed: "
            "Out of memory [NV_ERR_NO_MEMORY] returned from allocator"
        )
        result = self.evaluate(
            {
                "mj-spark-1": [event("2026-09-04T00:04:10Z", line)],
                "mj-spark-2": [event("2026-09-04T00:04:11Z", line)],
            }
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["windows"]["protected_flash_cold_start"][
                "allowed_recovered_retry_count"
            ],
            2,
        )

    def test_candidate_nvrm_is_hard_stop(self):
        result = self.evaluate(
            {
                "mj-spark-1": [
                    event(
                        "2026-09-04T00:00:30Z",
                        "kernel: NVRM: Out of memory [NV_ERR_NO_MEMORY]",
                    )
                ]
            }
        )
        self.assertFalse(result["passed"])
        self.assertIn("candidate_runtime_hard_faults:1", result["failure_reasons"])

    def test_cold_start_xid_and_third_rank_nvrm_are_hard_stops(self):
        result = self.evaluate(
            {
                "mj-spark-1": [event("2026-09-04T00:03:00Z", "kernel: NVRM: Xid 79")],
                "mj-spark-3": [
                    event(
                        "2026-09-04T00:03:01Z",
                        "kernel: NVRM: Out of memory [NV_ERR_NO_MEMORY]",
                    )
                ],
            }
        )
        self.assertFalse(result["passed"])
        self.assertIn(
            "protected_flash_cold_start_hard_faults:2",
            result["failure_reasons"],
        )

    def test_post_ready_nvrm_is_hard_stop(self):
        result = self.evaluate(
            {
                "mj-spark-2": [
                    event(
                        "2026-09-04T00:08:05Z",
                        "kernel: NVRM: Out of memory [NV_ERR_NO_MEMORY]",
                    )
                ]
            }
        )
        self.assertFalse(result["passed"])
        self.assertIn("post_ready_hard_faults:1", result["failure_reasons"])

    def test_persistent_retry_span_is_hard_stop(self):
        line = "kernel: NVRM: Out of memory [NV_ERR_NO_MEMORY]"
        result = self.evaluate(
            {
                "mj-spark-1": [
                    event("2026-09-04T00:02:00Z", line),
                    event("2026-09-04T00:05:00Z", line),
                ]
            }
        )
        self.assertFalse(result["passed"])
        self.assertIn(
            "persistent_flash_allocation_retry_loop", result["failure_reasons"]
        )

    def test_service_and_canary_failures_are_hard_stops(self):
        bad = services(worker_death_observed=True)
        bad["containers"]["mj-spark-2:glm53-exl3-worker"]["oom_killed"] = True
        bad["health"]["flash_http"] = 503
        result = self.evaluate(
            service_state=bad, generation=generation("length")
        )
        self.assertFalse(result["passed"])
        self.assertIn("worker_death_observed", result["failure_reasons"])
        self.assertIn("restored_health_failed", result["failure_reasons"])
        self.assertIn("restored_generation_failed", result["failure_reasons"])
        self.assertTrue(
            any(reason.startswith("container_state:") for reason in result["failure_reasons"])
        )

    def test_timeout_and_short_post_ready_window_fail(self):
        late = timeline(ready="2026-09-04T00:20:00Z")
        late["post_ready"]["end"] = "2026-09-04T00:20:05Z"
        result = self.evaluate(timeline=late)
        self.assertFalse(result["passed"])
        self.assertIn("flash_startup_timeout", result["failure_reasons"])
        self.assertIn("post_ready_observation_too_short", result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
