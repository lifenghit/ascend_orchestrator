import os
import tempfile
import time
import unittest

from ascend_orchestrator import ControlPlane


class PhaseFourTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp.name, "state.db")
        self.workspace = os.path.join(self.temp.name, "workspaces")
        self.control = ControlPlane(self.db, self.workspace)
        self.control.register_executor("node-0", [{"id": 0}, {"id": 1}])
        self.task = self.control.create_task("优化 AscendC 算子")

    def tearDown(self):
        self.control.close()
        self.temp.cleanup()

    def test_historical_p90_is_used_for_conservative_backfill(self):
        histories = []
        for duration in (10, 20, 100):
            step = self.control.add_steps(self.task["task_id"], [{
                "kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD"}
            }])[0]
            execution = {"execution_id": "history-%d" % duration, "task_id": self.task["task_id"],
                         "step_id": step["step_id"], "lease_id": "history", "epoch": 0,
                         "status": "SUCCEEDED", "runner": "single-card-test-runner",
                         "started_at": time.time() - duration, "finished_at": time.time()}
            self.control.store.put("executions", execution)
            histories.append(step)
        candidate = self.control.add_steps(self.task["task_id"], [{
            "kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD", "duration_seconds": 10}
        }])[0]
        self.assertEqual(100, self.control.scheduler.duration_estimate(candidate))
        large = self.control.add_steps(self.task["task_id"], [{
            "kind": "DISTRIBUTED_VERIFY", "resource": {"scope": "SINGLE_NODE", "cards_per_node": 2}
        }])[0]
        self.control.request_reservation(large["step_id"], time.time() + 50, guard_seconds=5)
        self.control.schedule()
        self.assertEqual("READY", self.control.store.require("steps", candidate["step_id"])["status"])

    def test_interference_profile_drives_auto_quiet(self):
        profile = self.control.interference_profile_record("SINGLE_NODE", 0.2, shared_network=True, sample_count=5)
        self.assertEqual("GLOBAL_QUIET", profile["recommended_quiet"])
        steps = self.control.add_steps(self.task["task_id"], [
            {"kind": "E2E_OPTIMIZE", "priority": 100,
             "resource": {"scope": "SINGLE_NODE", "cards_per_node": 1, "performance_isolation": "AUTO"}},
            {"kind": "FUNCTIONAL_VERIFY", "priority": 1, "resource": {"scope": "SINGLE_CARD"}},
        ])
        self.control.schedule()
        self.assertEqual("RUNNING", self.control.store.require("steps", steps[0]["step_id"])["status"])
        self.assertEqual("READY", self.control.store.require("steps", steps[1]["step_id"])["status"])

    def test_tuning_is_persistent(self):
        result = self.control.scheduler_tuning_apply(max_agent_workers=3, build_slots_per_node=4)
        self.assertEqual(4, result["build_slots_per_node"])
        self.control.close()
        self.control = ControlPlane(self.db, self.workspace)
        self.assertEqual(3, self.control.scheduler.max_agent_workers)
        self.assertEqual(4, self.control.store.require("nodes", "node-0")["cpu"]["build_slots"])

    def test_metrics_and_self_contained_dashboard(self):
        report = self.control.dashboard_generate(os.path.join(self.temp.name, "dashboard.html"))
        self.assertTrue(os.path.exists(report["path"]))
        with open(report["path"], encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("Prototype metrics from simulated executions", content)
        metrics = self.control.metrics_snapshot()
        self.assertEqual(0, metrics["correctness"]["same_card_conflicts"])
        self.assertIn("queue_wait_p90_seconds", metrics["tasks"])


if __name__ == "__main__":
    unittest.main()
