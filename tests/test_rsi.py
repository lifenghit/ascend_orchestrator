import os
import tempfile
import unittest

from ascend_orchestrator import ControlPlane
from ascend_orchestrator.models import ConflictError, FencingError
from ascend_orchestrator.rsi_demo import run_rsi_demo


class RSITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.control = ControlPlane(os.path.join(self.temp.name, "state.db"),
                                    os.path.join(self.temp.name, "workspaces"))
        self.control.register_executor("node-0", [{"id": 0}, {"id": 1}])
        self.task = self.control.create_task("优化 AscendC 算子")
        self.baseline = self.execute(metrics={"latency_ms": 10.0, "protocol_digest": "operator-v1"})

    def tearDown(self):
        self.control.close()
        self.temp.cleanup()

    def execute(self, metrics, correctness="PASSED", candidate_id=None, level="OPERATOR"):
        step = self.control.add_steps(self.task["task_id"], [{
            "kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD"}
        }])[0]
        self.control.schedule()
        step = self.control.store.require("steps", step["step_id"])
        lease = self.control.store.require("leases", step["lease_id"])
        result = self.control.run_step(step["step_id"], lease["lease_id"], lease["epoch"],
                                       correctness=correctness, metrics=metrics, candidate_id=candidate_id)
        return result["evidence"]

    def campaign(self, level="OPERATOR", role="GENERAL", baseline=None, rounds=3):
        return self.control.campaign_create(
            self.task["task_id"], level, (baseline or self.baseline)["evidence_id"],
            {"primary_metric": "latency_ms", "direction": "MINIMIZE", "correctness_gate": "REQUIRED",
             "min_improvement": 0.1, "measurement_noise": 0.05, "retest_on_noise": False},
            {"max_rounds": rounds, "max_no_improvement_rounds": 2, "max_wall_time_minutes": 10},
            role=role,
        )

    def test_operator_accepts_real_improvement_and_checkpoints(self):
        campaign = self.campaign()
        candidate = self.control.campaign_append_candidate(campaign["campaign_id"], "Reduce tiling overhead", "KERNEL_ONLY")
        evidence = self.execute({"latency_ms": 8.0, "protocol_digest": "operator-v1"}, candidate_id=candidate["candidate_id"])
        result = self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])
        self.assertEqual("KERNEL_CANDIDATE_ACCEPTED", result["evaluation"]["decision"])
        self.assertEqual(8.0, result["campaign"]["best_metric"])
        checkpoint = self.control.campaign_checkpoint(campaign["campaign_id"])
        self.assertEqual(candidate["candidate_id"], checkpoint["checkpoint"]["accepted_candidate_id"])
        with self.assertRaises(FencingError):
            self.control.campaign_append_candidate(campaign["campaign_id"], "other leader", "KERNEL_ONLY", leader="other")

    def test_correctness_failure_cannot_be_accepted(self):
        campaign = self.campaign()
        candidate = self.control.campaign_append_candidate(campaign["campaign_id"], "Unsafe fast path", "KERNEL_ONLY")
        evidence = self.execute({"latency_ms": 1.0, "protocol_digest": "operator-v1"},
                                correctness="FAILED", candidate_id=candidate["candidate_id"])
        result = self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])
        self.assertEqual("REJECTED_CORRECTNESS", result["evaluation"]["decision"])
        self.assertIsNone(result["campaign"]["accepted_candidate_id"])

    def test_no_improvement_budget_stops_campaign(self):
        campaign = self.campaign(rounds=5)
        for _ in range(2):
            candidate = self.control.campaign_append_candidate(campaign["campaign_id"], "No gain", "KERNEL_ONLY")
            evidence = self.execute({"latency_ms": 11.0, "protocol_digest": "operator-v1"}, candidate_id=candidate["candidate_id"])
            result = self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])
        self.assertEqual("STOPPED", result["campaign"]["status"])
        self.assertEqual("NO_IMPROVEMENT", result["campaign"]["stop_reason"])
        with self.assertRaises(ConflictError):
            self.control.campaign_append_candidate(campaign["campaign_id"], "over budget", "KERNEL_ONLY")

    def test_protocol_drift_is_rejected(self):
        campaign = self.campaign()
        candidate = self.control.campaign_append_candidate(campaign["campaign_id"], "Changed protocol", "KERNEL_ONLY")
        evidence = self.execute({"latency_ms": 8.0, "protocol_digest": "operator-v2"}, candidate_id=candidate["candidate_id"])
        with self.assertRaises(ConflictError):
            self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])

    def test_invalid_measurement_is_not_an_acceptance_or_round(self):
        campaign = self.campaign()
        candidate = self.control.campaign_append_candidate(campaign["campaign_id"], "Noisy result", "KERNEL_ONLY")
        evidence = self.execute({"latency_ms": 8.0, "protocol_digest": "operator-v1"}, candidate_id=candidate["candidate_id"])
        result = self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"], measurement_valid=False)
        self.assertEqual("MEASUREMENT_INVALID", result["evaluation"]["decision"])
        self.assertEqual(0, result["campaign"]["rounds"])


class RSIDemoTest(unittest.TestCase):
    def test_all_campaign_levels_have_independent_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_rsi_demo(os.path.join(directory, "demo.db"),
                                  os.path.join(directory, "workspaces"),
                                  dashboard_path=os.path.join(directory, "dashboard.html"))
            self.assertEqual("KERNEL_CANDIDATE_ACCEPTED", result["decisions"]["OPERATOR"])
            self.assertEqual("MODEL_CANDIDATE_ACCEPTED", result["decisions"]["MODEL"])
            self.assertEqual("REJECTED_NO_IMPROVEMENT", result["decisions"]["E2E"])
            self.assertTrue(os.path.exists(result["dashboard"]["path"]))


if __name__ == "__main__":
    unittest.main()
