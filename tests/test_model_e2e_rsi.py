import os
import tempfile
import time
import unittest

from ascend_orchestrator import ControlPlane
from ascend_orchestrator.models import ConflictError, ContractError


class ModelE2ERSITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.control = ControlPlane(os.path.join(self.temp.name, "state.db"),
                                    os.path.join(self.temp.name, "workspaces"))
        self.control.register_executor("node-0", [{"id": 0}, {"id": 1}])
        self.task = self.control.create_task("模型推理适配与算子优化", task_kind="MIXED")

    def tearDown(self):
        self.control.close()
        self.temp.cleanup()

    def execute(self, metrics, candidate_id=None, role="GENERAL", contract_version=None, correctness="PASSED"):
        step = self.control.add_steps(self.task["task_id"], [{
            "kind": "INTEGRATION_VERIFY", "resource": {"scope": "SINGLE_CARD"},
            "role": role, "contract_version": contract_version,
        }])[0]
        self.control.schedule()
        step = self.control.store.require("steps", step["step_id"])
        lease = self.control.store.require("leases", step["lease_id"])
        return self.control.run_step(step["step_id"], lease["lease_id"], lease["epoch"],
                                     metrics=metrics, candidate_id=candidate_id,
                                     correctness=correctness)["evidence"]

    def create_campaign(self, level, role="GENERAL", contract_version=None, parent=None, parent_candidate=None):
        metrics = {"latency_ms": 10.0, "protocol_digest": level + "-protocol"}
        if level == "MODEL":
            metrics["model_signature"] = "model-source-and-derived-v1"
        if level == "E2E":
            metrics.update({"workload_digest": "requests-v1", "deployment_signature": "service-v1"})
        baseline = self.execute(metrics, role=role, contract_version=contract_version)
        return self.control.campaign_create(
            self.task["task_id"], level, baseline["evidence_id"],
            {"primary_metric": "latency_ms", "direction": "MINIMIZE", "correctness_gate": "REQUIRED",
             "min_improvement": 0.1, "retest_on_noise": False},
            {"max_rounds": 5, "max_no_improvement_rounds": 3, "max_wall_time_minutes": 10},
            role=role, parent_campaign_id=parent, parent_candidate_id=parent_candidate,
        )

    def test_model_requires_integration_and_preserves_action_space(self):
        campaign = self.create_campaign("MODEL")
        candidate = self.control.campaign_append_candidate(campaign["campaign_id"], "Fuse graph nodes", "GRAPH_ONLY")
        metrics = {"latency_ms": 8.0, "protocol_digest": "MODEL-protocol",
                   "model_signature": "model-source-and-derived-v1"}
        evidence = self.execute(metrics, candidate_id=candidate["candidate_id"])
        result = self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])
        self.assertEqual("REJECTED_INTEGRATION", result["evaluation"]["decision"])
        candidate2 = self.control.campaign_append_candidate(campaign["campaign_id"], "Verified graph fusion", "GRAPH_ONLY")
        metrics["integration_verified"] = True
        evidence2 = self.execute(metrics, candidate_id=candidate2["candidate_id"])
        result2 = self.control.campaign_record_evaluation(candidate2["candidate_id"], evidence2["evidence_id"])
        self.assertEqual("MODEL_CANDIDATE_ACCEPTED", result2["evaluation"]["decision"])
        with self.assertRaises(ContractError):
            self.control.campaign_append_candidate(campaign["campaign_id"], "invalid action", "MODEL_ONLY")

    def test_e2e_has_final_veto_over_local_speedup(self):
        campaign = self.create_campaign("E2E")
        candidate = self.control.campaign_append_candidate(campaign["campaign_id"], "Kernel helps microbenchmark", "KERNEL_ONLY")
        evidence = self.execute({"latency_ms": 12.0, "protocol_digest": "E2E-protocol",
                                 "workload_digest": "requests-v1", "deployment_signature": "service-v1",
                                 "e2e_validated": True}, candidate_id=candidate["candidate_id"])
        result = self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])
        self.assertEqual("REJECTED_NO_IMPROVEMENT", result["evaluation"]["decision"])
        self.assertIsNone(result["campaign"]["accepted_candidate_id"])

    def test_subcampaign_requires_confirmed_bottleneck(self):
        parent = self.create_campaign("MODEL")
        candidate = self.control.campaign_append_candidate(parent["campaign_id"], "Possible hotspot", "OPERATOR_SET")
        with self.assertRaises(ConflictError):
            self.create_campaign("OPERATOR", parent=parent["campaign_id"], parent_candidate=candidate["candidate_id"])
        confirmed = self.control.campaign_append_candidate(parent["campaign_id"], "Profiled hotspot", "OPERATOR_SET",
                                                            payload={"bottleneck_confirmed": True})
        child = self.create_campaign("OPERATOR", parent=parent["campaign_id"], parent_candidate=confirmed["candidate_id"])
        self.assertEqual(1, child["depth"])

    def test_rl_joint_batch_filters_unscreened_and_incompatible_candidates(self):
        training = self.create_campaign("MODEL", role="TRAINING", contract_version="v1")
        inference = self.create_campaign("MODEL", role="INFERENCE", contract_version="v1")
        accepted = []
        for campaign, role in ((training, "TRAINING"), (inference, "INFERENCE")):
            candidate = self.control.campaign_append_candidate(campaign["campaign_id"], role + " candidate", "RUNTIME_ONLY",
                                                                payload={"contract_version": "v1"})
            evidence = self.execute({"latency_ms": 8.0, "protocol_digest": "MODEL-protocol",
                                     "model_signature": "model-source-and-derived-v1", "integration_verified": True},
                                    candidate_id=candidate["candidate_id"], role=role, contract_version="v1")
            self.control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])
            accepted.append(candidate["candidate_id"])
        e2e = self.create_campaign("E2E", role="RL_INTEGRATION", contract_version="v1")
        batch = self.control.campaign_joint_batch(e2e["campaign_id"], [accepted[0]], [accepted[1]], max_combinations=1)
        self.assertEqual(1, len(batch))
        unscreened = self.control.campaign_append_candidate(inference["campaign_id"], "not evaluated", "RUNTIME_ONLY",
                                                             payload={"contract_version": "v1"})
        with self.assertRaises(ConflictError):
            self.control.campaign_joint_batch(e2e["campaign_id"], [accepted[0]], [unscreened["candidate_id"]])

    def test_batch_window_close_restores_resources_and_checkpoints(self):
        campaign = self.create_campaign("E2E")
        steps = self.control.add_steps(self.task["task_id"], [{
            "kind": "E2E_OPTIMIZE", "resource": {"scope": "SINGLE_NODE", "cards_per_node": 1,
                                                   "performance_isolation": "GLOBAL_QUIET"}}
        ])
        window = self.control.measurement_window_create(campaign["campaign_id"], [steps[0]["step_id"]],
                                                         time.time() + 100, 60)
        closed = self.control.measurement_window_close(window["window_id"])
        self.assertEqual("CLOSED", closed["status"])
        reservation = self.control.store.require("reservations", window["reservation_ids"][0])
        self.assertEqual("CANCELLED", reservation["status"])
        checkpoint = self.control.store.require("campaigns", campaign["campaign_id"])["checkpoint"]
        self.assertEqual(window["window_id"], checkpoint["window_id"])

    def test_campaign_stop_cancels_unused_measurement_window(self):
        campaign = self.create_campaign("E2E")
        step = self.control.add_steps(self.task["task_id"], [{
            "kind": "E2E_OPTIMIZE", "resource": {"scope": "SINGLE_NODE", "cards_per_node": 1,
                                                   "performance_isolation": "GLOBAL_QUIET"}}
        ])[0]
        window = self.control.measurement_window_create(campaign["campaign_id"], [step["step_id"]],
                                                         time.time() + 100, 60)
        self.control.campaign_stop(campaign["campaign_id"], reason="TEST_STOP")
        self.assertEqual("CANCELLED", self.control.store.require("windows", window["window_id"])["status"])
        reservation = self.control.store.require("reservations", window["reservation_ids"][0])
        self.assertEqual("CANCELLED", reservation["status"])


if __name__ == "__main__":
    unittest.main()
