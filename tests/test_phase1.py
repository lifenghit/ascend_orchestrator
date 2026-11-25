import os
import tempfile
import time
import unittest

from ascend_orchestrator import ControlPlane
from ascend_orchestrator.demo import run_demo
from ascend_orchestrator.models import ConflictError, ContractError, FencingError


class ControlPlaneTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.temp.name, "state.db")
        self.control = ControlPlane(self.db, os.path.join(self.temp.name, "workspaces"), lease_ttl=60)

    def tearDown(self):
        self.control.close()
        self.temp.cleanup()

    def add_node(self, name="node-0", cards=2, labels=None):
        return self.control.register_executor(name, [{"id": i} for i in range(cards)], labels or {"runtime": "a"})

    def run_ready(self, limit=30):
        for _ in range(limit):
            self.control.schedule()
            running = [s for s in self.control.store.all("steps") if s["status"] == "RUNNING"]
            if not running:
                return
            for step in running:
                self.control.run_step(step["step_id"], step["lease_id"],
                                      self.control.store.require("leases", step["lease_id"])["epoch"],
                                      artifact_digests=["same"] if step["resource"]["scope"] == "MULTI_NODE" else None)
        self.fail("DAG did not quiesce")

    def model_spec(self):
        return self.control.register_model_spec({
            "model": {"name": "demo", "source_revision": "r1", "framework": "pytorch",
                      "mode": "INFERENCE", "weights_digest": "w1", "config_digest": "c1"},
            "target": {"intended_hardware_scope": "MULTI_NODE"},
            "workload": {"input_shapes": [[1, 8]]},
        })

    def rl_spec(self):
        return self.control.register_rl_spec({
            "weight_exchange_contract": {"format": "safetensors", "layout": "v1"},
            "trajectory_contract": {"schema": "trajectory-v1"},
            "generation_contract": {"sampling": "v1"},
            "integration_acceptance": {"rollout_train_closed_loop": "REQUIRED"},
        })

    def test_single_cards_fill_without_duplicate_and_dynamic_label_selection(self):
        self.add_node(cards=2)
        task = self.control.create_task("优化 AscendC kernel")
        steps = self.control.add_steps(task["task_id"], [
            {"kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD", "node_selector": {"runtime": "a"}}}
            for _ in range(4)
        ])
        result = self.control.schedule()
        resources = [r for lease_id in result["granted_lease_ids"]
                     for r in self.control.store.require("leases", lease_id)["resources"] if ":npu-" in r]
        self.assertEqual(2, len(resources))
        self.assertEqual(2, len(set(resources)))
        self.assertFalse(self.control.avoidable_idle())
        self.add_node("node-1", 1, {"runtime": "b"})
        unmatched = self.control.add_steps(task["task_id"], [
            {"kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD", "node_selector": {"runtime": "missing"}}}
        ])[0]
        self.control.schedule()
        self.assertEqual("READY", self.control.store.require("steps", unmatched["step_id"])["status"])
        self.assertIn("node_selector", self.control.store.require("steps", unmatched["step_id"])["rejection_reason"])

    def test_dependencies_and_analysis_hold_no_npu(self):
        self.add_node()
        task = self.control.create_task("优化 AscendC 算子")
        self.control.schedule()
        analysis = next(s for s in self.control.store.all("steps") if s["kind"] == "ANALYZE")
        lease = self.control.store.require("leases", analysis["lease_id"])
        self.assertEqual([], lease["resources"])
        self.assertTrue(all(s["status"] == "BLOCKED" for s in task["steps"] if s["kind"] != "ANALYZE"))
        result = self.control.run_step(analysis["step_id"], lease["lease_id"], lease["epoch"])
        self.assertTrue(os.path.exists(result["execution"]["log_uri"]))
        self.control.schedule()
        implement = next(s for s in self.control.store.all("steps") if s["kind"] == "IMPLEMENT")
        self.assertEqual("RUNNING", implement["status"])

    def test_cordon_drain_lost_and_reconcile(self):
        node = self.add_node()
        self.control.set_node_lifecycle(node["node_id"], "CORDONED")
        task = self.control.create_task("优化 AscendC 算子")
        step = self.control.add_steps(task["task_id"], [{"kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD"}}])[0]
        self.control.schedule()
        self.assertEqual("READY", self.control.store.require("steps", step["step_id"])["status"])
        self.control.set_node_lifecycle(node["node_id"], "READY")
        self.control.set_node_lifecycle(node["node_id"], "LOST")
        recovered = self.control.set_node_lifecycle(node["node_id"], "READY")
        self.assertEqual("DRAINING", recovered["lifecycle"])
        ready = self.control.confirm_node_reconciled(node["node_id"])
        self.assertEqual("READY", ready["lifecycle"])
        self.assertGreater(ready["resource_version"], node["resource_version"])

    def test_reservation_allows_only_conservative_backfill(self):
        self.add_node(cards=2)
        task = self.control.create_task("优化 AscendC 算子")
        large = self.control.add_steps(task["task_id"], [{
            "kind": "DISTRIBUTED_VERIFY", "resource": {"scope": "SINGLE_NODE", "cards_per_node": 2}
        }])[0]
        start = time.time() + 100
        reservation = self.control.request_reservation(large["step_id"], start, guard_seconds=10)
        short = self.control.add_steps(task["task_id"], [{
            "kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD", "duration_seconds": 20}},
            {"kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD", "duration_seconds": 200}},
        ])
        self.control.schedule()
        self.assertEqual("RUNNING", self.control.store.require("steps", short[0]["step_id"])["status"])
        self.assertEqual("READY", self.control.store.require("steps", short[1]["step_id"])["status"])
        self.assertIn("blocking_leases", self.control.reservation_status(reservation["reservation_id"]))

    def test_model_minimal_path_artifact_reuse_and_drift(self):
        self.add_node("node-0", 1)
        self.add_node("node-1", 1)
        spec = self.model_spec()
        task = self.control.create_task("模型推理适配并做分布式验证", model_spec_id=spec["model_spec_id"],
                                        acceptance={"distributed_verification_required": True})
        kinds = [s["kind"] for s in task["steps"]]
        self.assertLess(kinds.index("COVERAGE_VERIFY"), kinds.index("DISTRIBUTED_VERIFY"))
        coverage = next(s for s in task["steps"] if s["kind"] == "COVERAGE_VERIFY")
        self.assertEqual("SINGLE_CARD", coverage["resource"]["scope"])
        conversion = {"entrypoint": "convert.py", "args": {"precision": "fp16"}}
        artifact = self.control.register_model_artifact(spec["model_spec_id"], conversion, "out-1")
        self.assertEqual(1, len(self.control.reusable_artifacts(spec["model_spec_id"], conversion)))
        self.assertTrue(self.control.validate_model_artifact(artifact["artifact_id"], conversion=conversion)["valid"])
        self.assertFalse(self.control.validate_model_artifact(artifact["artifact_id"], conversion={"entrypoint": "other"})["valid"])

    def test_missing_operator_subdag_blocks_only_dependents(self):
        self.add_node()
        task = self.control.create_task("模型适配和缺失算子修复", task_kind="MIXED")
        coverage = next(s for s in task["steps"] if s["kind"] == "COVERAGE_VERIFY")
        children = self.control.report_missing_operators(coverage["step_id"], ["custom_add"])
        self.assertEqual(2, len(children))
        functional = next(s for s in self.control.store.all("steps")
                          if s["kind"] == "FUNCTIONAL_VERIFY" and not s["metadata"].get("child_dag"))
        self.assertIn(children[-1]["step_id"], functional["dependencies"])

    def test_rl_parallel_branches_atomic_group_and_closed_loop(self):
        self.add_node("node-0", 2)
        self.add_node("node-1", 2)
        spec = self.rl_spec()
        task = self.control.create_task("强化学习 rollout 训练推理适配", scenario="REINFORCEMENT_LEARNING",
                                        rl_spec_id=spec["rl_spec_id"])
        self.assertEqual(1, len([s for s in task["steps"] if s["status"] == "READY"]))
        self.control.freeze_rl_contract(spec["rl_spec_id"])
        self.control.schedule()
        running = self.control.store.all("steps")
        self.assertEqual("RUNNING", next(s for s in running if s["kind"] == "TRAINING_ADAPT")["status"])
        self.assertEqual("RUNNING", next(s for s in running if s["kind"] == "INFERENCE_ADAPT")["status"])
        self.run_ready()
        integration = next(s for s in self.control.store.all("steps") if s["kind"] == "RL_INTEGRATION_VERIFY")
        self.assertEqual("READY", integration["status"])
        self.assertTrue(self.control.validate_rl_integration(integration["step_id"])["valid"])
        group = self.control.create_resource_group(integration["step_id"], {
            "training": {"scope": "SINGLE_NODE", "cards_per_node": 2, "lease_class": "SERVICE"},
            "inference": {"scope": "SINGLE_NODE", "cards_per_node": 2, "lease_class": "SERVICE"},
        })
        self.control.schedule()
        group = self.control.store.require("resource_groups", group["resource_group_id"])
        self.assertEqual("ACTIVE", group["status"])
        self.assertEqual(2, len(group["lease_ids"]))
        lease = self.control.store.require("leases", group["lease_ids"][0])
        self.control.heartbeat_lease(lease["lease_id"], lease["epoch"], {"round": 1})
        with self.assertRaises(ContractError):
            self.control.run_step(integration["step_id"], lease["lease_id"], lease["epoch"],
                                  contract_versions=["v1"], owner="rl-integration-owner")
        result = self.control.run_step(integration["step_id"], lease["lease_id"], lease["epoch"],
                                       contract_versions=["v1"], owner="rl-integration-owner",
                                       metrics={"weight_export": True, "weight_import": True,
                                                "trajectory_generation": True, "trajectory_consumption": True,
                                                "weight_resync": True, "rollout_train_rounds": 2})
        self.assertEqual("SUCCEEDED", result["execution"]["status"])
        evidence = result["evidence"]
        self.assertEqual("RL_INTEGRATION", evidence["task_role"])

    def test_atomic_group_does_not_partially_allocate(self):
        self.add_node("node-0", 1)
        spec = self.rl_spec()
        task = self.control.create_task("强化学习适配", scenario="REINFORCEMENT_LEARNING", rl_spec_id=spec["rl_spec_id"])
        self.control.freeze_rl_contract(spec["rl_spec_id"])
        self.run_ready()
        integration = next(s for s in self.control.store.all("steps") if s["kind"] == "RL_INTEGRATION_VERIFY")
        group = self.control.create_resource_group(integration["step_id"], {
            "training": {"scope": "SINGLE_NODE", "cards_per_node": 1},
            "inference": {"scope": "SINGLE_NODE", "cards_per_node": 1},
        })
        self.control.schedule()
        group = self.control.store.require("resource_groups", group["resource_group_id"])
        self.assertEqual("PENDING", group["status"])
        self.assertEqual([], group["lease_ids"])
        self.assertEqual(0, len([l for l in self.control.store.all("leases") if l["status"] == "ACTIVE"]))

    def test_stale_lease_fenced_until_cleanup_then_rejected(self):
        self.add_node(cards=1)
        task = self.control.create_task("优化 AscendC 算子")
        step = self.control.add_steps(task["task_id"], [{"kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD"}}])[0]
        self.control.schedule()
        step = self.control.store.require("steps", step["step_id"])
        old = self.control.store.require("leases", step["lease_id"])
        with self.control.store.transaction():
            old["expires_at"] = time.time() - 1
            self.control.store.put("leases", old)
        self.control.schedule()
        old = self.control.store.require("leases", old["lease_id"])
        self.assertEqual("FENCED", old["status"])
        with self.assertRaises(FencingError):
            self.control.run_step(step["step_id"], old["lease_id"], old["epoch"])
        self.assertEqual(0, len([l for l in self.control.store.all("leases")
                                 if l["status"] == "ACTIVE" and any(":npu-" in r for r in l["resources"])]))
        self.control.confirm_cleanup(old["lease_id"], old["epoch"])
        self.control.schedule()
        new = self.control.store.require("leases", self.control.store.require("steps", step["step_id"])["lease_id"])
        self.assertGreater(new["epoch"], old["epoch"])
        with self.assertRaises(FencingError):
            self.control.heartbeat_lease(old["lease_id"], old["epoch"])

    def test_recovery_and_idempotency(self):
        self.add_node()
        request_id = "fixed-request"
        first = self.control.create_task("优化 AscendC 算子", request_id=request_id)
        second = self.control.create_task("优化 AscendC 算子", request_id=request_id)
        self.assertEqual(first["task_id"], second["task_id"])
        with self.assertRaises(ConflictError):
            self.control.create_task("另一个任务", request_id=request_id)
        self.control.schedule()
        status = self.control.recover()
        self.assertTrue(all(n["needs_reconciliation"] for n in status["nodes"]))
        self.assertTrue(all(l["status"] != "ACTIVE" for l in self.control.store.all("leases")))

    def test_multi_node_artifact_and_single_owner_gates(self):
        self.add_node("node-0", 1)
        self.add_node("node-1", 1)
        task = self.control.create_task("优化 AscendC 算子")
        step = self.control.add_steps(task["task_id"], [{
            "kind": "DISTRIBUTED_VERIFY", "resource": {"scope": "MULTI_NODE", "nodes": 2, "cards_per_node": 1}
        }])[0]
        self.control.schedule()
        step = self.control.store.require("steps", step["step_id"])
        lease = self.control.store.require("leases", step["lease_id"])
        with self.assertRaises(FencingError):
            self.control.run_step(step["step_id"], lease["lease_id"], lease["epoch"], artifact_digests=["a", "b"])
        with self.assertRaises(FencingError):
            self.control.run_step(step["step_id"], lease["lease_id"], lease["epoch"], artifact_digests=["a"], owner="other")
        result = self.control.run_step(step["step_id"], lease["lease_id"], lease["epoch"],
                                       artifact_digests=["a", "a"], owner="distributed-owner")
        self.assertEqual("SUCCEEDED", result["execution"]["status"])

    def test_adapters_return_evidence_and_explicit_failure(self):
        capabilities = self.control.list_capabilities()
        self.assertEqual(2, len(capabilities))
        result = self.control.invoke_capability("msagent.offline.profiler-analysis", {"profiling_data": {"hotspot": "add"}})
        self.assertEqual("SUCCEEDED", result["status"])
        failure = self.control.invoke_capability("msagent.offline.profiler-analysis",
                                                  {"profiling_data": {}, "simulate_error": True})
        self.assertEqual("FAILED", failure["status"])
        workspace = os.path.join(self.temp.name, "candidate")
        patch = self.control.invoke_capability("cannbot.controlled.patch-example", {"workspace": workspace})
        self.assertTrue(os.path.exists(patch["patch"]))
        self.assertIn("build", patch["required_next_tests"])

    def test_quiet_window_blocks_then_restores_work_conserving_scheduling(self):
        self.add_node(cards=2)
        task = self.control.create_task("优化 AscendC 算子")
        steps = self.control.add_steps(task["task_id"], [
            {"kind": "FUNCTIONAL_VERIFY", "priority": 100,
             "resource": {"scope": "SINGLE_NODE", "cards_per_node": 1,
                          "performance_isolation": "NODE_SET_QUIET"}},
            {"kind": "FUNCTIONAL_VERIFY", "priority": 1,
             "resource": {"scope": "SINGLE_CARD"}},
        ])
        self.control.schedule()
        quiet = self.control.store.require("steps", steps[0]["step_id"])
        normal = self.control.store.require("steps", steps[1]["step_id"])
        self.assertEqual("RUNNING", quiet["status"])
        self.assertEqual("READY", normal["status"])
        lease = self.control.store.require("leases", quiet["lease_id"])
        self.control.run_step(quiet["step_id"], lease["lease_id"], lease["epoch"])
        self.control.schedule()
        self.assertEqual("RUNNING", self.control.store.require("steps", normal["step_id"])["status"])

    def test_stale_executor_inventory_is_not_allocated(self):
        node = self.add_node(cards=1)
        with self.control.store.transaction():
            node["last_heartbeat_at"] = time.time() - 1000
            self.control.store.put("nodes", node)
        task = self.control.create_task("优化 AscendC 算子")
        step = self.control.add_steps(task["task_id"], [
            {"kind": "FUNCTIONAL_VERIFY", "resource": {"scope": "SINGLE_CARD"}}
        ])[0]
        self.control.schedule()
        self.assertEqual("READY", self.control.store.require("steps", step["step_id"])["status"])
        self.control.heartbeat_executor(node["node_id"])
        self.control.schedule()
        self.assertEqual("RUNNING", self.control.store.require("steps", step["step_id"])["status"])

    def test_build_slots_are_independent_from_npu_cards(self):
        self.add_node(cards=1)
        task = self.control.create_task("优化 AscendC 算子")
        steps = self.control.add_steps(task["task_id"], [
            {"kind": "MODEL_ADAPT", "resource": {"scope": "CPU_ONLY"}} for _ in range(3)
        ])
        self.control.schedule()
        statuses = [self.control.store.require("steps", s["step_id"])["status"] for s in steps]
        self.assertEqual(2, statuses.count("RUNNING"))
        self.assertEqual(1, statuses.count("READY"))

    def test_rl_contract_version_is_enforced_at_joint_execution(self):
        self.add_node("node-0", 1)
        self.add_node("node-1", 1)
        spec = self.rl_spec()
        task = self.control.create_task("强化学习适配", scenario="REINFORCEMENT_LEARNING",
                                        rl_spec_id=spec["rl_spec_id"])
        self.control.freeze_rl_contract(spec["rl_spec_id"])
        self.run_ready()
        integration = next(s for s in self.control.store.all("steps") if s["kind"] == "RL_INTEGRATION_VERIFY")
        self.control.update_rl_contract(spec["rl_spec_id"], {"trajectory_contract": {"schema": "trajectory-v2"}})
        group = self.control.create_resource_group(integration["step_id"], {
            "training": {"scope": "SINGLE_NODE", "cards_per_node": 1},
            "inference": {"scope": "SINGLE_NODE", "cards_per_node": 1},
        })
        self.control.schedule()
        group = self.control.store.require("resource_groups", group["resource_group_id"])
        lease = self.control.store.require("leases", group["lease_ids"][0])
        metrics = {"weight_export": True, "weight_import": True, "trajectory_generation": True,
                   "trajectory_consumption": True, "weight_resync": True, "rollout_train_rounds": 2}
        with self.assertRaises(FencingError):
            self.control.run_step(integration["step_id"], lease["lease_id"], lease["epoch"],
                                  contract_versions=["v1"], owner="rl-integration-owner", metrics=metrics)
        result = self.control.run_step(integration["step_id"], lease["lease_id"], lease["epoch"],
                                       contract_versions=["v2"], owner="rl-integration-owner", metrics=metrics)
        self.assertEqual("v2", result["evidence"]["contract_version"])


class DemoTest(unittest.TestCase):
    def test_demo_preserves_database_unless_reset_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "demo.db")
            workspace = os.path.join(directory, "workspaces")
            result = run_demo(db_path, workspace)
            self.assertEqual("SUCCEEDED", result["rl_task"])
            with self.assertRaises(FileExistsError):
                run_demo(db_path, workspace)
            reset_result = run_demo(db_path, workspace, reset=True)
            self.assertEqual("SUCCEEDED", reset_result["operator_task"])


if __name__ == "__main__":
    unittest.main()
