"""Phase-two/three RSI demonstration with operator, model and E2E decisions."""

import os

from .control import ControlPlane


def _prepare(db_path, reset):
    if os.path.exists(db_path) and not reset:
        raise FileExistsError("RSI demo database already exists; use --reset to replace it: %s" % db_path)
    if reset:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except FileNotFoundError:
                pass
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)


def run_rsi_demo(db_path="var/rsi-demo.db", workspace_root="var/rsi-demo-workspaces", reset=False,
                 dashboard_path="var/rsi-dashboard.html"):
    _prepare(db_path, reset)
    control = ControlPlane(db_path, workspace_root)
    try:
        control.register_executor("node-0", [{"id": 0}, {"id": 1}], {"runtime_profile": "demo"})
        task = control.create_task("模型与 AscendC 算子端到端优化", task_kind="MIXED")

        def execute(metrics, candidate_id=None):
            step = control.add_steps(task["task_id"], [{
                "kind": "INTEGRATION_VERIFY", "resource": {"scope": "SINGLE_CARD"}
            }])[0]
            control.schedule()
            step = control.store.require("steps", step["step_id"])
            lease = control.store.require("leases", step["lease_id"])
            return control.run_step(step["step_id"], lease["lease_id"], lease["epoch"],
                                    metrics=metrics, candidate_id=candidate_id)["evidence"]

        def campaign(level):
            metrics = {"latency_ms": 10.0, "protocol_digest": level + "-v1"}
            if level == "MODEL":
                metrics["model_signature"] = "model-v1"
            if level == "E2E":
                metrics.update({"workload_digest": "workload-v1", "deployment_signature": "deployment-v1"})
            baseline = execute(metrics)
            return control.campaign_create(
                task["task_id"], level, baseline["evidence_id"],
                {"primary_metric": "latency_ms", "direction": "MINIMIZE", "correctness_gate": "REQUIRED",
                 "min_improvement": 0.1, "measurement_noise": 0.05, "retest_on_noise": False},
                {"max_rounds": 5, "max_no_improvement_rounds": 3, "max_wall_time_minutes": 10},
            )

        decisions = {}
        for level, action, value in (("OPERATOR", "KERNEL_ONLY", 8.0),
                                     ("MODEL", "GRAPH_ONLY", 8.5),
                                     ("E2E", "KERNEL_ONLY", 11.0)):
            current = campaign(level)
            candidate = control.campaign_append_candidate(current["campaign_id"], level + " demo hypothesis", action)
            metrics = {"latency_ms": value, "protocol_digest": level + "-v1"}
            if level == "MODEL":
                metrics.update({"model_signature": "model-v1", "integration_verified": True})
            if level == "E2E":
                metrics.update({"workload_digest": "workload-v1", "deployment_signature": "deployment-v1",
                                "e2e_validated": True})
            evidence = execute(metrics, candidate["candidate_id"])
            result = control.campaign_record_evaluation(candidate["candidate_id"], evidence["evidence_id"])
            control.campaign_checkpoint(current["campaign_id"])
            decisions[level] = result["evaluation"]["decision"]
        dashboard = control.dashboard_generate(dashboard_path)
        return {"task_id": task["task_id"], "decisions": decisions, "dashboard": dashboard,
                "metrics": control.metrics_snapshot(),
                "note": "All campaign measurements are simulated; no real NPU speedup is claimed."}
    finally:
        control.close()
