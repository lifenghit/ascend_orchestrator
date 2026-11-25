"""Executable local demonstration; all hardware evidence is explicitly simulated."""

import os

from .control import ControlPlane


def _execute_ready(control):
    for _ in range(40):
        control.schedule()
        running = [s for s in control.store.all("steps") if s["status"] == "RUNNING"]
        if not running:
            return
        for step in running:
            lease = control.store.require("leases", step["lease_id"])
            control.run_step(step["step_id"], lease["lease_id"], lease["epoch"],
                             artifact_digests=["demo-artifact"] if step["resource"]["scope"] == "MULTI_NODE" else None)
    raise RuntimeError("demo DAG did not quiesce")


def run_demo(db_path="var/demo.db", workspace_root="var/demo-workspaces", reset=False):
    if os.path.exists(db_path) and not reset:
        raise FileExistsError("demo database already exists; use --reset to replace it: %s" % db_path)
    if reset:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except FileNotFoundError:
                pass
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)
    control = ControlPlane(db_path, workspace_root)
    try:
        for node in range(2):
            control.register_executor("node-%d" % node, [{"id": card} for card in range(2)],
                                      {"soc": "910C", "runtime_profile": "demo"})
        operator = control.create_task("开发并验证 AscendC 自定义算子")
        _execute_ready(control)

        rl_spec = control.register_rl_spec({
            "weight_exchange_contract": {"format": "safetensors", "version_handshake": "required"},
            "trajectory_contract": {"schema_version": "v1", "log_probability_fields": ["old_logp"]},
            "generation_contract": {"sampling_parameters": {"temperature": 1.0}},
            "integration_acceptance": {"weight_roundtrip": "REQUIRED", "trajectory_roundtrip": "REQUIRED",
                                        "rollout_train_closed_loop": "REQUIRED"},
        })
        rl_task = control.create_task("强化学习训练与推理适配", scenario="REINFORCEMENT_LEARNING",
                                      rl_spec_id=rl_spec["rl_spec_id"])
        control.freeze_rl_contract(rl_spec["rl_spec_id"])
        _execute_ready(control)
        integration = next(s for s in control.store.all("steps")
                           if s["task_id"] == rl_task["task_id"] and s["kind"] == "RL_INTEGRATION_VERIFY")
        group = control.create_resource_group(integration["step_id"], {
            "training": {"scope": "SINGLE_NODE", "cards_per_node": 2, "lease_class": "SERVICE"},
            "inference": {"scope": "SINGLE_NODE", "cards_per_node": 2, "lease_class": "SERVICE"},
        })
        control.schedule()
        group = control.store.require("resource_groups", group["resource_group_id"])
        lease = control.store.require("leases", group["lease_ids"][0])
        closed_loop = control.run_step(integration["step_id"], lease["lease_id"], lease["epoch"],
                                       contract_versions=["v1"], owner="rl-integration-owner",
                                       metrics={"weight_export": True, "weight_import": True,
                                                "trajectory_generation": True, "trajectory_consumption": True,
                                                "weight_resync": True, "rollout_train_rounds": 2,
                                                "simulated": True})
        _execute_ready(control)
        analysis = control.invoke_capability("msagent.offline.profiler-analysis",
                                              {"profiling_data": {"operator": "demo", "ratio": 0.4}})
        return {
            "operator_task": control.get_task(operator["task_id"])["state"],
            "rl_task": control.get_task(rl_task["task_id"])["state"],
            "rl_closed_loop_evidence": closed_loop["evidence"]["evidence_id"],
            "offline_capability": analysis["status"],
            "cluster": control.cluster_status(),
            "database": db_path,
            "note": "All executions are simulated; no real NPU correctness or performance is claimed.",
        }
    finally:
        control.close()
