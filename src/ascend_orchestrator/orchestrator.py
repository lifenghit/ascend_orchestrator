"""Automatic track detection and minimal-viable-resource DAG templates."""

from .models import ContractError, resource_spec, step_document


PHASE_BY_KIND = {
    "ANALYZE": "ANALYZE",
    "RL_CONTRACT_FREEZE": "RL_CONTRACT_FREEZE",
    "ENVIRONMENT_PREPARE": "ENVIRONMENT_PREPARE",
    "MODEL_ADAPT": "MODEL_ADAPT",
    "TRAINING_ADAPT": "TRAINING_ADAPT",
    "INFERENCE_ADAPT": "INFERENCE_ADAPT",
    "COVERAGE_VERIFY": "COVERAGE_VERIFY",
    "IMPLEMENT": "IMPLEMENT",
    "FUNCTIONAL_VERIFY": "FUNCTIONAL_VERIFY",
    "DISTRIBUTED_VERIFY": "DISTRIBUTED_VERIFY",
    "RL_INTEGRATION_VERIFY": "RL_INTEGRATION_VERIFY",
    "INTEGRATION_VERIFY": "INTEGRATION_VERIFY",
    "FINAL_REGRESSION": "FINAL_REGRESSION",
    "OPERATOR_IMPLEMENT": "IMPLEMENT",
}


def classify_task(objective, scenario=None):
    text = objective.lower()
    if scenario == "REINFORCEMENT_LEARNING" or any(k in text for k in ("强化学习", "reinforcement", "rollout", " rl ")):
        return "MODEL_ADAPTATION", "REINFORCEMENT_LEARNING"
    model_words = ("模型", "model", "训练", "推理", "训练适配", "inference", "training")
    operator_words = ("算子", "operator", "kernel", "ascendc")
    if any(k in text for k in model_words) and any(k in text for k in operator_words):
        return "MIXED", "GENERAL"
    if any(k in text for k in model_words):
        return "MODEL_ADAPTATION", "GENERAL"
    return "OPERATOR_DEVELOPMENT", "GENERAL"


def initial_steps(task, model_spec=None, rl_spec=None):
    task_id = task["task_id"]
    if task["scenario"] == "REINFORCEMENT_LEARNING":
        if not rl_spec:
            raise ContractError("RL task requires an RLSpec")
        contract = step_document(task_id, "RL_CONTRACT_FREEZE")
        train_adapt = step_document(task_id, "TRAINING_ADAPT", resource_spec("CPU_ONLY"),
                                    [contract["step_id"]], role="TRAINING")
        infer_adapt = step_document(task_id, "INFERENCE_ADAPT", resource_spec("CPU_ONLY"),
                                    [contract["step_id"]], role="INFERENCE")
        train_verify = step_document(task_id, "FUNCTIONAL_VERIFY", resource_spec("SINGLE_CARD"),
                                     [train_adapt["step_id"]], role="TRAINING",
                                     contract_version=rl_spec["contract_version"])
        infer_verify = step_document(task_id, "FUNCTIONAL_VERIFY", resource_spec("SINGLE_CARD"),
                                     [infer_adapt["step_id"]], role="INFERENCE",
                                     contract_version=rl_spec["contract_version"])
        integrate = step_document(task_id, "RL_INTEGRATION_VERIFY", resource_spec(),
                                  [train_verify["step_id"], infer_verify["step_id"]],
                                  role="RL_INTEGRATION", owner="rl-integration-owner",
                                  contract_version=rl_spec["contract_version"])
        final = step_document(task_id, "FINAL_REGRESSION", resource_spec("SINGLE_CARD"),
                              [integrate["step_id"]], role="RL_INTEGRATION")
        return [contract, train_adapt, infer_adapt, train_verify, infer_verify, integrate, final]
    analyze = step_document(task_id, "ANALYZE")
    if task["task_kind"] == "OPERATOR_DEVELOPMENT":
        implement = step_document(task_id, "IMPLEMENT", dependencies=[analyze["step_id"]])
        verify = step_document(task_id, "FUNCTIONAL_VERIFY", resource_spec("SINGLE_CARD"),
                               [implement["step_id"]])
        final = step_document(task_id, "FINAL_REGRESSION", resource_spec("SINGLE_CARD"),
                              [verify["step_id"]])
        return [analyze, implement, verify, final]
    prepare = step_document(task_id, "ENVIRONMENT_PREPARE", resource_spec("CPU_ONLY"),
                            [analyze["step_id"]])
    adapt = step_document(task_id, "MODEL_ADAPT", resource_spec("CPU_ONLY"),
                          [prepare["step_id"]])
    coverage = step_document(task_id, "COVERAGE_VERIFY", resource_spec("SINGLE_CARD"),
                             [adapt["step_id"]])
    functional = step_document(task_id, "FUNCTIONAL_VERIFY", resource_spec("SINGLE_CARD"),
                               [coverage["step_id"]])
    steps = [analyze, prepare, adapt, coverage, functional]
    previous = functional
    if task["acceptance"].get("distributed_verification_required"):
        intended = (model_spec or {}).get("target", {}).get("intended_hardware_scope", "SINGLE_NODE")
        nodes = 2 if intended == "MULTI_NODE" else 1
        scope = "MULTI_NODE" if nodes > 1 else "SINGLE_NODE"
        distributed = step_document(task_id, "DISTRIBUTED_VERIFY",
                                    resource_spec(scope, nodes=nodes, cards_per_node=1,
                                                  isolation="NODE_SET_EXCLUSIVE"),
                                    [previous["step_id"]], owner="distributed-owner")
        steps.append(distributed)
        previous = distributed
    final = step_document(task_id, "FINAL_REGRESSION", resource_spec("SINGLE_CARD"),
                          [previous["step_id"]])
    steps.append(final)
    return steps


def operator_subdag(task_id, missing_operators, parent_step_id, independent_step_id=None):
    if not missing_operators:
        raise ContractError("at least one missing operator is required")
    steps = []
    previous = parent_step_id
    for name in sorted(set(missing_operators)):
        implement = step_document(task_id, "OPERATOR_IMPLEMENT", dependencies=[previous],
                                  metadata={"operator": name, "child_dag": True})
        verify = step_document(task_id, "FUNCTIONAL_VERIFY", resource_spec("SINGLE_CARD"),
                               [implement["step_id"]], metadata={"operator": name, "child_dag": True})
        steps.extend([implement, verify])
        previous = verify["step_id"]
    return steps, previous


def update_phase(task, steps):
    active = [s for s in steps if s["status"] not in {"SUCCEEDED", "CANCELLED"}]
    if not active:
        task["current_phase"] = "DONE"
        task["state"] = "SUCCEEDED"
        return
    phases = [PHASE_BY_KIND.get(s["kind"], task["current_phase"]) for s in active]
    if "RL_INTEGRATION_VERIFY" in phases and all(s["status"] == "READY" for s in active if s["kind"] == "RL_INTEGRATION_VERIFY"):
        task["current_phase"] = "RL_INTEGRATION_VERIFY"
    elif "TRAINING_ADAPT" in phases and "INFERENCE_ADAPT" in phases:
        task["current_phase"] = "TRAINING_ADAPT | INFERENCE_ADAPT"
    else:
        task["current_phase"] = phases[0]
