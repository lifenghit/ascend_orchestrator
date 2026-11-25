"""Public contracts. Documents are stored as JSON so phase-two fields remain additive."""

import hashlib
import json
import time
import uuid


class ContractError(ValueError):
    pass


class NotFoundError(ContractError):
    pass


class ConflictError(ContractError):
    pass


class FencingError(ContractError):
    pass


def now():
    return time.time()


def new_id(prefix):
    return "%s-%s" % (prefix, uuid.uuid4().hex[:12])


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    if isinstance(value, (dict, list)):
        value = canonical(value)
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def resource_spec(scope="NONE", nodes=0, cards_per_node=0, isolation=None,
                  performance_isolation="NONE", node_selector=None,
                  duration_seconds=300, lease_class="STANDARD", **extra):
    scopes = {"NONE", "CPU_ONLY", "SINGLE_CARD", "SINGLE_NODE", "MULTI_NODE"}
    if scope not in scopes:
        raise ContractError("invalid resource scope: %s" % scope)
    if scope == "SINGLE_CARD":
        nodes, cards_per_node = 1, 1
    elif scope == "SINGLE_NODE":
        nodes = 1
    elif scope in {"NONE", "CPU_ONLY"}:
        nodes, cards_per_node = 0, 0
    if scope in {"SINGLE_NODE", "MULTI_NODE"} and (nodes < 1 or cards_per_node < 1):
        raise ContractError("node and card counts must be positive")
    if scope == "MULTI_NODE" and nodes < 2:
        raise ContractError("MULTI_NODE requires at least two nodes")
    if isolation is None:
        isolation = "SHARED_CPU" if scope in {"NONE", "CPU_ONLY"} else "DEVICE_EXCLUSIVE"
    result = {
        "scope": scope,
        "nodes": nodes,
        "cards_per_node": cards_per_node,
        "isolation": isolation,
        "performance_isolation": performance_isolation,
        "node_selector": node_selector or {},
        "duration_seconds": max(1, int(duration_seconds)),
        "lease_class": lease_class,
    }
    result.update(extra)
    return result


def task_document(task_kind, objective, scenario="GENERAL", priority=50,
                  acceptance=None, budget=None, model_spec_id=None, rl_spec_id=None):
    if task_kind not in {"OPERATOR_DEVELOPMENT", "MODEL_ADAPTATION", "MIXED"}:
        raise ContractError("invalid task kind")
    if scenario not in {"GENERAL", "REINFORCEMENT_LEARNING"}:
        raise ContractError("invalid scenario")
    return {
        "task_id": new_id("task"),
        "task_kind": task_kind,
        "scenario": scenario,
        "objective": objective,
        "priority": int(priority),
        "current_phase": "ANALYZE",
        "owner_agent": "ascend-orchestrator",
        "model_spec_id": model_spec_id,
        "rl_spec_id": rl_spec_id,
        "acceptance": acceptance or {},
        "budget": budget or {"max_exclusive_minutes": 120},
        "state": "ACTIVE",
        "created_at": now(),
        "updated_at": now(),
        "campaign_extensions": {},
    }


def step_document(task_id, kind, resource=None, dependencies=None, runner=None,
                  priority=50, estimated_duration_seconds=None, role="GENERAL",
                  contract_version=None, owner=None, workspace=None, metadata=None):
    resource = resource_spec(**resource) if resource else resource_spec()
    estimate = estimated_duration_seconds or resource.get("duration_seconds", 300)
    return {
        "step_id": new_id("step"),
        "task_id": task_id,
        "kind": kind,
        "dependencies": list(dependencies or []),
        "workspace": workspace or {"base_commit": "prototype", "patch_set": []},
        "execution": {
            "runner": runner or default_runner(resource["scope"]),
            "timeout_seconds": max(1, int(resource.get("duration_seconds", 300))),
        },
        "resource": resource,
        "scheduling": {
            "priority": int(priority),
            "preemptible": True,
            "checkpointable": False,
            "estimated_duration_seconds": max(1, int(estimate)),
        },
        "role": role,
        "contract_version": contract_version,
        "owner": owner or ("distributed-owner" if resource["scope"] == "MULTI_NODE" else "worker"),
        "metadata": metadata or {},
        "status": "DRAFT",
        "created_at": now(),
        "ready_at": None,
        "started_at": None,
        "finished_at": None,
        "lease_id": None,
        "resource_group_id": None,
        "attempt": 0,
        "outputs": [],
        "rejection_reason": None,
        "campaign_extensions": {},
    }


def default_runner(scope):
    return {
        "NONE": "analysis-runner",
        "CPU_ONLY": "build-runner",
        "SINGLE_CARD": "single-card-test-runner",
        "SINGLE_NODE": "single-node-test-runner",
        "MULTI_NODE": "multi-node-test-runner",
    }[scope]


def lease_document(task_id, step_id, owner, resources, ttl, epoch,
                   lease_class="STANDARD", resource_group_id=None, role=None):
    created = now()
    return {
        "lease_id": new_id("lease"),
        "epoch": epoch,
        "task_id": task_id,
        "step_id": step_id,
        "resources": resources,
        "granted_at": created,
        "expires_at": created + ttl,
        "heartbeat_timeout_seconds": ttl,
        "last_heartbeat_at": created,
        "owner": owner,
        "status": "ACTIVE",
        "lease_class": lease_class,
        "resource_group_id": resource_group_id,
        "role": role,
        "cleanup_confirmed": False,
    }


def evidence_document(task_id, step_id, source, correctness="PASSED",
                      candidate_id=None, role="GENERAL", contract_version=None,
                      performance=None, environment=None, logs_uri=None,
                      limitations=None, code=None, artifact_digest=None):
    return {
        "evidence_id": new_id("ev"),
        "task_id": task_id,
        "step_id": step_id,
        "candidate_id": candidate_id,
        "task_role": role,
        "contract_version": contract_version,
        "source": source,
        "code": code or {"base_commit": "prototype", "patch_digest": digest([])},
        "artifact_digest": artifact_digest,
        "environment_fingerprint": environment or {},
        "correctness": {"status": correctness},
        "performance": performance,
        "raw_logs_uri": logs_uri,
        "limitations": limitations or ["Simulated runner evidence; not real NPU validation"],
        "created_at": now(),
    }
