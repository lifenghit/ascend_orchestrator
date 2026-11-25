"""Phase-one control-plane facade. All mutations are transactional and idempotent."""

import os

from .adapters import AdapterRegistry
from .models import (ConflictError, ContractError, FencingError,
                     digest, evidence_document, new_id, now, resource_spec,
                     step_document, task_document)
from .orchestrator import classify_task, initial_steps, operator_subdag, update_phase
from .observability import ObservabilityManager
from .scheduler import Scheduler
from .rsi import RSIManager
from .store import Store
from .windows import WindowManager


class ControlPlane:
    def __init__(self, db_path="ascend.db", workspace_root="var/workspaces",
                 max_agent_workers=8, lease_ttl=60):
        self.store = Store(db_path)
        self.workspace_root = os.path.abspath(workspace_root)
        self.scheduler = Scheduler(self.store, max_agent_workers, lease_ttl)
        tuning = self.store.meta("scheduler_tuning", {})
        self.scheduler.max_agent_workers = tuning.get("max_agent_workers", max_agent_workers)
        self.adapters = AdapterRegistry(self)
        self.rsi = RSIManager(self)
        self.windows = WindowManager(self)
        self.observability = ObservabilityManager(self)
        os.makedirs(self.workspace_root, exist_ok=True)

    def close(self):
        self.store.close()

    @staticmethod
    def _request(request_id):
        return request_id or new_id("req")

    def _write(self, request_id, operation, payload, callback):
        return self.store.write_idempotent(self._request(request_id), operation, payload, callback)

    def register_executor(self, node_id, npus, labels=None, build_slots=2,
                          health="HEALTHY", request_id=None, **extra):
        payload = {"node_id": node_id, "npus": npus, "labels": labels or {}, "build_slots": build_slots,
                   "health": health, "extra": extra}
        def write():
            if health != "HEALTHY" or not npus:
                raise ContractError("executor fails health check and remains out of scheduling pool")
            previous = self.store.get("nodes", node_id)
            version = (previous or {}).get("resource_version", 0) + 1
            node = {
                "node_id": node_id,
                "resource_version": version,
                "lifecycle": "READY",
                "discovered_at": (previous or {}).get("discovered_at", now()),
                "last_heartbeat_at": now(),
                "heartbeat_timeout_seconds": extra.get("heartbeat_timeout_seconds", 120),
                "labels": labels or {},
                "npus": [{"id": n["id"], "health": n.get("health", "HEALTHY")} for n in npus],
                "health": health,
                "cpu": {"build_slots": max(1, int(build_slots))},
                "storage": extra.get("storage", {}),
                "network": extra.get("network", {}),
                "topology": extra.get("topology", {}),
                "inventory_digest": digest({"npus": npus, "labels": labels or {}}),
                "needs_reconciliation": False,
            }
            self.store.put("nodes", node)
            self.store.event("EXECUTOR_READY", node_id, after="READY",
                             data={"resource_version": version, "card_count": len(npus)})
            return node
        return self._write(request_id, "executor_register", payload, write)

    def heartbeat_executor(self, node_id, health="HEALTHY", npus=None, request_id=None):
        payload = {"node_id": node_id, "health": health, "npus": npus}
        def write():
            node = self.store.require("nodes", node_id)
            node["last_heartbeat_at"] = now()
            node["health"] = health
            if npus is not None:
                node["npus"] = [{"id": n["id"], "health": n.get("health", "HEALTHY")} for n in npus]
            node["resource_version"] += 1
            if health != "HEALTHY":
                node["lifecycle"] = "LOST"
                self._fence_node_leases(node_id)
            self.store.put("nodes", node)
            self.store.event("EXECUTOR_HEARTBEAT", node_id, after=node["lifecycle"])
            return node
        return self._write(request_id, "executor_heartbeat", payload, write)

    def set_node_lifecycle(self, node_id, lifecycle, request_id=None):
        if lifecycle not in {"READY", "CORDONED", "DRAINING", "LOST"}:
            raise ContractError("unsupported lifecycle")
        def write():
            node = self.store.require("nodes", node_id)
            before = node["lifecycle"]
            node["lifecycle"] = lifecycle
            node["resource_version"] += 1
            if lifecycle == "LOST":
                node["health"] = "LOST"
                self._fence_node_leases(node_id)
            if lifecycle == "READY" and before == "LOST":
                node["needs_reconciliation"] = True
                node["lifecycle"] = "DRAINING"
            self.store.put("nodes", node)
            self.store.event("NODE_LIFECYCLE", node_id, before=before, after=node["lifecycle"])
            return node
        return self._write(request_id, "node_lifecycle", {"node_id": node_id, "lifecycle": lifecycle}, write)

    def confirm_node_reconciled(self, node_id, health="HEALTHY", request_id=None):
        def write():
            node = self.store.require("nodes", node_id)
            if health != "HEALTHY":
                raise ContractError("node cannot rejoin before a healthy device check")
            node.update({"lifecycle": "READY", "health": health, "needs_reconciliation": False,
                         "last_heartbeat_at": now()})
            node["resource_version"] += 1
            self.store.put("nodes", node)
            self.store.event("NODE_RECONCILED", node_id, after="READY")
            return node
        return self._write(request_id, "node_reconciled", {"node_id": node_id, "health": health}, write)

    def _fence_node_leases(self, node_id):
        for lease in self.store.all("leases"):
            if lease["status"] == "ACTIVE" and any(r.startswith(node_id + ":") for r in lease["resources"]):
                lease["status"] = "FENCED"
                self.store.put("leases", lease)
                step = self.store.get("steps", lease["step_id"])
                if step and step["status"] == "RUNNING":
                    step["status"] = "RETRYABLE"
                    self.store.put("steps", step)
                self.store.event("LEASE_FENCED", lease["lease_id"], reason="executor lost")

    def cluster_status(self):
        nodes = self.store.all("nodes")
        steps = self.store.all("steps")
        leases = self.store.all("leases")
        healthy_cards = sum(1 for n in nodes if n["lifecycle"] == "READY" and n["health"] == "HEALTHY"
                            for card in n["npus"] if card["health"] == "HEALTHY")
        active_cards = sum(1 for lease in leases if lease["status"] == "ACTIVE"
                           for resource in lease["resources"] if ":npu-" in resource)
        return {
            "resource_version": sum(n["resource_version"] for n in nodes),
            "nodes": nodes,
            "healthy_cards": healthy_cards,
            "active_leased_cards": active_cards,
            "queues": {status: len([s for s in steps if s["status"] == status])
                       for status in ("BLOCKED", "READY", "RUNNING", "RETRYABLE", "SUCCEEDED")},
            "reservations": self.store.all("reservations"),
            "avoidable_idle": self.avoidable_idle(),
        }

    def avoidable_idle(self):
        ready = [s for s in self.store.all("steps")
                 if s["status"] == "READY" and (s["kind"] != "RL_INTEGRATION_VERIFY" or s.get("resource_group_id"))]
        if not ready:
            return False
        for step in ready:
            if self.scheduler.plan(step) is not None:
                return True
        return False

    def register_model_spec(self, spec, request_id=None):
        def write():
            required = ("name", "source_revision", "framework", "mode", "weights_digest", "config_digest")
            model = spec.get("model", {})
            missing = [key for key in required if not model.get(key)]
            if missing:
                raise ContractError("ModelSpec missing fields: %s" % ", ".join(missing))
            document = dict(spec)
            document.setdefault("model_spec_id", new_id("model-spec"))
            document["source_digest"] = digest({"model": model, "workload": spec.get("workload", {})})
            document["source_readonly"] = True
            document["created_at"] = now()
            self.store.put("model_specs", document)
            self.store.event("MODEL_SPEC_REGISTERED", document["model_spec_id"], after="FROZEN")
            return document
        return self._write(request_id, "model_spec_register", spec, write)

    def register_model_artifact(self, model_spec_id, conversion, content_digest,
                                kind="DERIVED", request_id=None):
        payload = {"model_spec_id": model_spec_id, "conversion": conversion,
                   "content_digest": content_digest, "kind": kind}
        def write():
            source = self.store.require("model_specs", model_spec_id)
            artifact = {
                "artifact_id": new_id("artifact"),
                "model_spec_id": model_spec_id,
                "kind": kind,
                "source_digest": source["source_digest"],
                "conversion_digest": digest(conversion),
                "conversion": conversion,
                "content_digest": content_digest,
                "created_at": now(),
            }
            self.store.put("model_artifacts", artifact)
            self.store.event("MODEL_ARTIFACT_REGISTERED", artifact["artifact_id"])
            return artifact
        return self._write(request_id, "model_artifact_register", payload, write)

    def validate_model_artifact(self, artifact_id, model_spec_id=None, conversion=None):
        artifact = self.store.require("model_artifacts", artifact_id)
        source = self.store.require("model_specs", model_spec_id or artifact["model_spec_id"])
        valid = artifact["source_digest"] == source["source_digest"]
        if conversion is not None:
            valid = valid and artifact["conversion_digest"] == digest(conversion)
        return {"valid": valid, "artifact_id": artifact_id, "source_digest": source["source_digest"]}

    def reusable_artifacts(self, model_spec_id, conversion):
        source = self.store.require("model_specs", model_spec_id)
        return [a for a in self.store.all("model_artifacts")
                if a["model_spec_id"] == model_spec_id and a["source_digest"] == source["source_digest"]
                and a["conversion_digest"] == digest(conversion)]

    def register_rl_spec(self, spec, request_id=None):
        def write():
            required = ("weight_exchange_contract", "trajectory_contract", "generation_contract")
            missing = [key for key in required if key not in spec]
            if missing:
                raise ContractError("RLSpec missing contracts: %s" % ", ".join(missing))
            document = dict(spec)
            document.setdefault("rl_spec_id", new_id("rl-spec"))
            document["contract_version"] = "v1"
            document["status"] = "DRAFT"
            document["history"] = []
            document["created_at"] = now()
            self.store.put("rl_specs", document)
            self.store.event("RL_SPEC_REGISTERED", document["rl_spec_id"], after="DRAFT")
            return document
        return self._write(request_id, "rl_spec_register", spec, write)

    def freeze_rl_contract(self, rl_spec_id, request_id=None):
        def write():
            spec = self.store.require("rl_specs", rl_spec_id)
            spec["status"] = "FROZEN"
            spec["frozen_at"] = now()
            self.store.put("rl_specs", spec)
            steps = [s for s in self.store.all("steps")
                     if s["kind"] == "RL_CONTRACT_FREEZE" and
                     self.store.get("tasks", s["task_id"]).get("rl_spec_id") == rl_spec_id]
            for step in steps:
                self.complete_step(step["step_id"], correctness="PASSED", _internal=True)
            self.store.event("RL_CONTRACT_FROZEN", rl_spec_id, after="FROZEN",
                             data={"contract_version": spec["contract_version"]})
            return spec
        return self._write(request_id, "rl_contract_freeze", {"rl_spec_id": rl_spec_id}, write)

    def update_rl_contract(self, rl_spec_id, changes, request_id=None):
        def write():
            spec = self.store.require("rl_specs", rl_spec_id)
            if spec["status"] != "FROZEN":
                raise ConflictError("only a frozen contract can be versioned")
            old_version = int(spec["contract_version"].lstrip("v"))
            previous = {k: v for k, v in spec.items() if k not in {"history", "created_at"}}
            spec["history"].append(previous)
            spec.update(changes)
            spec["contract_version"] = "v%d" % (old_version + 1)
            self.store.put("rl_specs", spec)
            for task in self.store.all("tasks"):
                if task.get("rl_spec_id") != rl_spec_id:
                    continue
                for step in self.store.all("steps"):
                    if step["task_id"] == task["task_id"] and step["kind"] == "RL_INTEGRATION_VERIFY":
                        step["contract_version"] = spec["contract_version"]
                        step["metadata"]["contract_changed"] = True
                        self.store.put("steps", step)
            self.store.event("RL_CONTRACT_UPDATED", rl_spec_id,
                             before=previous["contract_version"], after=spec["contract_version"])
            return spec
        return self._write(request_id, "rl_contract_update", {"rl_spec_id": rl_spec_id, "changes": changes}, write)

    def create_task(self, objective, task_kind=None, scenario=None, priority=50,
                    acceptance=None, budget=None, model_spec_id=None, rl_spec_id=None, request_id=None):
        payload = {"objective": objective, "task_kind": task_kind, "scenario": scenario,
                   "priority": priority, "acceptance": acceptance, "budget": budget,
                   "model_spec_id": model_spec_id, "rl_spec_id": rl_spec_id}
        def write():
            detected_kind, detected_scenario = classify_task(objective, scenario)
            task_kind_value = task_kind or detected_kind
            scenario_value = scenario or detected_scenario
            model_spec = self.store.get("model_specs", model_spec_id) if model_spec_id else None
            rl_spec = self.store.get("rl_specs", rl_spec_id) if rl_spec_id else None
            task = task_document(task_kind_value, objective, scenario_value, priority,
                                 acceptance, budget, model_spec_id, rl_spec_id)
            steps = initial_steps(task, model_spec, rl_spec)
            self.store.put("tasks", task)
            for step in steps:
                self.store.put("steps", step)
            self.store.event("TASK_CREATED", task["task_id"], after=task["current_phase"],
                             data={"task_kind": task_kind_value, "scenario": scenario_value,
                                   "step_ids": [s["step_id"] for s in steps]})
            self.scheduler.refresh()
            return self.get_task(task["task_id"])
        return self._write(request_id, "task_create", payload, write)

    def get_task(self, task_id):
        task = self.store.require("tasks", task_id)
        steps = [s for s in self.store.all("steps") if s["task_id"] == task_id]
        update_phase(task, steps)
        task["updated_at"] = now()
        self.store.put("tasks", task)
        task["steps"] = sorted(steps, key=lambda s: s["created_at"])
        task["evidence"] = [e for e in self.store.all("evidence") if e["task_id"] == task_id]
        return task

    def add_steps(self, task_id, steps, request_id=None):
        payload = {"task_id": task_id, "steps": steps}
        def write():
            task = self.store.require("tasks", task_id)
            existing = {s["step_id"] for s in self.store.all("steps")}
            new_steps = [step_document(task_id, s["kind"], s.get("resource"), s.get("dependencies"),
                                       priority=s.get("priority", task["priority"]), role=s.get("role", "GENERAL"),
                                       contract_version=s.get("contract_version"), metadata=s.get("metadata")) for s in steps]
            all_ids = existing | {s["step_id"] for s in new_steps}
            if any(d not in all_ids for s in new_steps for d in s["dependencies"]):
                raise ContractError("step dependency does not exist")
            for step in new_steps:
                self.store.put("steps", step)
                self.store.event("STEP_ADDED", step["step_id"], after=step["status"])
            self.scheduler.refresh()
            return new_steps
        return self._write(request_id, "task_update_dag", payload, write)

    def report_missing_operators(self, parent_step_id, operators, request_id=None):
        payload = {"parent_step_id": parent_step_id, "operators": operators}
        def write():
            parent = self.store.require("steps", parent_step_id)
            if parent["kind"] != "COVERAGE_VERIFY":
                raise ContractError("missing operator sub-DAG starts at COVERAGE_VERIFY")
            steps, final_id = operator_subdag(parent["task_id"], operators, parent_step_id)
            for step in steps:
                self.store.put("steps", step)
            for other in self.store.all("steps"):
                if parent_step_id in other["dependencies"] and other["step_id"] not in {s["step_id"] for s in steps}:
                    other["dependencies"] = [final_id if d == parent_step_id else d for d in other["dependencies"]]
                    self.store.put("steps", other)
            self.complete_step(parent_step_id, correctness="PASSED", _internal=True)
            self.scheduler.refresh()
            self.store.event("OPERATOR_SUBDAG_CREATED", parent["task_id"],
                             data={"operators": operators, "steps": [s["step_id"] for s in steps]})
            return steps
        return self._write(request_id, "operator_subdag_create", payload, write)

    def pause_task(self, task_id, request_id=None):
        return self._task_state(task_id, "PAUSED", request_id)

    def resume_task(self, task_id, request_id=None):
        return self._task_state(task_id, "ACTIVE", request_id)

    def cancel_task(self, task_id, request_id=None):
        return self._task_state(task_id, "CANCELLED", request_id)

    def _task_state(self, task_id, state, request_id):
        def write():
            task = self.store.require("tasks", task_id)
            task["state"] = state
            self.store.put("tasks", task)
            self.store.event("TASK_STATE", task_id, after=state)
            return task
        return self._write(request_id, "task_state", {"task_id": task_id, "state": state}, write)

    def schedule(self):
        with self.store.transaction():
            granted = self.scheduler.schedule()
        return {"granted_lease_ids": granted, "cluster": self.cluster_status()}

    def request_reservation(self, step_id, starts_at=None, guard_seconds=30, request_id=None):
        payload = {"step_id": step_id, "starts_at": starts_at, "guard_seconds": guard_seconds}
        return self._write(request_id, "reservation_request", payload,
                           lambda: self.scheduler.request_reservation(step_id, starts_at, guard_seconds))

    def reservation_status(self, reservation_id):
        with self.store.transaction():
            return self.scheduler.reservation_status(reservation_id)

    def cancel_reservation(self, reservation_id, request_id=None):
        def write():
            reservation = self.store.require("reservations", reservation_id)
            if reservation["status"] == "FULFILLED":
                raise ConflictError("fulfilled reservation cannot be cancelled")
            reservation["status"] = "CANCELLED"
            self.store.put("reservations", reservation)
            self.store.event("RESERVATION_CANCELLED", reservation_id, after="CANCELLED")
            return reservation
        return self._write(request_id, "reservation_cancel", {"reservation_id": reservation_id}, write)

    def heartbeat_lease(self, lease_id, epoch, progress=None, request_id=None):
        payload = {"lease_id": lease_id, "epoch": epoch, "progress": progress}
        def write():
            lease = self.store.require("leases", lease_id)
            self._validate_lease(lease, epoch)
            lease["last_heartbeat_at"] = now()
            lease["expires_at"] = now() + lease["heartbeat_timeout_seconds"]
            lease["progress"] = progress or {}
            self.store.put("leases", lease)
            self.store.event("LEASE_HEARTBEAT", lease_id, data={"progress": progress or {}})
            return lease
        return self._write(request_id, "lease_heartbeat", payload, write)

    def release_lease(self, lease_id, epoch, request_id=None):
        def write():
            lease = self.store.require("leases", lease_id)
            self._validate_lease(lease, epoch)
            lease["status"] = "RELEASING"
            self.store.put("leases", lease)
            self.store.event("LEASE_RELEASING", lease_id, after="RELEASING")
            return lease
        return self._write(request_id, "lease_release", {"lease_id": lease_id, "epoch": epoch}, write)

    def confirm_cleanup(self, lease_id, epoch=None, request_id=None):
        def write():
            lease = self.store.require("leases", lease_id)
            if epoch is not None and lease["epoch"] != epoch:
                raise FencingError("lease epoch mismatch")
            if lease["status"] not in {"RELEASING", "FENCED", "REVOKED"}:
                raise ConflictError("lease is not awaiting cleanup")
            lease["status"] = "RELEASED" if lease["status"] == "RELEASING" else "CLEANED"
            lease["cleanup_confirmed"] = True
            lease["released_at"] = now()
            self.store.put("leases", lease)
            self.store.event("LEASE_CLEANED", lease_id, after=lease["status"])
            return lease
        return self._write(request_id, "cleanup_confirm", {"lease_id": lease_id, "epoch": epoch}, write)

    def _validate_lease(self, lease, epoch, expected_step=None):
        if lease["epoch"] != epoch:
            raise FencingError("stale lease epoch")
        if lease["status"] != "ACTIVE" or lease["expires_at"] <= now():
            raise FencingError("lease is not active")
        if expected_step and lease["step_id"] != expected_step:
            raise FencingError("lease belongs to another step")
        for resource in lease["resources"]:
            node_id = resource.split(":", 1)[0]
            if node_id == "control":
                continue
            node = self.store.get("nodes", node_id)
            if not node or node["lifecycle"] != "READY" or node["health"] != "HEALTHY":
                raise FencingError("executor is not healthy")

    def run_step(self, step_id, lease_id, epoch, correctness="PASSED", metrics=None,
                 artifact_digests=None, contract_versions=None, owner=None,
                 simulated=True, candidate_id=None, request_id=None):
        payload = {"step_id": step_id, "lease_id": lease_id, "epoch": epoch, "correctness": correctness,
                   "metrics": metrics, "artifact_digests": artifact_digests,
                   "contract_versions": contract_versions, "owner": owner, "simulated": simulated,
                   "candidate_id": candidate_id}
        def write():
            step = self.store.require("steps", step_id)
            task = self.store.require("tasks", step["task_id"])
            lease = self.store.require("leases", lease_id)
            self._validate_lease(lease, epoch, step_id)
            group_lease = (step.get("resource_group_id") and
                           lease.get("resource_group_id") == step.get("resource_group_id"))
            if step["status"] != "RUNNING" or (step["lease_id"] != lease_id and not group_lease):
                raise FencingError("step is not running under this lease")
            if owner and owner != step["owner"]:
                raise FencingError("only the declared Owner can execute this step")
            expected_runner = step["execution"]["runner"]
            scope = step["resource"]["scope"]
            if ((scope == "SINGLE_CARD" and expected_runner != "single-card-test-runner") or
                    (scope == "MULTI_NODE" and expected_runner != "multi-node-test-runner")):
                raise FencingError("runner and resource scope do not match")
            if not simulated:
                raise ContractError("no real Node Executor is configured; use simulated prototype Runner")
            if scope == "MULTI_NODE":
                digests = artifact_digests or step["metadata"].get("artifact_digests")
                if not digests or len(set(digests)) != 1:
                    raise FencingError("multi-node artifact digests must be present and identical")
            if step["kind"] == "RL_INTEGRATION_VERIFY":
                group = self.store.get("resource_groups", step.get("resource_group_id") or "")
                if not group or group["status"] != "ACTIVE" or len(group.get("lease_ids", [])) != len(group["roles"]):
                    raise FencingError("RL integration requires a fully active atomic resource group")
                if contract_versions and len(set(contract_versions)) != 1:
                    raise FencingError("RL contract versions are incompatible")
                if contract_versions and contract_versions[0] != step["contract_version"]:
                    raise FencingError("RL execution uses a stale contract")
                required_exchanges = ("weight_export", "weight_import", "trajectory_generation",
                                      "trajectory_consumption", "weight_resync")
                if (not metrics or not all(metrics.get(name) for name in required_exchanges)
                        or metrics.get("rollout_train_rounds", 0) < 2):
                    raise ContractError("RL closed loop requires both exchanges and at least two rollout-train rounds")
            run_dir = os.path.join(self.workspace_root, task["task_id"], step_id, "run")
            os.makedirs(run_dir, exist_ok=True)
            execution_id = new_id("execution")
            log_path = os.path.join(run_dir, execution_id + ".log")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("simulated=%s\nrunner=%s\ncorrectness=%s\n" % (simulated, expected_runner, correctness))
            environment = {
                "resource_version": self.cluster_status()["resource_version"],
                "resources": lease["resources"],
                "runner": expected_runner,
                "simulated": True,
            }
            execution = {
                "execution_id": execution_id, "task_id": task["task_id"], "step_id": step_id,
                "lease_id": lease_id, "epoch": epoch, "status": "SUCCEEDED" if correctness == "PASSED" else "FAILED",
                "runner": expected_runner, "log_uri": log_path, "environment_fingerprint": environment,
                "started_at": now(), "finished_at": now(),
            }
            self.store.put("executions", execution)
            evidence = evidence_document(task["task_id"], step_id,
                                         {"kind": "CONTROLLED_RUNNER", "name": expected_runner, "version": "0.1.0"},
                                         correctness, candidate_id=candidate_id, role=step["role"],
                                         contract_version=step["contract_version"],
                                         performance=metrics, environment=environment, logs_uri=log_path)
            self.store.put("evidence", evidence)
            step["outputs"].append(evidence["evidence_id"])
            step["status"] = "SUCCEEDED" if correctness == "PASSED" else "FAILED"
            step["finished_at"] = now()
            self.store.put("steps", step)
            lease["status"] = "RELEASED"
            lease["cleanup_confirmed"] = True
            lease["released_at"] = now()
            self.store.put("leases", lease)
            if group_lease:
                group = self.store.require("resource_groups", step["resource_group_id"])
                for other_id in group["lease_ids"]:
                    other = self.store.require("leases", other_id)
                    other["status"] = "RELEASED"
                    other["cleanup_confirmed"] = True
                    other["released_at"] = now()
                    self.store.put("leases", other)
                group["status"] = "RELEASED"
                group["finished_at"] = now()
                self.store.put("resource_groups", group)
            self.store.event("EXECUTION_FINISHED", execution_id, after=execution["status"],
                             data={"evidence_id": evidence["evidence_id"], "lease_id": lease_id})
            self.scheduler.refresh()
            return {"execution": execution, "evidence": evidence}
        return self._write(request_id, "execution_run", payload, write)

    def complete_step(self, step_id, correctness="PASSED", source=None, _internal=False, request_id=None):
        payload = {"step_id": step_id, "correctness": correctness, "source": source}
        def write():
            step = self.store.require("steps", step_id)
            if step["status"] == "SUCCEEDED":
                return {"step": step, "evidence": None}
            if step["status"] == "RUNNING":
                raise ConflictError("running NPU step must complete through run_step")
            if step["status"] not in {"READY", "BLOCKED", "DRAFT"} and not _internal:
                raise ConflictError("step cannot complete from %s" % step["status"])
            evidence = evidence_document(step["task_id"], step_id,
                                         source or {"kind": "WORKER", "name": "prototype-worker", "version": "0.1.0"},
                                         correctness, role=step["role"], contract_version=step["contract_version"])
            self.store.put("evidence", evidence)
            step["status"] = "SUCCEEDED" if correctness == "PASSED" else "FAILED"
            step["finished_at"] = now()
            step["outputs"].append(evidence["evidence_id"])
            self.store.put("steps", step)
            self.store.event("STEP_COMPLETED", step_id, after=step["status"])
            self.scheduler.refresh()
            return {"step": step, "evidence": evidence}
        if _internal:
            return write()
        return self._write(request_id, "step_complete", payload, write)

    def create_resource_group(self, step_id, roles, owner="rl-integration-owner", request_id=None):
        payload = {"step_id": step_id, "roles": roles, "owner": owner}
        def write():
            step = self.store.require("steps", step_id)
            if step["kind"] != "RL_INTEGRATION_VERIFY" or step["status"] != "READY":
                raise ConflictError("RL step is not ready for integration")
            if owner != step["owner"]:
                raise FencingError("RL integration has a different single Owner")
            normalized = {name: resource_spec(**spec) for name, spec in roles.items()}
            if len(normalized) < 2:
                raise ContractError("joint resource group needs training and inference roles")
            group = {
                "resource_group_id": new_id("rg"), "task_id": step["task_id"], "step_id": step_id,
                "purpose": "RL_ROLLOUT_TRAIN", "allocation_mode": "ATOMIC", "roles": normalized,
                "owner": owner, "status": "PENDING", "lease_ids": [], "created_at": now(),
            }
            self.store.put("resource_groups", group)
            step["resource_group_id"] = group["resource_group_id"]
            self.store.put("steps", step)
            self.store.event("RESOURCE_GROUP_PENDING", group["resource_group_id"], after="PENDING")
            return group
        return self._write(request_id, "resource_group_create", payload, write)

    def release_resource_group(self, group_id, request_id=None):
        def write():
            group = self.store.require("resource_groups", group_id)
            for lease_id in group["lease_ids"]:
                lease = self.store.require("leases", lease_id)
                lease["status"] = "RELEASED"
                lease["cleanup_confirmed"] = True
                lease["released_at"] = now()
                self.store.put("leases", lease)
            group["status"] = "RELEASED"
            group["finished_at"] = now()
            self.store.put("resource_groups", group)
            self.store.event("RESOURCE_GROUP_RELEASED", group_id, after="RELEASED")
            return group
        return self._write(request_id, "resource_group_release", {"resource_group_id": group_id}, write)

    def validate_rl_integration(self, step_id):
        step = self.store.require("steps", step_id)
        task = self.store.require("tasks", step["task_id"])
        spec = self.store.require("rl_specs", task["rl_spec_id"])
        if step["contract_version"] != spec["contract_version"]:
            return {"valid": False, "reason": "stale contract version"}
        dependencies = [self.store.require("steps", d) for d in step["dependencies"]]
        roles = {d["role"] for d in dependencies if d["status"] == "SUCCEEDED"}
        valid = {"TRAINING", "INFERENCE"}.issubset(roles)
        return {"valid": valid, "contract_version": spec["contract_version"],
                "reason": None if valid else "both branch functional evidence packs are required"}

    def campaign_create(self, *args, **kwargs):
        return self.rsi.create(*args, **kwargs)

    def campaign_get(self, campaign_id):
        return self.rsi.get(campaign_id)

    def campaign_append_candidate(self, *args, **kwargs):
        return self.rsi.append_candidate(*args, **kwargs)

    def campaign_record_evaluation(self, *args, **kwargs):
        return self.rsi.record_evaluation(*args, **kwargs)

    def campaign_assert_baseline(self, campaign_id, evidence_id):
        return self.rsi.assert_baseline(campaign_id, evidence_id)

    def campaign_checkpoint(self, *args, **kwargs):
        return self.rsi.checkpoint(*args, **kwargs)

    def campaign_stop(self, *args, **kwargs):
        return self.rsi.stop(*args, **kwargs)

    def campaign_joint_batch(self, *args, **kwargs):
        return self.rsi.joint_batch(*args, **kwargs)

    def measurement_window_create(self, *args, **kwargs):
        return self.windows.create(*args, **kwargs)

    def measurement_window_close(self, *args, **kwargs):
        return self.windows.close(*args, **kwargs)

    def interference_profile_record(self, *args, **kwargs):
        return self.observability.record_interference_profile(*args, **kwargs)

    def scheduler_tuning(self):
        return self.observability.tuning_recommendations()

    def scheduler_tuning_apply(self, *args, **kwargs):
        return self.observability.apply_tuning(*args, **kwargs)

    def metrics_snapshot(self):
        return self.observability.metrics()

    def dashboard_generate(self, path):
        return self.observability.dashboard(path)

    def list_capabilities(self):
        return self.adapters.list()

    def invoke_capability(self, capability_id, inputs, request_id=None):
        return self.adapters.invoke(capability_id, inputs, request_id)

    def evidence(self, evidence_id):
        return self.store.require("evidence", evidence_id)

    def events(self, limit=200):
        return self.store.events(limit)

    def recover(self):
        with self.store.transaction():
            for lease in self.store.all("leases"):
                if lease["status"] == "ACTIVE":
                    lease["status"] = "FENCED"
                    self.store.put("leases", lease)
            for step in self.store.all("steps"):
                if step["status"] == "RUNNING":
                    step["status"] = "RETRYABLE"
                    step["rejection_reason"] = "controller recovery requires executor reconciliation"
                    self.store.put("steps", step)
            for node in self.store.all("nodes"):
                node["lifecycle"] = "DRAINING"
                node["needs_reconciliation"] = True
                node["resource_version"] += 1
                self.store.put("nodes", node)
            self.store.event("CONTROLLER_RECOVERED", "controller", after="AWAITING_RECONCILIATION")
        return self.cluster_status()
