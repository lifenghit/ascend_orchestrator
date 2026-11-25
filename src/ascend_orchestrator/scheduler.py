"""Deterministic work-conserving scheduler and fencing-aware lease allocator."""

from .models import ConflictError, lease_document, new_id, now


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
LIVE_LEASES = {"ACTIVE", "RELEASING", "EXPIRED", "FENCED", "REVOKED"}


class Scheduler:
    def __init__(self, store, max_agent_workers=8, default_lease_ttl=60):
        self.store = store
        self.max_agent_workers = max_agent_workers
        self.default_lease_ttl = default_lease_ttl

    def refresh(self):
        changed = []
        steps = {s["step_id"]: s for s in self.store.all("steps")}
        for step in steps.values():
            if step["status"] not in {"DRAFT", "BLOCKED", "RETRYABLE"}:
                continue
            task = self.store.get("tasks", step["task_id"])
            if not task or task["state"] != "ACTIVE":
                continue
            unresolved = [d for d in step["dependencies"] if steps.get(d, {}).get("status") != "SUCCEEDED"]
            new_status = "BLOCKED" if unresolved else "READY"
            if step["status"] != new_status:
                before = step["status"]
                step["status"] = new_status
                step["ready_at"] = now() if new_status == "READY" else None
                step["rejection_reason"] = None
                self.store.put("steps", step)
                self.store.event("STEP_STATUS", step["step_id"], before=before, after=new_status,
                                 data={"unresolved_dependencies": unresolved})
                changed.append(step["step_id"])
        return changed

    def reconcile_expired(self):
        expired = []
        for lease in self.store.all("leases"):
            if lease["status"] == "ACTIVE" and lease["expires_at"] <= now():
                lease["status"] = "FENCED"
                self.store.put("leases", lease)
                step = self.store.get("steps", lease["step_id"])
                if step and step["status"] == "RUNNING":
                    step["status"] = "RETRYABLE"
                    step["rejection_reason"] = "lease fenced; cleanup required before retry"
                    self.store.put("steps", step)
                self.store.event("LEASE_FENCED", lease["lease_id"], reason="heartbeat timeout")
                expired.append(lease["lease_id"])
        return expired

    def schedule(self):
        self.refresh()
        self.reconcile_expired()
        granted = []
        for group in sorted(self.store.all("resource_groups"), key=lambda g: g["created_at"]):
            if group["status"] == "PENDING":
                result = self.acquire_resource_group(group["resource_group_id"], _internal=True)
                if result["status"] == "ACTIVE":
                    granted.extend(result["lease_ids"])
        while len([s for s in self.store.all("steps") if s["status"] == "RUNNING"]) < self.max_agent_workers:
            candidates = [s for s in self.store.all("steps")
                          if s["status"] == "READY" and not s.get("resource_group_id")
                          and s["kind"] != "RL_INTEGRATION_VERIFY"]
            allocated = False
            for step in sorted(candidates, key=self._score, reverse=True):
                plan = self.plan(step)
                if plan is not None:
                    lease = self._create_lease(step, plan)
                    granted.append(lease["lease_id"])
                    allocated = True
                    break
                if candidates:
                    reason = self.rejection_reason(step)
                    if step.get("rejection_reason") != reason:
                        step["rejection_reason"] = reason
                        self.store.put("steps", step)
            if not allocated:
                break
        return granted

    def _score(self, step):
        age = max(0, now() - (step.get("ready_at") or step["created_at"]))
        dependents = sum(1 for other in self.store.all("steps") if step["step_id"] in other["dependencies"])
        task = self.store.get("tasks", step["task_id"]) or {"priority": 50}
        return (step["scheduling"]["priority"] * 100 + task["priority"] * 10 +
                dependents * 50 + min(age, 3600) - step.get("attempt", 0) * 5)

    def eligible_nodes(self, resource, include_busy=False):
        nodes = []
        for node in self.store.all("nodes"):
            if node["lifecycle"] != "READY":
                continue
            if now() - node["last_heartbeat_at"] > node.get("heartbeat_timeout_seconds", 120):
                continue
            if include_busy is False and node.get("health") != "HEALTHY":
                continue
            if any(node["labels"].get(k) != v for k, v in resource.get("node_selector", {}).items()):
                continue
            nodes.append(node)
        return sorted(nodes, key=lambda n: n["node_id"])

    def _occupied(self):
        occupied = {}
        quiet_nodes = set()
        global_quiet = False
        exclusive_nodes = set()
        for lease in self.store.all("leases"):
            if lease["status"] not in LIVE_LEASES:
                continue
            for resource in lease["resources"]:
                occupied[resource] = lease["lease_id"]
            step = self.store.get("steps", lease["step_id"])
            group_role = lease.get("role")
            if group_role and not step:
                continue
            spec = step["resource"] if step else {"performance_isolation": "NONE", "isolation": "DEVICE_EXCLUSIVE"}
            nodes = {r.split(":", 1)[0] for r in lease["resources"]}
            if spec.get("performance_isolation") == "GLOBAL_QUIET":
                global_quiet = True
            elif spec.get("performance_isolation") == "NODE_SET_QUIET":
                quiet_nodes.update(nodes)
            if spec.get("isolation") == "NODE_SET_EXCLUSIVE":
                exclusive_nodes.update(nodes)
        return occupied, quiet_nodes, global_quiet, exclusive_nodes

    def plan(self, step, additional_occupied=None):
        resource = step["resource"]
        now_value = now()
        reservation = self._step_reservation(step["step_id"])
        if reservation and reservation["starts_at"] > now_value:
            return None
        occupied, quiet_nodes, global_quiet, exclusive_nodes = self._occupied()
        if additional_occupied:
            occupied.update({resource: "pending-resource-group" for resource in additional_occupied})
        scope = resource["scope"]
        if scope == "NONE":
            return []
        nodes = self.eligible_nodes(resource)
        if not nodes:
            return None
        if global_quiet:
            return None
        if scope == "CPU_ONLY":
            for node in nodes:
                if node["node_id"] in quiet_nodes or node["node_id"] in exclusive_nodes:
                    continue
                if self._reservation_blocks(step, [node["node_id"]], now_value):
                    continue
                used = sum(1 for r in occupied if r.startswith(node["node_id"] + ":build-"))
                if used < node["cpu"]["build_slots"]:
                    return ["%s:build-%d" % (node["node_id"], used)]
            return None
        chosen = self._choose_nodes(nodes, resource, occupied, quiet_nodes, exclusive_nodes)
        if not chosen:
            return None
        selected_nodes = [n["node_id"] for n, _ in chosen]
        if any(n in quiet_nodes or n in exclusive_nodes for n in selected_nodes):
            return None
        if self._reservation_blocks(step, selected_nodes, now_value):
            return None
        resources = []
        for node, cards in chosen:
            resources.extend("%s:npu-%s" % (node["node_id"], card) for card in cards)
        return resources

    def _choose_nodes(self, nodes, resource, occupied, quiet_nodes, exclusive_nodes):
        available = []
        for node in nodes:
            if node["node_id"] in quiet_nodes or node["node_id"] in exclusive_nodes:
                continue
            cards = [str(npu["id"]) for npu in node["npus"]
                     if npu["health"] == "HEALTHY" and "%s:npu-%s" % (node["node_id"], npu["id"]) not in occupied]
            available.append((node, cards))
        scope = resource["scope"]
        if scope == "SINGLE_CARD":
            for node, cards in available:
                if cards:
                    return [(node, cards[:1])]
            return None
        fitting = [(n, c[:resource["cards_per_node"]]) for n, c in available
                   if len(c) >= resource["cards_per_node"]]
        if scope == "SINGLE_NODE":
            return fitting[:1] or None
        if len(fitting) < resource["nodes"]:
            return None
        return fitting[:resource["nodes"]]

    def rejection_reason(self, step):
        nodes = self.eligible_nodes(step["resource"])
        if not nodes:
            return "no READY healthy nodes match node_selector"
        if self._step_reservation(step["step_id"]):
            return "waiting for reservation start or target resources"
        if self._occupied()[2]:
            return "global quiet window active"
        return "compatible resources busy, draining, fenced, quiet, or reserved"

    def _step_reservation(self, step_id):
        for reservation in self.store.all("reservations"):
            if reservation["step_id"] == step_id and reservation["status"] in {"PENDING", "ACTIVE"}:
                return reservation
        return None

    def _reservation_blocks(self, step, node_ids, at):
        estimate = step["scheduling"]["estimated_duration_seconds"]
        for reservation in self.store.all("reservations"):
            if reservation["step_id"] == step["step_id"] or reservation["status"] not in {"PENDING", "ACTIVE"}:
                continue
            if not set(node_ids).intersection(reservation["target_nodes"]):
                continue
            if at + estimate + reservation["guard_seconds"] > reservation["starts_at"]:
                return True
        return False

    def _create_lease(self, step, resources):
        task = self.store.require("tasks", step["task_id"])
        ttl = min(step["execution"]["timeout_seconds"], self.default_lease_ttl)
        if step["resource"].get("lease_class") == "SERVICE":
            ttl = min(step["resource"].get("max_duration_seconds", 7200), self.default_lease_ttl)
        lease = lease_document(task["task_id"], step["step_id"], step["owner"], resources,
                               ttl, self.store.next_epoch(), step["resource"].get("lease_class", "STANDARD"))
        self.store.put("leases", lease)
        step["status"] = "RUNNING"
        step["lease_id"] = lease["lease_id"]
        step["started_at"] = now()
        step["attempt"] += 1
        step["rejection_reason"] = None
        self.store.put("steps", step)
        reservation = self._step_reservation(step["step_id"])
        if reservation:
            reservation["status"] = "FULFILLED"
            self.store.put("reservations", reservation)
        self.store.event("LEASE_GRANTED", lease["lease_id"], reason="work-conserving deterministic allocation",
                         after=lease["status"], data={"resources": resources, "step_id": step["step_id"]})
        return lease

    def request_reservation(self, step_id, starts_at=None, guard_seconds=30):
        step = self.store.require("steps", step_id)
        if step["status"] != "READY":
            raise ConflictError("only a READY step can reserve resources")
        if step["resource"]["scope"] not in {"SINGLE_NODE", "MULTI_NODE"}:
            raise ConflictError("reservations are for node-set steps")
        nodes = self.eligible_nodes(step["resource"])
        if len(nodes) < step["resource"]["nodes"]:
            raise ConflictError("not enough compatible nodes for reservation")
        selected = nodes[:step["resource"]["nodes"]]
        reservation = {
            "reservation_id": new_id("reservation"),
            "task_id": step["task_id"],
            "step_id": step_id,
            "target_nodes": [n["node_id"] for n in selected],
            "starts_at": starts_at or now(),
            "guard_seconds": guard_seconds,
            "status": "PENDING",
            "created_at": now(),
            "blocking_leases": [],
        }
        self.store.put("reservations", reservation)
        self.store.event("RESERVATION_CREATED", reservation["reservation_id"], after="PENDING")
        return reservation

    def reservation_status(self, reservation_id):
        reservation = self.store.require("reservations", reservation_id)
        occupied, _, _, _ = self._occupied()
        blockers = sorted({lease_id for resource, lease_id in occupied.items()
                           if resource.split(":", 1)[0] in reservation["target_nodes"]})
        reservation["blocking_leases"] = blockers
        if not blockers and reservation["status"] == "PENDING" and reservation["starts_at"] <= now():
            reservation["status"] = "ACTIVE"
        self.store.put("reservations", reservation)
        return reservation

    def acquire_resource_group(self, group_id, _internal=False):
        group = self.store.require("resource_groups", group_id)
        if group["status"] == "ACTIVE":
            return group
        if group["status"] != "PENDING":
            raise ConflictError("resource group is not pending")
        all_plans = {}
        used_resources = set()
        for role, spec in sorted(group["roles"].items()):
            fake_step = {
                "step_id": group["step_id"], "task_id": group["task_id"],
                "resource": spec, "scheduling": {"estimated_duration_seconds": spec["duration_seconds"]},
            }
            plan = self.plan(fake_step, used_resources)
            if not plan or used_resources.intersection(plan):
                group["status"] = "PENDING"
                group["rejection_reason"] = "atomic allocation unavailable for role %s" % role
                self.store.put("resource_groups", group)
                return group
            all_plans[role] = plan
            used_resources.update(plan)
        lease_ids = []
        for role, resources in all_plans.items():
            spec = group["roles"][role]
            ttl = min(spec.get("max_duration_seconds", 7200), self.default_lease_ttl)
            lease = lease_document(group["task_id"], group["step_id"], group["owner"], resources,
                                   ttl, self.store.next_epoch(), "SERVICE", group_id, role)
            self.store.put("leases", lease)
            lease_ids.append(lease["lease_id"])
        group["status"] = "ACTIVE"
        group["lease_ids"] = lease_ids
        group["started_at"] = now()
        group["rejection_reason"] = None
        self.store.put("resource_groups", group)
        step = self.store.require("steps", group["step_id"])
        step["status"] = "RUNNING"
        step["started_at"] = now()
        step["attempt"] += 1
        self.store.put("steps", step)
        self.store.event("RESOURCE_GROUP_ACTIVE", group_id, after="ACTIVE", data={"lease_ids": lease_ids})
        return group
