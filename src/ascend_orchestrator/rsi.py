"""Budgeted RSI campaigns shared by operator, model and end-to-end optimization."""

from .models import ConflictError, ContractError, FencingError, digest, new_id, now


ACTIONS = {
    "OPERATOR": {"KERNEL_ONLY"},
    "MODEL": {"OPERATOR_SET", "GRAPH_ONLY", "MEMORY_ONLY", "PARALLEL_ONLY", "RUNTIME_ONLY", "HYBRID"},
    "E2E": {"KERNEL_ONLY", "MODEL_ONLY", "RUNTIME_ONLY", "HYBRID"},
}
ACCEPTED_LABELS = {
    "OPERATOR": "KERNEL_CANDIDATE_ACCEPTED",
    "MODEL": "MODEL_CANDIDATE_ACCEPTED",
    "E2E": "E2E_CANDIDATE_ACCEPTED",
}


def environment_signature(evidence):
    environment = evidence.get("environment_fingerprint", {})
    return digest({key: environment.get(key) for key in
                   ("soc", "driver", "firmware", "cann", "framework", "runtime_profile")})


def comparison_signature(evidence):
    performance = evidence.get("performance") or {}
    return digest({
        "base_commit": evidence.get("code", {}).get("base_commit"),
        "protocol_digest": performance.get("protocol_digest"),
        "model_signature": performance.get("model_signature"),
        "workload_digest": performance.get("workload_digest"),
        "deployment_signature": performance.get("deployment_signature"),
        "contract_version": evidence.get("contract_version"),
        "environment": environment_signature(evidence),
    })


class RSIManager:
    def __init__(self, control):
        self.control = control
        self.store = control.store

    def create(self, task_id, level, baseline_evidence_id, comparison_contract, budget,
               role="GENERAL", parent_campaign_id=None, parent_candidate_id=None, leader="rsi-leader-01",
               request_id=None):
        payload = {"task_id": task_id, "level": level, "baseline_evidence_id": baseline_evidence_id,
                   "comparison_contract": comparison_contract, "budget": budget, "role": role,
                   "parent_campaign_id": parent_campaign_id, "parent_candidate_id": parent_candidate_id,
                   "leader": leader}
        def write():
            if level not in ACTIONS:
                raise ContractError("campaign level must be OPERATOR, MODEL or E2E")
            if role not in {"GENERAL", "TRAINING", "INFERENCE", "RL_INTEGRATION"}:
                raise ContractError("invalid campaign role")
            task = self.store.require("tasks", task_id)
            baseline = self.store.require("evidence", baseline_evidence_id)
            if baseline["task_id"] != task_id or baseline["correctness"]["status"] != "PASSED":
                raise ConflictError("campaign baseline must be passing evidence from the same task")
            performance = baseline.get("performance") or {}
            required = {"primary_metric", "direction", "correctness_gate"}
            if not required.issubset(comparison_contract):
                raise ContractError("comparison contract requires primary_metric, direction and correctness_gate")
            if comparison_contract["direction"] not in {"MAXIMIZE", "MINIMIZE"}:
                raise ContractError("metric direction must be MAXIMIZE or MINIMIZE")
            if comparison_contract["correctness_gate"] != "REQUIRED":
                raise ContractError("correctness gate cannot be disabled")
            metric = comparison_contract["primary_metric"]
            if not isinstance(performance.get(metric), (int, float)):
                raise ContractError("baseline evidence lacks numeric primary metric")
            if not performance.get("protocol_digest"):
                raise ContractError("baseline evidence lacks protocol_digest")
            if level == "MODEL" and not performance.get("model_signature"):
                raise ContractError("model campaign baseline must bind model_signature")
            if level == "E2E" and not (performance.get("workload_digest") and performance.get("deployment_signature")):
                raise ContractError("E2E baseline must bind workload and deployment signatures")
            if role != "GENERAL" and not baseline.get("contract_version"):
                raise ContractError("RL branch baseline must bind contract_version")
            for key in ("max_rounds", "max_no_improvement_rounds", "max_wall_time_minutes"):
                if int(budget.get(key, 0)) < 1:
                    raise ContractError("budget requires positive %s" % key)
            depth = 0
            if parent_campaign_id:
                parent = self.store.require("campaigns", parent_campaign_id)
                if parent["task_id"] != task_id or parent["status"] not in {"ACTIVE", "STOPPED"}:
                    raise ConflictError("parent campaign is not available")
                if not parent_candidate_id:
                    raise ContractError("subcampaign requires a bottleneck parent candidate")
                parent_candidate = self.store.require("candidates", parent_candidate_id)
                if parent_candidate["campaign_id"] != parent_campaign_id or not parent_candidate["payload"].get("bottleneck_confirmed"):
                    raise ConflictError("subcampaign requires profiling-confirmed bottleneck evidence")
                depth = parent.get("depth", 0) + 1
                if depth > 1:
                    raise ConflictError("nested campaigns are limited to one budgeted level")
            campaign = {
                "campaign_id": new_id("camp"), "task_id": task_id, "level": level, "role": role,
                "leader": leader, "baseline": {"evidence_id": baseline_evidence_id},
                "baseline_signature": comparison_signature(baseline),
                "baseline_metric": float(performance[metric]),
                "best_metric": float(performance[metric]),
                "accepted_candidate_id": None,
                "comparison_contract": dict(comparison_contract),
                "budget": dict(budget), "status": "ACTIVE", "stop_reason": None,
                "parent_campaign_id": parent_campaign_id, "parent_candidate_id": parent_candidate_id,
                "depth": depth, "rounds": 0, "no_improvement_rounds": 0,
                "created_at": now(), "updated_at": now(), "checkpoint": None,
                "window_ids": [],
            }
            self.store.put("campaigns", campaign)
            self.store.event("CAMPAIGN_CREATED", campaign["campaign_id"], after="ACTIVE",
                             data={"level": level, "role": role, "baseline_signature": campaign["baseline_signature"]})
            return campaign
        return self.control._write(request_id, "campaign_create", payload, write)

    def get(self, campaign_id):
        campaign = self.store.require("campaigns", campaign_id)
        campaign["candidates"] = [c for c in self.store.all("candidates") if c["campaign_id"] == campaign_id]
        campaign["evaluations"] = [e for e in self.store.all("evaluations") if e["campaign_id"] == campaign_id]
        return campaign

    def assert_baseline(self, campaign_id, evidence_id):
        campaign = self.store.require("campaigns", campaign_id)
        evidence = self.store.require("evidence", evidence_id)
        if comparison_signature(evidence) != campaign["baseline_signature"]:
            raise ConflictError("baseline drift detected; create a new campaign baseline")
        return True

    def append_candidate(self, campaign_id, hypothesis, action_type, payload=None,
                         parent_candidate_id=None, leader="rsi-leader-01", request_id=None):
        data = {"campaign_id": campaign_id, "hypothesis": hypothesis, "action_type": action_type,
                "payload": payload, "parent_candidate_id": parent_candidate_id, "leader": leader}
        def write():
            campaign = self.store.require("campaigns", campaign_id)
            self._leader_and_active(campaign, leader)
            if action_type not in ACTIONS[campaign["level"]]:
                raise ContractError("action_type is not allowed at this campaign level")
            if parent_candidate_id:
                parent = self.store.require("candidates", parent_candidate_id)
                if parent["campaign_id"] != campaign_id:
                    raise ConflictError("candidate parent belongs to another campaign")
            candidate_count = len([c for c in self.store.all("candidates") if c["campaign_id"] == campaign_id])
            if candidate_count >= int(campaign["budget"].get("max_candidates", campaign["budget"]["max_rounds"])):
                raise ConflictError("candidate budget exhausted")
            candidate = {
                "candidate_id": new_id("candidate"), "campaign_id": campaign_id,
                "task_id": campaign["task_id"], "parent_candidate_id": parent_candidate_id,
                "hypothesis": hypothesis, "action_type": action_type, "payload": payload or {},
                "payload_digest": digest(payload or {}), "status": "PROPOSED",
                "evaluation_ids": [], "created_at": now(),
            }
            self.store.put("candidates", candidate)
            self.store.event("CANDIDATE_PROPOSED", candidate["candidate_id"], after="PROPOSED",
                             data={"action_type": action_type, "hypothesis": hypothesis})
            return candidate
        return self.control._write(request_id, "campaign_append_candidate", data, write)

    def record_evaluation(self, candidate_id, evidence_id, measurement_valid=True,
                          leader="rsi-leader-01", request_id=None):
        payload = {"candidate_id": candidate_id, "evidence_id": evidence_id,
                   "measurement_valid": measurement_valid, "leader": leader}
        def write():
            candidate = self.store.require("candidates", candidate_id)
            campaign = self.store.require("campaigns", candidate["campaign_id"])
            self._leader_and_active(campaign, leader)
            evidence = self.store.require("evidence", evidence_id)
            if evidence.get("candidate_id") != candidate_id or evidence["task_id"] != campaign["task_id"]:
                raise ConflictError("evaluation evidence is not bound to this candidate and task")
            if comparison_signature(evidence) != campaign["baseline_signature"]:
                raise ConflictError("evaluation protocol, model, workload or environment differs from frozen baseline")
            performance = evidence.get("performance") or {}
            metric = campaign["comparison_contract"]["primary_metric"]
            value = performance.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ContractError("candidate evidence lacks numeric primary metric")
            correctness = evidence["correctness"]["status"] == "PASSED"
            if not measurement_valid:
                decision = "MEASUREMENT_INVALID"
            elif not correctness:
                decision = "REJECTED_CORRECTNESS"
            elif campaign["level"] == "MODEL" and not performance.get("integration_verified"):
                decision = "REJECTED_INTEGRATION"
            elif campaign["level"] == "E2E" and not performance.get("e2e_validated"):
                decision = "REJECTED_E2E_VALIDATION"
            elif campaign["role"] == "RL_INTEGRATION" and not performance.get("closed_loop_valid"):
                decision = "REJECTED_CLOSED_LOOP"
            else:
                threshold = max(float(campaign["comparison_contract"].get("min_improvement", 0)),
                                float(campaign["comparison_contract"].get("measurement_noise", 0)))
                delta = value - campaign["best_metric"]
                improvement = delta if campaign["comparison_contract"]["direction"] == "MAXIMIZE" else -delta
                if improvement > threshold:
                    decision = ACCEPTED_LABELS[campaign["level"]]
                elif abs(improvement) <= threshold and campaign["comparison_contract"].get("retest_on_noise", True):
                    decision = "RETEST_REQUIRED"
                else:
                    decision = "REJECTED_NO_IMPROVEMENT"
            evaluation = {
                "evaluation_id": new_id("evaluation"), "campaign_id": campaign["campaign_id"],
                "candidate_id": candidate_id, "evidence_id": evidence_id,
                "metric": metric, "value": float(value), "previous_best": campaign["best_metric"],
                "decision": decision, "created_at": now(),
            }
            self.store.put("evaluations", evaluation)
            candidate["evaluation_ids"].append(evaluation["evaluation_id"])
            if decision.startswith("RETEST") or decision == "MEASUREMENT_INVALID":
                candidate["status"] = decision
            elif decision == ACCEPTED_LABELS[campaign["level"]]:
                previous_id = campaign.get("accepted_candidate_id")
                if previous_id:
                    previous = self.store.require("candidates", previous_id)
                    previous["status"] = "SUPERSEDED"
                    self.store.put("candidates", previous)
                candidate["status"] = "ACCEPTED"
                campaign["accepted_candidate_id"] = candidate_id
                campaign["best_metric"] = float(value)
                campaign["no_improvement_rounds"] = 0
            else:
                candidate["status"] = "REJECTED"
                campaign["no_improvement_rounds"] += 1
            if not decision.startswith("RETEST") and decision != "MEASUREMENT_INVALID":
                campaign["rounds"] += 1
            campaign["updated_at"] = now()
            self.store.put("candidates", candidate)
            self._apply_stop_conditions(campaign)
            self.store.put("campaigns", campaign)
            self.store.event("CANDIDATE_EVALUATED", candidate_id, after=decision,
                             data={"metric": metric, "value": value, "round": campaign["rounds"]})
            return {"evaluation": evaluation, "campaign": campaign, "candidate": candidate}
        return self.control._write(request_id, "campaign_record_evaluation", payload, write)

    def _apply_stop_conditions(self, campaign):
        if campaign["rounds"] >= int(campaign["budget"]["max_rounds"]):
            campaign["status"], campaign["stop_reason"] = "STOPPED", "MAX_ROUNDS"
        elif campaign["no_improvement_rounds"] >= int(campaign["budget"]["max_no_improvement_rounds"]):
            campaign["status"], campaign["stop_reason"] = "STOPPED", "NO_IMPROVEMENT"
        elif now() - campaign["created_at"] >= int(campaign["budget"]["max_wall_time_minutes"]) * 60:
            campaign["status"], campaign["stop_reason"] = "STOPPED", "WALL_TIME"

    @staticmethod
    def _leader_and_active(campaign, leader):
        if campaign["leader"] != leader:
            raise FencingError("campaign has a single RSI Leader")
        if campaign["status"] != "ACTIVE":
            raise ConflictError("campaign is stopped: %s" % campaign.get("stop_reason"))

    def checkpoint(self, campaign_id, leader="rsi-leader-01", request_id=None):
        def write():
            campaign = self.store.require("campaigns", campaign_id)
            if campaign["leader"] != leader:
                raise FencingError("campaign has a single RSI Leader")
            campaign["checkpoint"] = {
                "rounds": campaign["rounds"], "best_metric": campaign["best_metric"],
                "accepted_candidate_id": campaign["accepted_candidate_id"],
                "no_improvement_rounds": campaign["no_improvement_rounds"],
                "status": campaign["status"], "saved_at": now(),
            }
            self.store.put("campaigns", campaign)
            self.store.event("CAMPAIGN_CHECKPOINT", campaign_id, data=campaign["checkpoint"])
            return campaign
        return self.control._write(request_id, "campaign_checkpoint", {"campaign_id": campaign_id, "leader": leader}, write)

    def stop(self, campaign_id, reason="USER_STOP", leader="rsi-leader-01", request_id=None):
        def write():
            campaign = self.store.require("campaigns", campaign_id)
            self._leader_and_active(campaign, leader)
            campaign["status"], campaign["stop_reason"] = "STOPPED", reason
            campaign["updated_at"] = now()
            for window_id in campaign.get("window_ids", []):
                window = self.store.require("windows", window_id)
                if window["status"] == "CLOSED":
                    continue
                for reservation_id in window["reservation_ids"]:
                    reservation = self.store.require("reservations", reservation_id)
                    if reservation["status"] in {"PENDING", "ACTIVE"}:
                        reservation["status"] = "CANCELLED"
                        self.store.put("reservations", reservation)
                window["status"] = "CANCELLED"
                self.store.put("windows", window)
            self.store.put("campaigns", campaign)
            self.store.event("CAMPAIGN_STOPPED", campaign_id, after="STOPPED", reason=reason)
            return campaign
        return self.control._write(request_id, "campaign_stop", {"campaign_id": campaign_id, "reason": reason, "leader": leader}, write)

    def joint_batch(self, campaign_id, training_candidate_ids, inference_candidate_ids,
                    max_combinations=4, request_id=None):
        payload = {"campaign_id": campaign_id, "training_candidate_ids": training_candidate_ids,
                   "inference_candidate_ids": inference_candidate_ids, "max_combinations": max_combinations}
        def write():
            campaign = self.store.require("campaigns", campaign_id)
            if campaign["level"] != "E2E" or campaign["role"] != "RL_INTEGRATION":
                raise ConflictError("joint candidate batches belong to RL integration E2E campaigns")
            training = [self.store.require("candidates", cid) for cid in training_candidate_ids]
            inference = [self.store.require("candidates", cid) for cid in inference_candidate_ids]
            for candidate in training + inference:
                if candidate["status"] not in {"ACCEPTED", "SUPERSEDED"}:
                    raise ConflictError("joint candidates must pass independent screening first")
            baseline = self.store.require("evidence", campaign["baseline"]["evidence_id"])
            batch = []
            for train in training:
                for infer in inference:
                    train_version = train["payload"].get("contract_version")
                    infer_version = infer["payload"].get("contract_version")
                    if train_version != infer_version or train_version != baseline.get("contract_version"):
                        continue
                    batch.append({"training_candidate_id": train["candidate_id"],
                                  "inference_candidate_id": infer["candidate_id"],
                                  "contract_version": train_version})
                    if len(batch) >= max_combinations:
                        return batch
            return batch
        return self.control._write(request_id, "campaign_joint_batch", payload, write)
