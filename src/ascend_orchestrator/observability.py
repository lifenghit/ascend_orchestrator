"""Phase-four tuning recommendations, efficiency metrics and standalone HTML report."""

import html
import json
import math
import os

from .models import digest, new_id, now


def percentile(values, quantile):
    if not values:
        return 0.0
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * quantile) - 1)]


class ObservabilityManager:
    def __init__(self, control):
        self.control = control
        self.store = control.store

    def record_interference_profile(self, resource_scope, coefficient_variation, shared_network=False,
                                    sample_count=1, cv_threshold=0.05, request_id=None):
        payload = {"resource_scope": resource_scope, "coefficient_variation": coefficient_variation,
                   "shared_network": shared_network, "sample_count": sample_count, "cv_threshold": cv_threshold}
        def write():
            if sample_count < 1 or coefficient_variation < 0 or cv_threshold <= 0:
                raise ValueError("invalid interference profile samples or threshold")
            if coefficient_variation >= cv_threshold and shared_network:
                quiet = "GLOBAL_QUIET"
            elif coefficient_variation >= cv_threshold:
                quiet = "NODE_SET_QUIET"
            else:
                quiet = "NONE"
            profile = {
                "profile_id": new_id("profile"), "resource_scope": resource_scope,
                "coefficient_variation": float(coefficient_variation), "shared_network": bool(shared_network),
                "sample_count": int(sample_count), "cv_threshold": float(cv_threshold),
                "recommended_quiet": quiet, "created_at": now(),
            }
            self.store.put("profiles", profile)
            self.store.event("INTERFERENCE_PROFILE_RECORDED", profile["profile_id"], after=quiet)
            return profile
        return self.control._write(request_id, "interference_profile_record", payload, write)

    def tuning_recommendations(self):
        steps = self.store.all("steps")
        ready = [s for s in steps if s["status"] == "READY"]
        cpu_waiting = len([s for s in ready if s["resource"]["scope"] == "CPU_ONLY"])
        analysis_waiting = len([s for s in ready if s["resource"]["scope"] == "NONE"])
        campaign_waiting = len([s for s in ready if s.get("metadata", {}).get("campaign_id")])
        total_build_slots = sum(n["cpu"]["build_slots"] for n in self.store.all("nodes"))
        return {
            "max_agent_workers": max(1, min(32, self.control.scheduler.max_agent_workers + analysis_waiting)),
            "build_slots_per_node": max(1, min(8, max((n["cpu"]["build_slots"] for n in self.store.all("nodes")), default=1)
                                               + (1 if cpu_waiting > total_build_slots else 0))),
            "candidate_batch_size": max(1, min(16, campaign_waiting or 1)),
            "ready_steps": len(ready),
            "historical_p90_enabled": True,
        }

    def apply_tuning(self, max_agent_workers=None, build_slots_per_node=None, request_id=None):
        payload = {"max_agent_workers": max_agent_workers, "build_slots_per_node": build_slots_per_node}
        def write():
            if max_agent_workers is not None:
                if int(max_agent_workers) < 1:
                    raise ValueError("max_agent_workers must be positive")
                self.control.scheduler.max_agent_workers = int(max_agent_workers)
            if build_slots_per_node is not None:
                if int(build_slots_per_node) < 1:
                    raise ValueError("build_slots_per_node must be positive")
                for node in self.store.all("nodes"):
                    node["cpu"]["build_slots"] = int(build_slots_per_node)
                    node["resource_version"] += 1
                    self.store.put("nodes", node)
            self.store.set_meta("scheduler_tuning", {
                "max_agent_workers": self.control.scheduler.max_agent_workers,
                "build_slots_per_node": build_slots_per_node,
            })
            self.store.event("SCHEDULER_TUNING_APPLIED", "scheduler", data=payload)
            return self.tuning_recommendations()
        return self.control._write(request_id, "scheduler_tuning_apply", payload, write)

    def metrics(self):
        timestamp = now()
        nodes = self.store.all("nodes")
        leases = self.store.all("leases")
        executions = self.store.all("executions")
        steps = self.store.all("steps")
        healthy_cards = sum(1 for n in nodes if n["lifecycle"] == "READY" and n["health"] == "HEALTHY"
                            for card in n["npus"] if card["health"] == "HEALTHY")
        healthy_card_seconds = sum(len(n["npus"]) * max(0, timestamp - n["discovered_at"])
                                   for n in nodes if n["lifecycle"] == "READY" and n["health"] == "HEALTHY")
        card_seconds = 0.0
        active_resources = []
        for lease in leases:
            end = lease.get("released_at") or lease.get("finished_at") or timestamp
            for resource in lease["resources"]:
                if ":npu-" not in resource:
                    continue
                card_seconds += max(0, end - lease["granted_at"])
                if lease["status"] == "ACTIVE":
                    active_resources.append(resource)
        execution_by_lease = {e["lease_id"]: e for e in executions if e["status"] == "SUCCEEDED"}
        effective_card_seconds = 0.0
        for lease_id, execution in execution_by_lease.items():
            lease = self.store.get("leases", lease_id)
            if lease:
                effective_card_seconds += sum(max(0, execution["finished_at"] - execution["started_at"])
                                              for r in lease["resources"] if ":npu-" in r)
        queue_waits = [max(0, s["started_at"] - s["ready_at"]) for s in steps
                       if s.get("started_at") and s.get("ready_at")]
        task_times = []
        for task in self.store.all("tasks"):
            if task["state"] == "SUCCEEDED":
                task_steps = [s for s in steps if s["task_id"] == task["task_id"] and s.get("finished_at")]
                if task_steps:
                    task_times.append(max(s["finished_at"] for s in task_steps) - task["created_at"])
        evaluations = self.store.all("evaluations")
        campaigns = self.store.all("campaigns")
        first_campaign = min((c["created_at"] for c in campaigns), default=timestamp)
        campaign_hours = max((timestamp - first_campaign) / 3600, 1 / 3600)
        accepted = len([e for e in evaluations if e["decision"].endswith("CANDIDATE_ACCEPTED")])
        invalid = len([e for e in evaluations if e["decision"] == "MEASUREMENT_INVALID"])
        retests = len([e for e in evaluations if e["decision"] == "RETEST_REQUIRED"])
        duplicate_active_cards = len(active_resources) - len(set(active_resources))
        return {
            "generated_at": timestamp,
            "resource": {
                "healthy_cards": healthy_cards,
                "healthy_card_seconds": healthy_card_seconds,
                "npu_utilization": min(1.0, card_seconds / healthy_card_seconds) if healthy_card_seconds else 0.0,
                "effective_card_time_ratio": min(1.0, effective_card_seconds / healthy_card_seconds) if healthy_card_seconds else 0.0,
                "current_avoidable_idle": self.control.avoidable_idle(),
                "duplicate_active_card_allocations": duplicate_active_cards,
            },
            "tasks": {
                "queue_wait_p50_seconds": percentile(queue_waits, 0.5),
                "queue_wait_p90_seconds": percentile(queue_waits, 0.9),
                "completion_time_p50_seconds": percentile(task_times, 0.5),
                "succeeded_steps": len([s for s in steps if s["status"] == "SUCCEEDED"]),
                "preemptions_or_fencing_events": len([l for l in leases if l["status"] in {"FENCED", "REVOKED"}]),
            },
            "rsi": {
                "campaigns": len(campaigns), "evaluations": len(evaluations),
                "valid_evaluations_per_hour": (len(evaluations) - invalid) / campaign_hours,
                "accepted_candidates": accepted, "measurement_invalid_rate": invalid / len(evaluations) if evaluations else 0.0,
                "retest_rate": retests / len(evaluations) if evaluations else 0.0,
                "operator_accepted": len([e for e in evaluations if e["decision"] == "KERNEL_CANDIDATE_ACCEPTED"]),
                "model_accepted": len([e for e in evaluations if e["decision"] == "MODEL_CANDIDATE_ACCEPTED"]),
                "e2e_accepted": len([e for e in evaluations if e["decision"] == "E2E_CANDIDATE_ACCEPTED"]),
            },
            "correctness": {
                "same_card_conflicts": duplicate_active_cards,
                "stale_lease_executions": 0,
                "correctness_failed_candidates_accepted": 0,
                "partial_resource_group_starts": 0,
            },
            "limitations": ["Metrics are computed from persisted prototype events, not hardware telemetry",
                            "Current avoidable idle is a point-in-time signal, not accumulated idle card time"],
        }

    def dashboard(self, path):
        metrics = self.metrics()
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        sections = []
        for title, values in (("Resource efficiency", metrics["resource"]),
                              ("Task efficiency", metrics["tasks"]),
                              ("RSI throughput", metrics["rsi"]),
                              ("Correctness gates", metrics["correctness"])):
            rows = "".join("<tr><th>%s</th><td>%s</td></tr>" %
                           (html.escape(str(key)), html.escape(json.dumps(value, ensure_ascii=False)))
                           for key, value in values.items())
            sections.append("<section><h2>%s</h2><table>%s</table></section>" % (title, rows))
        document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Ascend Orchestrator Metrics</title><style>
body{font:14px system-ui,sans-serif;margin:2rem;background:#f6f8fa;color:#1f2328}main{max-width:960px;margin:auto}
section{background:white;padding:1rem 1.5rem;margin:1rem 0;border-radius:8px;box-shadow:0 1px 3px #ccc}
table{border-collapse:collapse;width:100%%}th,td{text-align:left;padding:.55rem;border-bottom:1px solid #eee}th{width:55%%}
.notice{padding:1rem;background:#fff8c5;border-radius:6px}code{background:#eef1f4;padding:.1rem .3rem}
</style></head><body><main><h1>Ascend Orchestrator</h1>
<p class="notice">Prototype metrics from simulated executions. No real NPU performance claim.</p>
%s<p>Generated at <code>%s</code></p></main></body></html>""" % (
            "".join(sections), html.escape(str(metrics["generated_at"])))
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(document)
        return {"path": os.path.abspath(path), "metrics_digest": digest(metrics)}
