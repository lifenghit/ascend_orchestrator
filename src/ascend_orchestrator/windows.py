"""Limited-time batch measurement windows for model and E2E campaigns."""

from .models import ConflictError, ContractError, new_id, now


class WindowManager:
    def __init__(self, control):
        self.control = control
        self.store = control.store

    def create(self, campaign_id, step_ids, starts_at, duration_seconds, guard_seconds=30,
               isolation="GLOBAL_QUIET", request_id=None):
        payload = {"campaign_id": campaign_id, "step_ids": step_ids, "starts_at": starts_at,
                   "duration_seconds": duration_seconds, "guard_seconds": guard_seconds, "isolation": isolation}
        def write():
            campaign = self.store.require("campaigns", campaign_id)
            if campaign["status"] != "ACTIVE" or not step_ids:
                raise ConflictError("measurement window requires an active campaign and steps")
            if isolation not in {"GLOBAL_QUIET", "NODE_SET_QUIET"}:
                raise ContractError("unsupported measurement isolation")
            reservations = []
            for step_id in step_ids:
                step = self.store.require("steps", step_id)
                if step["task_id"] != campaign["task_id"] or step["status"] != "READY":
                    raise ConflictError("window steps must be READY and belong to the campaign task")
                if step["resource"].get("performance_isolation") != isolation:
                    raise ConflictError("step isolation does not match the requested window")
                reservation = self.control.scheduler.request_reservation(step_id, starts_at, guard_seconds)
                reservations.append(reservation["reservation_id"])
            window = {
                "window_id": new_id("window"), "campaign_id": campaign_id, "task_id": campaign["task_id"],
                "step_ids": list(step_ids), "reservation_ids": reservations,
                "starts_at": starts_at, "ends_at": starts_at + duration_seconds,
                "guard_seconds": guard_seconds, "isolation": isolation,
                "status": "SCHEDULED", "created_at": now(),
            }
            self.store.put("windows", window)
            campaign["window_ids"].append(window["window_id"])
            self.store.put("campaigns", campaign)
            self.store.event("MEASUREMENT_WINDOW_CREATED", window["window_id"], after="SCHEDULED",
                             data={"step_count": len(step_ids), "isolation": isolation})
            return window
        return self.control._write(request_id, "measurement_window_create", payload, write)

    def close(self, window_id, request_id=None):
        def write():
            window = self.store.require("windows", window_id)
            if window["status"] == "CLOSED":
                return window
            active = []
            for step_id in window["step_ids"]:
                step = self.store.require("steps", step_id)
                if step["status"] == "RUNNING":
                    active.append(step_id)
            if active:
                raise ConflictError("cannot close window with running steps: %s" % ", ".join(active))
            for reservation_id in window["reservation_ids"]:
                reservation = self.store.require("reservations", reservation_id)
                if reservation["status"] in {"PENDING", "ACTIVE"}:
                    reservation["status"] = "CANCELLED"
                    self.store.put("reservations", reservation)
            window["status"] = "CLOSED"
            window["closed_at"] = now()
            self.store.put("windows", window)
            campaign = self.store.require("campaigns", window["campaign_id"])
            campaign["checkpoint"] = {
                "rounds": campaign["rounds"], "best_metric": campaign["best_metric"],
                "accepted_candidate_id": campaign["accepted_candidate_id"],
                "no_improvement_rounds": campaign["no_improvement_rounds"],
                "status": campaign["status"], "saved_at": now(), "window_id": window_id,
            }
            self.store.put("campaigns", campaign)
            self.store.event("MEASUREMENT_WINDOW_CLOSED", window_id, after="CLOSED")
            return window
        return self.control._write(request_id, "measurement_window_close", {"window_id": window_id}, write)
