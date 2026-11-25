"""Dependency-free HTTP JSON API for local prototype integration."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from .control import ControlPlane
from .models import ConflictError, ContractError, FencingError, NotFoundError


class ApiHandler(BaseHTTPRequestHandler):
    control = None

    def _send(self, status, value):
        body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size) or b"{}")

    def do_GET(self):
        try:
            path = urlparse(self.path).path.strip("/").split("/")
            if path == ["health"]:
                result = {"status": "ok", "prototype": True, "runner": "simulated"}
            elif path == ["cluster"]:
                result = self.control.cluster_status()
            elif path == ["events"]:
                result = self.control.events()
            elif path == ["capabilities"]:
                result = self.control.list_capabilities()
            elif len(path) == 2 and path[0] == "tasks":
                result = self.control.get_task(path[1])
            elif len(path) == 2 and path[0] == "evidence":
                result = self.control.evidence(path[1])
            elif len(path) == 2 and path[0] == "reservations":
                result = self.control.reservation_status(path[1])
            else:
                return self._send(404, {"error": "not found"})
            self._send(200, result)
        except Exception as error:
            self._error(error)

    def do_POST(self):
        try:
            data = self._body()
            path = urlparse(self.path).path.strip("/").split("/")
            route = "/".join(path)
            c = self.control
            routes = {
                "executors/register": lambda: c.register_executor(**data),
                "executors/heartbeat": lambda: c.heartbeat_executor(**data),
                "nodes/lifecycle": lambda: c.set_node_lifecycle(**data),
                "nodes/reconciled": lambda: c.confirm_node_reconciled(**data),
                "models/specs": lambda: c.register_model_spec(
                    {k: v for k, v in data.items() if k != "request_id"}, request_id=data.get("request_id")),
                "models/artifacts": lambda: c.register_model_artifact(**data),
                "rl/specs": lambda: c.register_rl_spec(
                    {k: v for k, v in data.items() if k != "request_id"}, request_id=data.get("request_id")),
                "rl/freeze": lambda: c.freeze_rl_contract(**data),
                "rl/update": lambda: c.update_rl_contract(**data),
                "tasks": lambda: c.create_task(**data),
                "tasks/steps": lambda: c.add_steps(**data),
                "tasks/missing-operators": lambda: c.report_missing_operators(**data),
                "schedule": lambda: c.schedule(),
                "reservations": lambda: c.request_reservation(**data),
                "resource-groups": lambda: c.create_resource_group(**data),
                "steps/run": lambda: c.run_step(**data),
                "steps/complete": lambda: c.complete_step(**data),
                "leases/heartbeat": lambda: c.heartbeat_lease(**data),
                "leases/release": lambda: c.release_lease(**data),
                "leases/cleanup": lambda: c.confirm_cleanup(**data),
                "capabilities/invoke": lambda: c.invoke_capability(**data),
                "recover": lambda: c.recover(),
            }
            if route in routes:
                result = routes[route]()
            elif len(path) == 3 and path[0] == "tasks" and path[2] in {"pause", "resume", "cancel"}:
                result = getattr(c, path[2] + "_task")(path[1], request_id=data.get("request_id"))
            else:
                return self._send(404, {"error": "not found"})
            self._send(200, result)
        except Exception as error:
            self._error(error)

    def _error(self, error):
        if isinstance(error, NotFoundError):
            status = 404
        elif isinstance(error, (ConflictError, FencingError)):
            status = 409
        elif isinstance(error, (ContractError, KeyError, TypeError, json.JSONDecodeError)):
            status = 400
        else:
            status = 500
        self._send(status, {"error": str(error), "type": type(error).__name__})

    def log_message(self, format_string, *args):
        pass


def serve(host="127.0.0.1", port=8765, db_path="ascend.db", workspace_root="var/workspaces"):
    control = ControlPlane(db_path, workspace_root)
    handler = type("BoundApiHandler", (ApiHandler,), {"control": control})
    server = HTTPServer((host, port), handler)
    try:
        print("Ascend Orchestrator prototype listening on http://%s:%d" % (host, port), flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        control.close()
