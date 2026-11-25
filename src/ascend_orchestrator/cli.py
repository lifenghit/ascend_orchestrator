"""Command-line entry point."""

import argparse
import json
import sys

from .api import serve
from .control import ControlPlane
from .demo import run_demo
from .rsi_demo import run_rsi_demo


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ascend-orchestrator", description="Ascend orchestration and RSI prototype")
    parser.add_argument("--db", help="SQLite state path")
    parser.add_argument("--workspace", help="isolated workspace root")
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve", help="start local HTTP API")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    for name, help_text in (("demo", "run operator and RL closed-loop demo"),
                            ("rsi-demo", "run operator, model and E2E RSI demo")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--reset", action="store_true", help="replace the demo database")
    dashboard = sub.add_parser("dashboard", help="generate a standalone HTML metrics dashboard")
    dashboard.add_argument("--output", default="var/dashboard.html")
    sub.add_parser("metrics", help="print resource, task and RSI metrics")
    sub.add_parser("tuning", help="print scheduler tuning recommendations")
    sub.add_parser("status", help="print cluster and queue status")
    sub.add_parser("events", help="print append-only audit events")
    args = parser.parse_args(argv)
    is_demo = args.command in {"demo", "rsi-demo"}
    default_db = "var/%s.db" % ("demo" if args.command == "demo" else "rsi-demo") if is_demo else "ascend.db"
    default_workspace = "var/%s-workspaces" % ("demo" if args.command == "demo" else "rsi-demo") if is_demo else "var/workspaces"
    db_path = args.db or default_db
    workspace = args.workspace or default_workspace
    if args.command == "serve":
        serve(args.host, args.port, db_path, workspace)
        return 0
    if is_demo:
        try:
            runner = run_demo if args.command == "demo" else run_rsi_demo
            result = runner(db_path, workspace, reset=args.reset)
        except FileExistsError as error:
            parser.exit(1, "error: %s\n" % error)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    control = ControlPlane(db_path, workspace)
    try:
        if args.command == "dashboard":
            result = control.dashboard_generate(args.output)
        elif args.command == "metrics":
            result = control.metrics_snapshot()
        elif args.command == "tuning":
            result = control.scheduler_tuning()
        elif args.command == "status":
            result = control.cluster_status()
        else:
            result = control.events()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        control.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
