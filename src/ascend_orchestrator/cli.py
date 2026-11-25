"""Command-line entry point."""

import argparse
import json
import sys

from .api import serve
from .control import ControlPlane
from .demo import run_demo


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ascend-orchestrator", description="Ascend phase-one prototype")
    parser.add_argument("--db", help="SQLite state path (default: ascend.db; demo: var/demo.db)")
    parser.add_argument("--workspace", help="workspace root (default: var/workspaces; demo: var/demo-workspaces)")
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve", help="start local HTTP API")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    demo = sub.add_parser("demo", help="run simulated operator and RL closed-loop demo")
    demo.add_argument("--reset", action="store_true", help="remove demo database before running")
    sub.add_parser("status", help="print cluster and queue status")
    sub.add_parser("events", help="print append-only audit events")
    args = parser.parse_args(argv)
    is_demo = args.command == "demo"
    db_path = args.db or ("var/demo.db" if is_demo else "ascend.db")
    workspace = args.workspace or ("var/demo-workspaces" if is_demo else "var/workspaces")
    if args.command == "serve":
        serve(args.host, args.port, db_path, workspace)
        return 0
    if is_demo:
        try:
            result = run_demo(db_path, workspace, reset=args.reset)
        except FileExistsError as error:
            parser.exit(1, "error: %s\n" % error)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    control = ControlPlane(db_path, workspace)
    try:
        result = control.cluster_status() if args.command == "status" else control.events()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        control.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
