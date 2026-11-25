"""SQLite document store with transactional writes, idempotency and append-only events."""

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager

from .models import ConflictError, canonical, new_id, now


TABLES = (
    "tasks", "steps", "nodes", "leases", "reservations", "evidence",
    "model_specs", "model_artifacts", "rl_specs", "resource_groups",
    "executions", "campaigns", "candidates", "evaluations", "windows", "profiles",
    "capabilities", "requests",
)


class Store:
    def __init__(self, path="ascend.db"):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA busy_timeout=5000")
            for table in TABLES:
                self.conn.execute(
                    "CREATE TABLE IF NOT EXISTS %s "
                    "(id TEXT PRIMARY KEY, kind TEXT, data TEXT NOT NULL, updated_at REAL NOT NULL)" % table
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_%s_kind ON %s(kind)" % (table, table)
                )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS events "
                "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, "
                "event_type TEXT, entity_id TEXT, actor TEXT, reason TEXT, data TEXT, created_at REAL)"
            )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def close(self):
        with self.lock:
            self.conn.close()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def put(self, table, document, kind=None):
        if table not in TABLES:
            raise ValueError("unknown table")
        id_field = "request_id" if table == "requests" else table[:-1] + "_id"
        if table == "evidence":
            id_field = "evidence_id"
        elif table == "model_specs":
            id_field = "model_spec_id"
        elif table == "model_artifacts":
            id_field = "artifact_id"
        elif table == "rl_specs":
            id_field = "rl_spec_id"
        elif table == "resource_groups":
            id_field = "resource_group_id"
        elif table == "capabilities":
            id_field = "capability_id"
        elif table == "evaluations":
            id_field = "evaluation_id"
        elif table == "windows":
            id_field = "window_id"
        elif table == "profiles":
            id_field = "profile_id"
        entity_id = document[id_field]
        self.conn.execute(
            "INSERT INTO %s(id, kind, data, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,data=excluded.data,updated_at=excluded.updated_at" % table,
            (entity_id, kind, canonical(document), now()),
        )
        return document

    def get(self, table, entity_id):
        row = self.conn.execute("SELECT data FROM %s WHERE id=?" % table, (entity_id,)).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def require(self, table, entity_id):
        from .models import NotFoundError
        value = self.get(table, entity_id)
        if value is None:
            raise NotFoundError("%s not found: %s" % (table, entity_id))
        return value

    def all(self, table, kind=None):
        if kind is None:
            rows = self.conn.execute("SELECT data FROM %s" % table).fetchall()
        else:
            rows = self.conn.execute("SELECT data FROM %s WHERE kind=?" % table, (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def event(self, event_type, entity_id, actor="control-plane", reason="", before=None, after=None, data=None):
        event = {
            "event_id": new_id("event"),
            "event_type": event_type,
            "entity_id": entity_id,
            "actor": actor,
            "reason": reason,
            "input_digest": None,
            "before": before,
            "after": after,
            "data": data or {},
            "created_at": now(),
        }
        self.conn.execute(
            "INSERT INTO events(event_id,event_type,entity_id,actor,reason,data,created_at) VALUES(?,?,?,?,?,?,?)",
            (event["event_id"], event_type, entity_id, actor, reason, canonical(event), event["created_at"]),
        )
        return event

    def events(self, limit=200):
        rows = self.conn.execute("SELECT data FROM events ORDER BY sequence DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, canonical(value)),
        )

    def next_epoch(self):
        epoch = int(self.meta("epoch", 0)) + 1
        self.set_meta("epoch", epoch)
        return epoch

    def write_idempotent(self, request_id, operation, payload, callback):
        if not request_id:
            raise ConflictError("request_id is required for write operations")
        with self.transaction():
            previous = self.get("requests", request_id)
            if previous:
                if previous["operation"] != operation or previous["payload_digest"] != hashlib.sha256(canonical(payload).encode()).hexdigest():
                    raise ConflictError("request_id was already used with different input")
                return previous["response"]
            response = callback()
            self.put("requests", {
                "request_id": request_id,
                "operation": operation,
                "payload_digest": hashlib.sha256(canonical(payload).encode()).hexdigest(),
                "response": response,
                "created_at": now(),
            })
            return response
