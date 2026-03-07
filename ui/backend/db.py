"""SQLite persistence for evaluation runs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "evaluations.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                run_id       TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'running',
                target_model TEXT NOT NULL,
                concern      TEXT NOT NULL,
                config_json  TEXT NOT NULL,
                report_json  TEXT,
                error        TEXT
            )
        """)
        conn.commit()


def insert_evaluation(run_id: str, created_at: str, target_model: str, concern: str, config: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO evaluations (run_id, created_at, status, target_model, concern, config_json) VALUES (?,?,?,?,?,?)",
            (run_id, created_at, "running", target_model, concern, json.dumps(config)),
        )
        conn.commit()


def complete_evaluation(run_id: str, report: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE evaluations SET status='completed', report_json=? WHERE run_id=?",
            (json.dumps(report), run_id),
        )
        conn.commit()


def fail_evaluation(run_id: str, error: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE evaluations SET status='failed', error=? WHERE run_id=?",
            (error, run_id),
        )
        conn.commit()


def list_evaluations() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT run_id, created_at, status, target_model, concern, error FROM evaluations ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_evaluation(run_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE run_id=?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if result.get("report_json"):
        result["report"] = json.loads(result["report_json"])
    if result.get("config_json"):
        result["config"] = json.loads(result["config_json"])
    return result
