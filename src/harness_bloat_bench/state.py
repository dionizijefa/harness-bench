"""Durable campaign, combination, and task-execution state.

The state database is the source of truth for what was planned and what is
currently executing. Dagster run IDs and remote job IDs are physical execution
identifiers; the deterministic combination and task IDs in this module are the
logical identities used to prevent duplicate work.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


CAMPAIGN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
TERMINAL_EXECUTION_STATES = frozenset(
    {"completed", "error", "timed_out", "cancelled"}
)
TERMINAL_COMBINATION_STATES = frozenset({"completed", "error", "cancelled"})


class CampaignConflict(ValueError):
    """An existing campaign ID was reused with a different immutable plan."""


class StateConflict(RuntimeError):
    """A logical execution was bound to conflicting physical execution state."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def state_database_path() -> Path:
    configured = os.environ.get("HARNESS_BLOAT_STATE_DB")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (Path.cwd() / "outputs" / "results.sqlite").resolve()
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def new_campaign_id() -> str:
    return f"campaign_{uuid.uuid4().hex}"


def validate_campaign_id(campaign_id: str) -> str:
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        raise ValueError(
            "campaign_id must start with an alphanumeric character, contain only "
            "letters, numbers, '.', '_' or '-', and be at most 200 characters"
        )
    return campaign_id


def _deterministic_id(prefix: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()[:24]
    return f"{prefix}_{digest}"


def combination_id(
    campaign_id: str,
    model: str,
    harness: str,
    harness_version: str,
    repetition: int,
) -> str:
    return _deterministic_id(
        "combination",
        {
            "campaign_id": campaign_id,
            "model": model,
            "harness": harness,
            "harness_version": harness_version,
            "repetition": repetition,
        },
    )


def task_execution_id(combination_id: str, task_id: str) -> str:
    return _deterministic_id(
        "task",
        {"combination_id": combination_id, "task_id": task_id},
    )


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def ensure_state_schema(path: Path) -> Path:
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_state_schema(connection)
    finally:
        connection.close()
    return path


def _ensure_state_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_campaigns (
            campaign_id TEXT PRIMARY KEY,
            batch_id TEXT,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            planner_run_id TEXT NOT NULL,
            dataset TEXT NOT NULL,
            task_selection_mode TEXT NOT NULL,
            task_ids_json TEXT NOT NULL,
            task_count INTEGER NOT NULL,
            combination_count INTEGER NOT NULL,
            task_execution_count INTEGER NOT NULL,
            manifest_hash TEXT NOT NULL,
            config_json TEXT NOT NULL,
            error_type TEXT,
            error_message TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_combinations (
            combination_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL REFERENCES benchmark_campaigns(campaign_id),
            model TEXT NOT NULL,
            harness TEXT NOT NULL,
            harness_version TEXT NOT NULL,
            repetition INTEGER NOT NULL,
            state TEXT NOT NULL,
            expected_task_count INTEGER NOT NULL,
            dagster_run_id TEXT,
            run_config_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            queued_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            error_type TEXT,
            error_message TEXT,
            UNIQUE (campaign_id, model, harness, harness_version, repetition)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_task_executions (
            task_execution_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL REFERENCES benchmark_campaigns(campaign_id),
            combination_id TEXT NOT NULL
                REFERENCES benchmark_combinations(combination_id),
            task_id TEXT NOT NULL,
            state TEXT NOT NULL,
            outcome TEXT,
            dagster_run_id TEXT,
            dagster_step_key TEXT,
            remote_job_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            queued_at TEXT,
            started_at TEXT,
            heartbeat_at TEXT,
            finished_at TEXT,
            error_type TEXT,
            error_message TEXT,
            UNIQUE (combination_id, task_id)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS benchmark_campaign_state "
        "ON benchmark_campaigns (state, created_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS benchmark_combination_campaign_state "
        "ON benchmark_combinations (campaign_id, state)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS benchmark_task_combination_state "
        "ON benchmark_task_executions (combination_id, state, task_id)"
    )


def create_campaign(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    planner_run_id: str,
    combinations: Iterable[Mapping[str, Any]],
) -> bool:
    """Persist a complete immutable plan, returning False when it already exists."""

    campaign_id = validate_campaign_id(str(manifest["campaign_id"]))
    serialized_manifest = canonical_json(manifest)
    fingerprint = hashlib.sha256(serialized_manifest.encode()).hexdigest()
    planned = list(combinations)
    now = utc_now()
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_state_schema(connection)
            existing = connection.execute(
                "SELECT manifest_hash FROM benchmark_campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if existing is not None:
                if existing["manifest_hash"] != fingerprint:
                    raise CampaignConflict(
                        f"campaign {campaign_id!r} already exists with a different plan"
                    )
                return False

            task_ids = list(manifest["task_ids"])
            task_count = len(task_ids)
            connection.execute(
                """
                INSERT INTO benchmark_campaigns (
                    campaign_id, batch_id, state, created_at, updated_at,
                    planner_run_id, dataset, task_selection_mode, task_ids_json,
                    task_count, combination_count, task_execution_count,
                    manifest_hash, config_json
                ) VALUES (?, ?, 'planned', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    manifest.get("batch_id"),
                    now,
                    now,
                    planner_run_id,
                    manifest["dataset"],
                    manifest["task_selection_mode"],
                    canonical_json(task_ids),
                    task_count,
                    len(planned),
                    len(planned) * task_count,
                    fingerprint,
                    serialized_manifest,
                ),
            )
            for item in planned:
                combination_key = str(item["combination_id"])
                connection.execute(
                    """
                    INSERT INTO benchmark_combinations (
                        combination_id, campaign_id, model, harness,
                        harness_version, repetition, state, expected_task_count,
                        run_config_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?)
                    """,
                    (
                        combination_key,
                        campaign_id,
                        item["model"],
                        item["harness"],
                        item["harness_version"],
                        item["repetition"],
                        task_count,
                        canonical_json(item["run_config"]),
                        now,
                        now,
                    ),
                )
                for task_id in task_ids:
                    execution_key = task_execution_id(combination_key, task_id)
                    connection.execute(
                        """
                        INSERT INTO benchmark_task_executions (
                            task_execution_id, campaign_id, combination_id,
                            task_id, state, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'planned', ?, ?)
                        """,
                        (
                            execution_key,
                            campaign_id,
                            combination_key,
                            task_id,
                            now,
                            now,
                        ),
                    )
    finally:
        connection.close()
    return True


def launchable_combinations(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    connection = _connect(path)
    try:
        _ensure_state_schema(connection)
        rows = connection.execute(
            """
            SELECT * FROM benchmark_combinations
            WHERE state IN ('planned', 'queued') AND dagster_run_id IS NULL
            ORDER BY created_at, combination_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def nonterminal_task_ids(path: Path, combination_key: str) -> list[str]:
    """Return only logical tasks that still need execution for a combination."""

    connection = _connect(path)
    try:
        _ensure_state_schema(connection)
        rows = connection.execute(
            """
            SELECT task_id
            FROM benchmark_task_executions
            WHERE combination_id = ?
              AND state NOT IN ('completed', 'error', 'timed_out', 'cancelled')
            ORDER BY task_id
            """,
            (combination_key,),
        ).fetchall()
        return [str(row["task_id"]) for row in rows]
    finally:
        connection.close()


def validate_combination_config(
    path: Path, combination_key: str, child_config: Mapping[str, Any]
) -> None:
    connection = _connect(path)
    try:
        row = connection.execute(
            "SELECT run_config_json FROM benchmark_combinations "
            "WHERE combination_id = ?",
            (combination_key,),
        ).fetchone()
        if row is None:
            raise StateConflict(f"unknown combination {combination_key}")
        stored = json.loads(row["run_config_json"])
        expected = stored["ops"]["plan_task_executions"]["config"]
        if canonical_json(expected) != canonical_json(child_config):
            raise StateConflict(
                f"combination {combination_key} was launched with configuration "
                "that differs from its immutable campaign manifest"
            )
    finally:
        connection.close()


def active_combinations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    connection = _connect(path)
    try:
        _ensure_state_schema(connection)
        rows = connection.execute(
            """
            SELECT * FROM benchmark_combinations
            WHERE state NOT IN ('completed', 'error', 'cancelled')
              AND (dagster_run_id IS NOT NULL OR state = 'queued')
            ORDER BY created_at, combination_id
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def mark_combination_queued(path: Path, combination_key: str) -> None:
    now = utc_now()
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_state_schema(connection)
            connection.execute(
                """
                UPDATE benchmark_combinations
                SET state = CASE WHEN state = 'planned' THEN 'queued' ELSE state END,
                    queued_at = COALESCE(queued_at, ?), updated_at = ?
                WHERE combination_id = ?
                  AND state IN ('planned', 'queued')
                """,
                (now, now, combination_key),
            )
            _refresh_campaign_for_combination(connection, combination_key, now)
    finally:
        connection.close()


def bind_combination_run(
    path: Path,
    combination_key: str,
    dagster_run_id: str,
    *,
    started: bool,
) -> None:
    now = utc_now()
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_state_schema(connection)
            row = connection.execute(
                "SELECT dagster_run_id, state FROM benchmark_combinations "
                "WHERE combination_id = ?",
                (combination_key,),
            ).fetchone()
            if row is None:
                raise StateConflict(f"unknown combination {combination_key}")
            existing_run_id = row["dagster_run_id"]
            if existing_run_id not in (None, dagster_run_id):
                raise StateConflict(
                    f"combination {combination_key} is already bound to Dagster run "
                    f"{existing_run_id}"
                )
            if row["state"] in TERMINAL_COMBINATION_STATES:
                raise StateConflict(
                    f"combination {combination_key} is already terminal as "
                    f"{row['state']}"
                )

            state = "running" if started else "queued"
            connection.execute(
                """
                UPDATE benchmark_combinations
                SET dagster_run_id = ?, state = ?,
                    queued_at = COALESCE(queued_at, ?),
                    started_at = CASE
                        WHEN ? THEN COALESCE(started_at, ?)
                        ELSE started_at
                    END,
                    updated_at = ?
                WHERE combination_id = ?
                """,
                (
                    dagster_run_id,
                    state,
                    now,
                    int(started),
                    now,
                    now,
                    combination_key,
                ),
            )
            if started:
                connection.execute(
                    """
                    UPDATE benchmark_task_executions
                    SET state = 'queued', queued_at = COALESCE(queued_at, ?),
                        dagster_run_id = ?, updated_at = ?
                    WHERE combination_id = ? AND state = 'planned'
                    """,
                    (now, dagster_run_id, now, combination_key),
                )
            _refresh_campaign_for_combination(connection, combination_key, now)
    finally:
        connection.close()


def mark_task_running(
    path: Path,
    execution_key: str,
    *,
    dagster_run_id: str,
    dagster_step_key: str,
    remote_job_id: str | None = None,
) -> None:
    now = utc_now()
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_state_schema(connection)
            row = connection.execute(
                "SELECT combination_id, dagster_run_id, state "
                "FROM benchmark_task_executions WHERE task_execution_id = ?",
                (execution_key,),
            ).fetchone()
            if row is None:
                raise StateConflict(f"unknown task execution {execution_key}")
            if row["dagster_run_id"] not in (None, dagster_run_id):
                raise StateConflict(
                    f"task execution {execution_key} is already bound to Dagster run "
                    f"{row['dagster_run_id']}"
                )
            if row["state"] in TERMINAL_EXECUTION_STATES:
                raise StateConflict(
                    f"task execution {execution_key} is already terminal as "
                    f"{row['state']}"
                )
            connection.execute(
                """
                UPDATE benchmark_task_executions
                SET state = 'running', dagster_run_id = ?, dagster_step_key = ?,
                    remote_job_id = COALESCE(?, remote_job_id),
                    queued_at = COALESCE(queued_at, ?),
                    started_at = COALESCE(started_at, ?),
                    heartbeat_at = ?, updated_at = ?
                WHERE task_execution_id = ?
                """,
                (
                    dagster_run_id,
                    dagster_step_key,
                    remote_job_id,
                    now,
                    now,
                    now,
                    now,
                    execution_key,
                ),
            )
            connection.execute(
                """
                UPDATE benchmark_combinations
                SET state = 'running', dagster_run_id = ?,
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE combination_id = ?
                  AND state NOT IN ('completed', 'error', 'cancelled')
                """,
                (dagster_run_id, now, now, row["combination_id"]),
            )
            _refresh_campaign_for_combination(
                connection, row["combination_id"], now
            )
    finally:
        connection.close()


def touch_task(
    path: Path,
    execution_key: str,
    *,
    remote_job_id: str | None = None,
) -> None:
    now = utc_now()
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE benchmark_task_executions
                SET heartbeat_at = ?, updated_at = ?,
                    remote_job_id = COALESCE(?, remote_job_id)
                WHERE task_execution_id = ? AND state = 'running'
                """,
                (now, now, remote_job_id, execution_key),
            )
    finally:
        connection.close()


def mark_task_terminal(
    path: Path,
    execution_key: str,
    *,
    state: str,
    outcome: str | None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    if state not in TERMINAL_EXECUTION_STATES:
        raise ValueError(f"invalid terminal task state: {state}")
    now = utc_now()
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_state_schema(connection)
            row = connection.execute(
                "SELECT combination_id, state FROM benchmark_task_executions "
                "WHERE task_execution_id = ?",
                (execution_key,),
            ).fetchone()
            if row is None:
                raise StateConflict(f"unknown task execution {execution_key}")
            if row["state"] in TERMINAL_EXECUTION_STATES:
                if row["state"] != state:
                    raise StateConflict(
                        f"task execution {execution_key} is already terminal as "
                        f"{row['state']}"
                    )
                return
            connection.execute(
                """
                UPDATE benchmark_task_executions
                SET state = ?, outcome = ?, error_type = ?, error_message = ?,
                    heartbeat_at = ?, finished_at = ?, updated_at = ?
                WHERE task_execution_id = ?
                """,
                (
                    state,
                    outcome,
                    error_type,
                    error_message,
                    now,
                    now,
                    now,
                    execution_key,
                ),
            )
            _refresh_combination(connection, row["combination_id"], now)
    finally:
        connection.close()


def combination_progress(path: Path, combination_key: str) -> dict[str, int | str]:
    connection = _connect(path)
    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(
                    state IN ('completed', 'error', 'timed_out', 'cancelled')
                ) AS terminal,
                SUM(state = 'planned') AS planned,
                SUM(state = 'queued') AS queued,
                SUM(state = 'running') AS running,
                SUM(state = 'completed') AS completed,
                SUM(state IN ('error', 'timed_out')) AS errors,
                SUM(state = 'cancelled') AS cancelled,
                SUM(outcome = 'passed') AS passed,
                SUM(outcome = 'failed') AS failed
            FROM benchmark_task_executions
            WHERE combination_id = ?
            """,
            (combination_key,),
        ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}
    finally:
        connection.close()


def reconcile_combination(
    path: Path,
    combination_key: str,
    *,
    dagster_run_id: str,
    dagster_status: str,
) -> None:
    """Project authoritative Dagster run state onto nonterminal work items."""

    status = dagster_status.upper()
    if status in {"QUEUED", "NOT_STARTED", "MANAGED"}:
        bind_combination_run(
            path, combination_key, dagster_run_id, started=False
        )
        return
    if status in {"STARTING", "STARTED", "CANCELING"}:
        bind_combination_run(path, combination_key, dagster_run_id, started=True)
        return

    now = utc_now()
    connection = _connect(path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            _ensure_state_schema(connection)
            row = connection.execute(
                "SELECT dagster_run_id FROM benchmark_combinations "
                "WHERE combination_id = ?",
                (combination_key,),
            ).fetchone()
            if row is None:
                return
            if row["dagster_run_id"] not in (None, dagster_run_id):
                raise StateConflict(
                    f"combination {combination_key} is already bound to Dagster run "
                    f"{row['dagster_run_id']}"
                )

            if status == "SUCCESS":
                terminal_state = "error"
                error_type = "MissingTaskResult"
                error_message = "Dagster run succeeded without a terminal task result"
            elif status in {"CANCELED"}:
                terminal_state = "cancelled"
                error_type = "DagsterRunCancelled"
                error_message = "Dagster combination run was cancelled"
            elif status == "FAILURE":
                terminal_state = "error"
                error_type = "DagsterRunFailed"
                error_message = "Dagster combination run failed"
            else:
                return

            _recover_persisted_results(connection, combination_key, now)
            updated_tasks = connection.execute(
                """
                UPDATE benchmark_task_executions
                SET state = ?, error_type = ?, error_message = ?,
                    dagster_run_id = COALESCE(dagster_run_id, ?),
                    finished_at = ?, updated_at = ?
                WHERE combination_id = ?
                  AND state NOT IN ('completed', 'error', 'timed_out', 'cancelled')
                """,
                (
                    terminal_state,
                    error_type,
                    error_message,
                    dagster_run_id,
                    now,
                    now,
                    combination_key,
                ),
            )
            connection.execute(
                """
                UPDATE benchmark_combinations
                SET dagster_run_id = ?,
                    error_type = CASE WHEN ? THEN ? ELSE error_type END,
                    error_message = CASE WHEN ? THEN ? ELSE error_message END,
                    updated_at = ?
                WHERE combination_id = ?
                """,
                (
                    dagster_run_id,
                    int(updated_tasks.rowcount > 0),
                    error_type,
                    int(updated_tasks.rowcount > 0),
                    error_message,
                    now,
                    combination_key,
                ),
            )
            _refresh_combination(connection, combination_key, now)
    finally:
        connection.close()


def _recover_persisted_results(
    connection: sqlite3.Connection, combination_key: str, now: str
) -> None:
    """Close the small crash window between result persistence and state update."""

    result_table = connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type = 'table' AND name = 'rollout_results'"
    ).fetchone()
    if result_table is None:
        return
    result_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(rollout_results)")
    }
    if "task_execution_id" not in result_columns:
        return
    rows = connection.execute(
        """
        SELECT r.task_execution_id, r.status, r.error_type, r.error_message
        FROM rollout_results r
        JOIN benchmark_task_executions t USING (task_execution_id)
        WHERE t.combination_id = ?
          AND t.state NOT IN ('completed', 'error', 'timed_out', 'cancelled')
        """,
        (combination_key,),
    ).fetchall()
    for row in rows:
        state = "error" if row["status"] == "error" else "completed"
        outcome = None if state == "error" else row["status"]
        connection.execute(
            """
            UPDATE benchmark_task_executions
            SET state = ?, outcome = ?, error_type = ?, error_message = ?,
                heartbeat_at = ?, finished_at = ?, updated_at = ?
            WHERE task_execution_id = ?
            """,
            (
                state,
                outcome,
                row["error_type"],
                row["error_message"],
                now,
                now,
                now,
                row["task_execution_id"],
            ),
        )


def _refresh_combination(
    connection: sqlite3.Connection, combination_key: str, now: str
) -> None:
    combination = connection.execute(
        "SELECT state FROM benchmark_combinations WHERE combination_id = ?",
        (combination_key,),
    ).fetchone()
    if combination is None:
        return
    counts = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(state = 'planned') AS planned,
            SUM(state = 'queued') AS queued,
            SUM(state = 'running') AS running,
            SUM(state = 'completed') AS completed,
            SUM(state IN ('error', 'timed_out')) AS errors,
            SUM(state = 'cancelled') AS cancelled
        FROM benchmark_task_executions WHERE combination_id = ?
        """,
        (combination_key,),
    ).fetchone()
    total = int(counts["total"] or 0)
    completed = int(counts["completed"] or 0)
    errors = int(counts["errors"] or 0)
    cancelled = int(counts["cancelled"] or 0)
    running = int(counts["running"] or 0)
    queued = int(counts["queued"] or 0)
    planned = int(counts["planned"] or 0)
    current = combination["state"]

    if total and completed == total:
        state = "completed"
    elif running or (current == "running" and (queued or planned)):
        state = "running"
    elif queued:
        state = "queued"
    elif planned:
        state = "planned"
    elif errors:
        state = "error"
    elif cancelled:
        state = "cancelled"
    else:
        state = current
    finished_at = now if state in TERMINAL_COMBINATION_STATES else None
    connection.execute(
        """
        UPDATE benchmark_combinations
        SET state = ?, updated_at = ?,
            finished_at = COALESCE(finished_at, ?),
            error_type = CASE
                WHEN ? = 'error'
                THEN COALESCE(error_type, 'TaskExecutionError')
                ELSE error_type
            END,
            error_message = CASE
                WHEN ? = 'error'
                THEN COALESCE(
                    error_message,
                    'one or more task executions failed'
                )
                ELSE error_message
            END
        WHERE combination_id = ?
        """,
        (state, now, finished_at, state, state, combination_key),
    )
    _refresh_campaign_for_combination(connection, combination_key, now)


def _refresh_campaign_for_combination(
    connection: sqlite3.Connection, combination_key: str, now: str
) -> None:
    row = connection.execute(
        "SELECT campaign_id FROM benchmark_combinations WHERE combination_id = ?",
        (combination_key,),
    ).fetchone()
    if row is not None:
        _refresh_campaign(connection, row["campaign_id"], now)


def _refresh_campaign(
    connection: sqlite3.Connection, campaign_id: str, now: str
) -> None:
    counts = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(state = 'planned') AS planned,
            SUM(state = 'queued') AS queued,
            SUM(state = 'running') AS running,
            SUM(state = 'completed') AS completed,
            SUM(state = 'error') AS errors,
            SUM(state = 'cancelled') AS cancelled
        FROM benchmark_combinations WHERE campaign_id = ?
        """,
        (campaign_id,),
    ).fetchone()
    total = int(counts["total"] or 0)
    completed = int(counts["completed"] or 0)
    errors = int(counts["errors"] or 0)
    cancelled = int(counts["cancelled"] or 0)
    running = int(counts["running"] or 0)
    queued = int(counts["queued"] or 0)
    planned = int(counts["planned"] or 0)
    terminal = completed + errors + cancelled
    if total and completed == total:
        state = "completed"
    elif running or (terminal and (queued or planned)):
        state = "running"
    elif queued:
        state = "queued"
    elif planned:
        state = "planned"
    elif errors:
        state = "error"
    elif cancelled and cancelled == total:
        state = "cancelled"
    elif terminal == total:
        state = "error"
    else:
        state = "planned"
    connection.execute(
        "UPDATE benchmark_campaigns SET state = ?, updated_at = ? "
        "WHERE campaign_id = ?",
        (state, now, campaign_id),
    )


def campaigns(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    connection = _connect(path)
    try:
        _ensure_state_schema(connection)
        rows = connection.execute(
            """
            WITH combination_counts AS (
                SELECT campaign_id,
                       SUM(state = 'planned') AS planned_combinations,
                       SUM(state = 'queued') AS queued_combinations,
                       SUM(state = 'running') AS running_combinations,
                       SUM(state = 'completed') AS completed_combinations,
                       SUM(
                           state IN ('error', 'cancelled')
                       ) AS failed_combinations
                FROM benchmark_combinations
                GROUP BY campaign_id
            ), task_counts AS (
                SELECT campaign_id,
                       SUM(
                           state IN (
                               'completed', 'error', 'timed_out', 'cancelled'
                           )
                       ) AS terminal_tasks,
                       SUM(state = 'running') AS running_tasks,
                       SUM(state = 'completed') AS completed_tasks,
                       SUM(
                           state IN ('error', 'timed_out', 'cancelled')
                       ) AS errored_tasks,
                       SUM(outcome = 'passed') AS passed_tasks,
                       SUM(outcome = 'failed') AS failed_tasks
                FROM benchmark_task_executions
                GROUP BY campaign_id
            )
            SELECT c.*, k.*, t.*
            FROM benchmark_campaigns c
            LEFT JOIN combination_counts k USING (campaign_id)
            LEFT JOIN task_counts t USING (campaign_id)
            ORDER BY c.created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def campaign_combinations(path: Path, campaign_id: str) -> list[dict[str, Any]]:
    connection = _connect(path)
    try:
        rows = connection.execute(
            """
            SELECT k.*,
                   SUM(
                       t.state IN ('completed', 'error', 'timed_out', 'cancelled')
                   ) AS terminal_tasks,
                   SUM(t.state = 'running') AS running_tasks,
                   SUM(t.outcome = 'passed') AS passed_tasks,
                   SUM(t.outcome = 'failed') AS failed_tasks,
                   SUM(t.state IN ('error', 'timed_out')) AS errored_tasks
            FROM benchmark_combinations k
            LEFT JOIN benchmark_task_executions t USING (combination_id)
            WHERE k.campaign_id = ?
            GROUP BY k.combination_id
            ORDER BY k.model, k.harness, k.harness_version, k.repetition
            """,
            (campaign_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def running_tasks(path: Path, campaign_id: str) -> list[dict[str, Any]]:
    connection = _connect(path)
    try:
        rows = connection.execute(
            """
            SELECT t.*, k.model, k.harness, k.harness_version, k.repetition
            FROM benchmark_task_executions t
            JOIN benchmark_combinations k USING (combination_id)
            WHERE t.campaign_id = ? AND t.state = 'running'
            ORDER BY t.started_at, t.task_id
            """,
            (campaign_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
