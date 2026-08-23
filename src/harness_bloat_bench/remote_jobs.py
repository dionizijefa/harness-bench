"""Durable, idempotent rollout jobs for SSH workers.

The submit command returns as soon as a rollout has been detached from its SSH
session. Status and final result rows are persisted atomically under
``.remote-jobs`` so a local Dagster process can reconnect after transport loss
without restarting (and paying for) the rollout.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from harness_bloat_bench.definitions import (
    _execute_rollout,
    _remote_wall_timeout_seconds,
)


REMOTE_JOB_RESULT_PREFIX = "__HARNESS_BLOAT_REMOTE_JOB__="
JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _job_root() -> Path:
    configured = os.environ.get("HARNESS_BLOAT_REMOTE_JOB_DIR")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path.cwd() / ".remote-jobs"
    )


def _job_dir(job_id: str) -> Path:
    if JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise ValueError(f"invalid remote job id: {job_id!r}")
    return _job_root() / job_id


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _process_group_alive(process_group_id: object) -> bool:
    if not isinstance(process_group_id, int) or process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _failure(
    job_id: str,
    error_type: str,
    message: str,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "job_id": job_id,
        "state": "failed",
        "updated_at": _utc_now(),
        "error": {"type": error_type, "message": message},
    }
    if detail:
        value["error"]["detail"] = detail
    return value


def _status(job_id: str, *, reap: bool = True) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    if result := _read_json(job_dir / "result.json"):
        return {
            "job_id": job_id,
            "state": "completed",
            "updated_at": result.get("timestamp", _utc_now()),
            "row": result,
        }
    if failure := _read_json(job_dir / "failure.json"):
        return failure

    status_path = job_dir / "status.json"
    status = _read_json(status_path)
    if status is None:
        return {"job_id": job_id, "state": "missing", "updated_at": _utc_now()}
    if status.get("state") in TERMINAL_STATES:
        return status
    if not reap or _process_group_alive(status.get("process_group_id")):
        return status

    failure = _failure(
        job_id,
        "RemoteWorkerExited",
        "detached remote worker exited without writing a final result",
    )
    _atomic_write_json(job_dir / "failure.json", failure)
    _atomic_write_json(status_path, failure)
    return failure


def _request_without_secret(request: dict[str, Any]) -> dict[str, Any]:
    persisted = dict(request)
    persisted.pop("api_key", None)
    return persisted


def _start_job(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    lock_path = job_dir / "submit.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = _status(job_id, reap=False)
        if current["state"] in TERMINAL_STATES:
            return current
        if current["state"] in {"queued", "running"} and _process_group_alive(
            current.get("process_group_id")
        ):
            return current

        persisted_request = _request_without_secret(request)
        request_path = job_dir / "request.json"
        if existing := _read_json(request_path):
            if existing != persisted_request:
                failure = _failure(
                    job_id,
                    "RemoteJobConflict",
                    "job id already exists with a different rollout request",
                )
                _atomic_write_json(job_dir / "failure.json", failure)
                _atomic_write_json(job_dir / "status.json", failure)
                return failure
        else:
            _atomic_write_json(request_path, persisted_request)
            request_path.chmod(0o600)

        spec = request.get("spec")
        if not isinstance(spec, dict):
            failure = _failure(
                job_id, "InvalidRemoteRequest", "request is missing its rollout spec"
            )
            _atomic_write_json(job_dir / "failure.json", failure)
            _atomic_write_json(job_dir / "status.json", failure)
            return failure

        command = [sys.executable, "-m", "harness_bloat_bench.remote_jobs", "run", job_id]
        wall_timeout = _remote_wall_timeout_seconds(spec)
        if wall_timeout is not None:
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                f"{wall_timeout:g}s",
                *command,
            ]

        environment = os.environ.copy()
        if api_key := request.get("api_key"):
            environment[spec["api_key_var"]] = api_key
        log_path = job_dir / "worker.log"
        try:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=Path.cwd(),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except Exception as error:
            failure = _failure(job_id, type(error).__name__, str(error))
            _atomic_write_json(job_dir / "failure.json", failure)
            _atomic_write_json(job_dir / "status.json", failure)
            return failure

        status = {
            "job_id": job_id,
            "state": "queued",
            "submitted_at": _utc_now(),
            "updated_at": _utc_now(),
            "launcher_pid": process.pid,
            "process_group_id": process.pid,
            "log_path": str(log_path),
        }
        _atomic_write_json(job_dir / "status.json", status)
        return status


def _run_job(job_id: str) -> int:
    job_dir = _job_dir(job_id)
    worker_lock_path = job_dir / "worker.lock"
    with worker_lock_path.open("a+b") as worker_lock:
        try:
            fcntl.flock(worker_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        request = _read_json(job_dir / "request.json")
        if request is None:
            failure = _failure(
                job_id, "InvalidRemoteRequest", "durable request file is missing"
            )
            _atomic_write_json(job_dir / "failure.json", failure)
            _atomic_write_json(job_dir / "status.json", failure)
            return 1

        previous = _read_json(job_dir / "status.json") or {}
        status = {
            **previous,
            "job_id": job_id,
            "state": "running",
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "worker_pid": os.getpid(),
            "process_group_id": os.getpgrp(),
        }
        _atomic_write_json(job_dir / "status.json", status)

        try:
            spec = request["spec"]
            spec["remote"] = None
            row = _execute_rollout(spec, request["dagster_run_id"])
            _atomic_write_json(job_dir / "result.json", row)
            completed = {
                **status,
                "state": "completed",
                "completed_at": _utc_now(),
                "updated_at": _utc_now(),
            }
            _atomic_write_json(job_dir / "status.json", completed)
            return 0
        except BaseException as error:
            failure = _failure(
                job_id,
                type(error).__name__,
                str(error),
                detail="".join(traceback.format_exception(error)),
            )
            failure.update(
                {
                    "started_at": status.get("started_at"),
                    "log_path": status.get("log_path"),
                }
            )
            _atomic_write_json(job_dir / "failure.json", failure)
            _atomic_write_json(job_dir / "status.json", failure)
            return 1


def _cancel_job(job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    status = _status(job_id)
    if status["state"] in TERMINAL_STATES or status["state"] == "missing":
        return status
    process_group_id = status.get("process_group_id")
    if _process_group_alive(process_group_id):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGTERM)
    cancelled = {
        **status,
        "state": "cancelled",
        "cancelled_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _atomic_write_json(job_dir / "status.json", cancelled)
    return cancelled


def _emit(value: dict[str, Any]) -> None:
    print(f"{REMOTE_JOB_RESULT_PREFIX}{json.dumps(value, sort_keys=True)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage durable remote rollout jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("submit", "status", "cancel", "run"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("job_id")
    args = parser.parse_args()

    if args.command == "submit":
        request = json.load(sys.stdin)
        _emit(_start_job(args.job_id, request))
    elif args.command == "status":
        _emit(_status(args.job_id))
    elif args.command == "cancel":
        _emit(_cancel_job(args.job_id))
    else:
        raise SystemExit(_run_job(args.job_id))


if __name__ == "__main__":
    main()
