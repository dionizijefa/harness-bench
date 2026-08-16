import json
import os
from pathlib import Path
from types import SimpleNamespace

import harness_bloat_bench.remote_jobs as remote_jobs


def _request(tmp_path: Path) -> dict:
    return {
        "dagster_run_id": "run-a",
        "api_key": "secret-value",
        "spec": {
            "key": "rollout_000001",
            "api_key_var": "OPENROUTER_API_KEY",
            "output_dir": str(tmp_path / "outputs"),
            "remote": {
                "wall_timeout_seconds": 60.0,
                "timeout_grace_seconds": 30.0,
            },
        },
    }


def test_submit_is_idempotent_and_does_not_persist_api_key(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_REMOTE_JOB_DIR", str(tmp_path / "jobs"))
    launches: list[dict] = []

    def fake_popen(command, **kwargs):
        launches.append({"command": command, **kwargs})
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(remote_jobs.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        remote_jobs,
        "_process_group_alive",
        lambda process_group_id: process_group_id == 4242,
    )
    job_id = "run-a--rollout_000001"

    first = remote_jobs._start_job(job_id, _request(tmp_path))
    second = remote_jobs._start_job(job_id, _request(tmp_path))

    assert first["state"] == "queued"
    assert second["state"] == "queued"
    assert len(launches) == 1
    assert launches[0]["start_new_session"] is True
    assert launches[0]["env"]["OPENROUTER_API_KEY"] == "secret-value"
    persisted = json.loads(
        (tmp_path / "jobs" / job_id / "request.json").read_text()
    )
    assert "api_key" not in persisted
    assert "secret-value" not in json.dumps(persisted)


def test_worker_atomically_persists_exact_result_row(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_REMOTE_JOB_DIR", str(tmp_path / "jobs"))
    request = _request(tmp_path)
    request.pop("api_key")
    job_id = "run-a--rollout_000001"
    job_dir = tmp_path / "jobs" / job_id
    remote_jobs._atomic_write_json(job_dir / "request.json", request)
    remote_jobs._atomic_write_json(
        job_dir / "status.json",
        {
            "job_id": job_id,
            "state": "queued",
            "process_group_id": os.getpgrp(),
        },
    )
    expected = {
        "dagster_run_id": "run-a",
        "rollout_key": "rollout_000001",
        "output_dir": str(tmp_path / "outputs" / "run-a" / "rollout_000001"),
        "status": "passed",
        "reward": 1.0,
        "cpu_seconds": 12.5,
    }
    monkeypatch.setattr(remote_jobs, "_execute_rollout", lambda *_args: expected)

    assert remote_jobs._run_job(job_id) == 0
    assert json.loads((job_dir / "result.json").read_text()) == expected
    status = remote_jobs._status(job_id)
    assert status["state"] == "completed"
    assert status["row"] == expected


def test_status_marks_crashed_worker_as_infrastructure_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_REMOTE_JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(remote_jobs, "_process_group_alive", lambda _pid: False)
    job_id = "run-a--rollout_000001"
    job_dir = tmp_path / "jobs" / job_id
    remote_jobs._atomic_write_json(
        job_dir / "status.json",
        {"job_id": job_id, "state": "running", "process_group_id": 4242},
    )

    status = remote_jobs._status(job_id)

    assert status["state"] == "failed"
    assert status["error"]["type"] == "RemoteWorkerExited"
    assert (job_dir / "failure.json").is_file()
