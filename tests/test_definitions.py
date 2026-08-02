import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import dagster as dg
import pytest
from verifiers.v1.types import Usage

from harness_bloat_bench.definitions import (
    MatrixConfig,
    SSHExecutionConfig,
    _configured_remote,
    _eval_config,
    _hard_failure_row,
    _persist_rollout_row,
    _remote_dict,
    _task_ids,
    _trace_usage_fields,
    terminal_bench_rollouts,
)


def test_dry_run_matrix(tmp_path: Path) -> None:
    result = terminal_bench_rollouts.execute_in_process(
        run_config={
            "ops": {
                "plan_rollouts": {
                    "config": {
                        "models": ["model-a", "model-b"],
                        "harness_versions": ["0.137.0"],
                        "task_ids": ["terminal-bench/task-a"],
                        "num_rollouts": 2,
                        "output_dir": str(tmp_path),
                        "dry_run": True,
                    }
                }
            }
        }
    )

    assert result.success
    database_path = Path(result.output_for_node("write_results"))
    assert database_path == tmp_path / "results.sqlite"
    with sqlite3.connect(database_path) as connection:
        stored = connection.execute(
            """
            SELECT model, task_id, status
            FROM rollout_results
            ORDER BY rollout_key
            """
        ).fetchall()
    assert stored == [
        ("model-a", "task-a", "dry_run"),
        ("model-a", "task-a", "dry_run"),
        ("model-b", "task-a", "dry_run"),
        ("model-b", "task-a", "dry_run"),
    ]


def test_remote_config_uses_explicit_tasks_without_local_discovery() -> None:
    config = MatrixConfig(
        task_ids=["terminal-bench/task-a"],
        remote=SSHExecutionConfig(
            host="terminal-bench",
            project_dir="/srv/harness-bloat-bench",
        ),
    )

    assert _task_ids(config) == ["task-a"]
    validated = dg.validate_run_config(
        terminal_bench_rollouts,
        {
            "ops": {
                "plan_rollouts": {
                    "config": {
                        "task_ids": ["task-a"],
                        "remote": {
                            "host": "terminal-bench",
                            "project_dir": "/srv/harness-bloat-bench",
                        },
                    }
                }
            }
        },
    )
    assert validated["ops"]["plan_rollouts"]["config"]["remote"] == {
        "host": "terminal-bench",
        "project_dir": "/srv/harness-bloat-bench",
        "ssh_options": [],
        "copy_artifacts": True,
    }


def test_private_remote_config_loader(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "remote.json"
    path.write_text(
        json.dumps(
            {
                "host": "terminal-bench",
                "project_dir": "/srv/harness-bloat-bench",
                "copy_artifacts": True,
            }
        )
    )
    monkeypatch.setenv("HARNESS_BLOAT_REMOTE_CONFIG", str(path))

    assert _configured_remote() == {
        "host": "terminal-bench",
        "project_dir": "/srv/harness-bloat-bench",
        "copy_artifacts": True,
    }
    assert _remote_dict(_configured_remote()) == _configured_remote()


def test_provider_usage_fields_include_cost_reasoning_and_calls() -> None:
    usage = Usage(
        prompt_tokens=1_000,
        cached_input_tokens=200,
        completion_tokens=300,
        reasoning_tokens=125,
        cost=0.0042,
    )
    trace = SimpleNamespace(
        usage=usage,
        nodes=[
            SimpleNamespace(sampled=False, usage=None),
            SimpleNamespace(sampled=True, usage=object()),
            SimpleNamespace(sampled=True, usage=object()),
        ],
        num_input_tokens=0,
        num_output_tokens=0,
        num_total_tokens=0,
    )

    assert _trace_usage_fields(trace) == {
        "input_tokens": 1_200,
        "cached_input_tokens": 200,
        "output_tokens": 300,
        "total_tokens": 1_500,
        "reasoning_tokens": 125,
        "model_call_count": 2,
        "usage_source": "provider",
        "cost_usd": 0.0042,
    }


def test_hard_failure_is_persisted_immediately(tmp_path: Path) -> None:
    spec = {
        "key": "rollout_000001",
        "model": "model-a",
        "harness": "codex",
        "harness_version": "0.130.0",
        "dataset": "terminal-bench/terminal-bench-2-1",
        "task_id": "task-a",
        "rollout": 1,
        "output_dir": str(tmp_path),
    }
    row = _hard_failure_row(spec, "run-a", RuntimeError("worker disconnected"), 3.5)

    database_path = _persist_rollout_row(row)

    with sqlite3.connect(database_path) as connection:
        stored = connection.execute(
            """
            SELECT harness_version, task_id, status, passed, error_type,
                   error_message, runtime_seconds
            FROM rollout_results
            """
        ).fetchone()
    assert stored == (
        "0.130.0",
        "task-a",
        "error",
        0,
        "RuntimeError",
        "worker disconnected",
        3.5,
    )


def test_eval_config_uses_v1_components(tmp_path: Path) -> None:
    config = _eval_config(
        {
            "model": "deepseek/deepseek-v4-flash-latest",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_var": "OPENROUTER_API_KEY",
            "dataset": "terminal-bench/terminal-bench-2-1",
            "task_id": "crack-7z-hash",
            "harness_version": "0.137.0",
            "runtime": "docker",
            "rollout_retries": 0,
            "max_tokens": None,
            "temperature": None,
        },
        tmp_path,
    )

    assert type(config.taskset).__name__ == "HarborConfig"
    assert config.taskset.tasks == ["crack-7z-hash"]
    assert type(config.harness).__name__ == "CodexHarnessConfig"
    assert config.harness.runtime.type == "docker"


@pytest.mark.parametrize(
    ("harness", "version", "config_type"),
    [
        ("opencode", "1.18.1", "OpenCodeHarnessConfig"),
        ("pi", "0.80.7", "PiHarnessConfig"),
        ("omp_agent", "16.5.2", "OmpAgentHarnessConfig"),
    ],
)
def test_eval_config_resolves_local_harness_plugins(
    tmp_path: Path, harness: str, version: str, config_type: str
) -> None:
    config = _eval_config(
        {
            "model": "deepseek/deepseek-v4-flash-latest",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_var": "OPENROUTER_API_KEY",
            "dataset": "terminal-bench/terminal-bench-2-1",
            "task_id": "crack-7z-hash",
            "harness": harness,
            "harness_version": version,
            "runtime": "docker",
            "rollout_retries": 0,
            "max_tokens": None,
            "temperature": None,
        },
        tmp_path,
    )

    assert type(config.harness).__name__ == config_type
    assert config.harness.version == version


def test_dry_run_expands_harness_defaults(tmp_path: Path) -> None:
    result = terminal_bench_rollouts.execute_in_process(
        run_config={
            "ops": {
                "plan_rollouts": {
                    "config": {
                        "models": ["model-a"],
                        "harnesses": [
                            {"id": "codex"},
                            {"id": "opencode"},
                            {"id": "pi"},
                            {"id": "omp_agent"},
                        ],
                        "task_ids": ["task-a"],
                        "output_dir": str(tmp_path),
                        "dry_run": True,
                    }
                }
            }
        }
    )

    assert result.success
    database_path = Path(result.output_for_node("write_results"))
    with sqlite3.connect(database_path) as connection:
        harnesses = set(
            connection.execute(
                "SELECT harness, harness_version FROM rollout_results"
            ).fetchall()
        )
    assert harnesses == {
        ("codex", "0.137.0"),
        ("opencode", "1.18.1"),
        ("pi", "0.80.7"),
        ("omp_agent", "16.5.2"),
    }
