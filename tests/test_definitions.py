import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import dagster as dg
from verifiers.v1.types import Usage

import harness_bloat_bench.definitions as definitions
from harness_bloat_bench.definitions import (
    BATCH_ID_TAG,
    DRY_RUN_TAG,
    HARNESS_MATRIX_TAG,
    HARNESS_TAG,
    HARNESS_VERSION_TAG,
    IMAGE_INPUT_TASK_IDS,
    MODEL_TAG,
    RUN_TYPE_TAG,
    MatrixConfig,
    SSHExecutionConfig,
    _classification_tags,
    _configured_remote,
    _eval_config,
    _execute_rollout,
    _hard_failure_row,
    _persist_rollout_row,
    _remote_dict,
    _validate_remote_limits,
    _remote_wall_timeout_seconds,
    _ssh_command,
    _task_ids,
    _trace_usage_fields,
    terminal_bench_rollouts,
)
from harness_bloat_bench.resource_monitor import _read_cgroup_usage


def test_dry_run_matrix(tmp_path: Path) -> None:
    instance = dg.DagsterInstance.ephemeral()
    result = terminal_bench_rollouts.execute_in_process(
        instance=instance,
        run_config={
            "ops": {
                "plan_rollouts": {
                    "config": {
                        "batch_id": "experiment-2026-08-15-a",
                        "models": ["model-a", "model-b"],
                        "harness_versions": ["0.137.0"],
                        "task_ids": ["terminal-bench/task-a"],
                        "num_rollouts": 2,
                        "output_dir": str(tmp_path),
                        "dry_run": True,
                    }
                }
            }
        },
    )

    assert result.success
    run = instance.get_run_by_id(result.run_id)
    assert run is not None
    assert run.tags[BATCH_ID_TAG] == "experiment-2026-08-15-a"
    assert run.tags[RUN_TYPE_TAG] == "test"
    assert run.tags[DRY_RUN_TAG] == "true"
    assert run.tags[MODEL_TAG] == "model-a,model-b"
    assert run.tags[HARNESS_TAG] == "codex_agent"
    assert run.tags[HARNESS_VERSION_TAG] == "0.137.0"
    assert run.tags[HARNESS_MATRIX_TAG] == "codex_agent@0.137.0"
    database_path = Path(result.output_for_node("write_results"))
    assert database_path == tmp_path / "results.sqlite"
    with sqlite3.connect(database_path) as connection:
        stored = connection.execute(
            """
            SELECT batch_id, model, task_id, status
            FROM rollout_results
            ORDER BY rollout_key
            """
        ).fetchall()
    assert stored == [
        ("experiment-2026-08-15-a", "model-a", "task-a", "dry_run"),
        ("experiment-2026-08-15-a", "model-a", "task-a", "dry_run"),
        ("experiment-2026-08-15-a", "model-b", "task-a", "dry_run"),
        ("experiment-2026-08-15-a", "model-b", "task-a", "dry_run"),
    ]


def test_full_runs_get_matrix_classification_tags() -> None:
    config = MatrixConfig(
        models=["model-a"],
        harnesses=[
            {"id": "codex_agent", "version": "0.140.0"},
            {"id": "codex_agent", "version": "0.147.0"},
        ],
        remote=None,
    )

    assert _classification_tags(
        config,
        [("codex_agent", "0.140.0"), ("codex_agent", "0.147.0")],
    ) == {
        RUN_TYPE_TAG: "full",
        DRY_RUN_TAG: "false",
        MODEL_TAG: "model-a",
        HARNESS_TAG: "codex_agent",
        HARNESS_VERSION_TAG: "0.140.0,0.147.0",
        HARNESS_MATRIX_TAG: "codex_agent@0.140.0,codex_agent@0.147.0",
    }


def test_real_test_rollout_can_be_tagged_without_being_a_dry_run() -> None:
    config = MatrixConfig(run_type="test", remote=None)

    tags = _classification_tags(config, [("codex_agent", "0.137.0")])

    assert tags[RUN_TYPE_TAG] == "test"
    assert tags[DRY_RUN_TAG] == "false"


def test_default_resource_profile_uses_eight_cpus_per_rollout() -> None:
    config = MatrixConfig()

    assert config.container_cpus == 8.0
    assert config.container_memory_gb == 18.0
    assert config.rollout_timeout_seconds == 5_400.0


def test_remote_config_uses_explicit_tasks_without_local_discovery() -> None:
    config = MatrixConfig(
        task_ids=["terminal-bench/task-a", "terminal-bench/code-from-image"],
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
        "wall_timeout_seconds": None,
        "timeout_grace_seconds": 1_800.0,
        "artifact_copy_timeout_seconds": 900.0,
    }
    assert validated["execution"]["multiprocess"]["max_concurrent"] == 8


def test_remote_ssh_has_keepalives_and_an_outer_deadline() -> None:
    remote = {
        "host": "terminal-bench",
        "project_dir": "/srv/harness-bloat-bench",
        "ssh_options": ["-o", "ServerAliveInterval=10"],
        "timeout_grace_seconds": 1_800.0,
    }
    command = _ssh_command(remote, "true")

    assert command[:4] == ["ssh", "-o", "ServerAliveInterval=10", "-o"]
    assert "BatchMode=yes" in command
    assert "ConnectTimeout=30" in command
    assert "ServerAliveInterval=30" in command
    assert "ServerAliveCountMax=3" in command
    assert command[-3:] == ["--", "terminal-bench", "true"]
    assert (
        _remote_wall_timeout_seconds(
            {
                "remote": remote,
                "rollout_timeout_seconds": 5_400.0,
            }
        )
        == 7_200.0
    )


def test_explicit_remote_wall_timeout_overrides_derived_deadline() -> None:
    assert (
        _remote_wall_timeout_seconds(
            {
                "remote": {"wall_timeout_seconds": 600.0},
                "rollout_timeout_seconds": 5_400.0,
            }
        )
        == 600.0
    )


def test_image_input_tasks_are_excluded_from_explicit_task_lists() -> None:
    config = MatrixConfig(
        task_ids=[
            *sorted(IMAGE_INPUT_TASK_IDS),
            "crack-7z-hash",
        ],
        dry_run=True,
        remote=None,
    )

    assert _task_ids(config) == ["crack-7z-hash"]


def test_image_input_tasks_are_excluded_from_dataset_discovery(monkeypatch) -> None:
    class FakeHarborTaskset:
        def __init__(self, _config) -> None:
            pass

        def load(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    data=SimpleNamespace(task_dir="/tasks/code-from-image")
                ),
                SimpleNamespace(data=SimpleNamespace(task_dir="/tasks/task-a")),
                SimpleNamespace(
                    data=SimpleNamespace(task_dir="/tasks/video-processing")
                ),
            ]

    monkeypatch.setattr(definitions, "HarborTaskset", FakeHarborTaskset)
    config = MatrixConfig(task_ids=[], remote=None)

    assert _task_ids(config) == ["task-a"]


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
    _validate_remote_limits(_configured_remote())


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
        calls=[
            SimpleNamespace(usage=object()),
            SimpleNamespace(usage=object()),
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
        "harness": "codex_agent",
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


def test_existing_database_is_migrated_for_batch_ids(tmp_path: Path) -> None:
    spec = {
        "key": "rollout_000001",
        "model": "model-a",
        "harness": "codex_agent",
        "harness_version": "0.130.0",
        "dataset": "terminal-bench/terminal-bench-2-1",
        "task_id": "task-a",
        "rollout": 1,
        "output_dir": str(tmp_path),
    }
    row = _hard_failure_row(spec, "run-a", RuntimeError("failed"), 1.0)
    database_path = _persist_rollout_row(row)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX rollout_results_batch")
        connection.execute("ALTER TABLE rollout_results DROP COLUMN batch_id")

    row["batch_id"] = "experiment-a"
    _persist_rollout_row(row)

    with sqlite3.connect(database_path) as connection:
        stored = connection.execute(
            "SELECT batch_id FROM rollout_results"
        ).fetchone()
    assert stored == ("experiment-a",)


def test_concurrent_results_writers_serialize_schema_migrations(tmp_path: Path) -> None:
    seed = {
        "key": "rollout_seed",
        "model": "model-a",
        "harness": "codex_agent",
        "harness_version": "0.130.0",
        "dataset": "terminal-bench/terminal-bench-2-1",
        "task_id": "task-a",
        "rollout": 1,
        "output_dir": str(tmp_path),
    }
    database_path = _persist_rollout_row(
        _hard_failure_row(seed, "seed", RuntimeError("failed"), 1.0)
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP INDEX rollout_results_batch")
        connection.execute("ALTER TABLE rollout_results DROP COLUMN batch_id")

    def persist(index: int) -> None:
        spec = {
            **seed,
            "key": f"rollout_{index:06d}",
        }
        row = _hard_failure_row(
            spec, f"run-{index}", RuntimeError("failed"), 1.0
        )
        row["batch_id"] = "concurrent-migration"
        _persist_rollout_row(row)

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(persist, range(6)))

    with sqlite3.connect(database_path) as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM rollout_results "
            "WHERE batch_id = 'concurrent-migration'"
        ).fetchone()
    assert stored == (6,)


def test_eval_config_uses_v1_components(tmp_path: Path) -> None:
    config = _eval_config(
        {
            "model": "~deepseek/deepseek-v4-flash-latest",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_var": "OPENROUTER_API_KEY",
            "dataset": "terminal-bench/terminal-bench-2-1",
            "task_id": "crack-7z-hash",
            "harness_version": "0.137.0",
            "runtime": "docker",
            "container_cpus": 10.0,
            "container_memory_gb": 18.0,
            "rollout_retries": 0,
            "max_tokens": None,
            "temperature": None,
        },
        tmp_path,
    )

    assert type(config.taskset).__name__ == "HarborConfig"
    assert config.taskset.tasks == ["crack-7z-hash"]
    assert type(config.harness).__name__ == "CodexHarnessConfig"
    assert config.harness.id == "codex_agent"
    assert config.harness.runtime.type == "docker"
    assert config.harness.runtime.cpu == 10.0
    assert config.harness.runtime.memory == 18.0
    assert config.timeout.rollout == 5_400.0


def test_cgroup_v2_resource_usage_is_aggregated(tmp_path: Path) -> None:
    (tmp_path / "cpu.stat").write_text("usage_usec 2500000\nuser_usec 2000000\n")
    (tmp_path / "memory.peak").write_text("1073741824\n")
    (tmp_path / "memory.events").write_text("oom 2\noom_kill 1\n")
    (tmp_path / "io.stat").write_text(
        "8:0 rbytes=100 wbytes=200 rios=1 wios=2\n"
        "8:16 rbytes=300 wbytes=400 rios=3 wios=4\n"
    )
    (tmp_path / "pids.peak").write_text("42\n")

    assert _read_cgroup_usage(tmp_path) == {
        "resource_usage_source": "cgroup_v2",
        "cpu_seconds": 2.5,
        "peak_memory_bytes": 1_073_741_824,
        "io_read_bytes": 400,
        "io_write_bytes": 600,
        "peak_pids": 42,
        "oom_kill_count": 1,
    }


def test_eval_config_resolves_a_local_harness_plugin(tmp_path: Path) -> None:
    config = _eval_config(
        {
            "model": "~deepseek/deepseek-v4-flash-latest",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_var": "OPENROUTER_API_KEY",
            "dataset": "terminal-bench/terminal-bench-2-1",
            "task_id": "crack-7z-hash",
            "harness": "hermes_agent",
            "harness_version": "0.20.0",
            "runtime": "docker",
            "rollout_retries": 0,
            "max_tokens": None,
            "temperature": None,
        },
        tmp_path,
    )

    assert type(config.harness).__name__ == "HermesAgentHarnessConfig"
    assert config.harness.version == "0.20.0"


def test_rollout_prefetches_harness_before_its_recorded_timer(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[object] = []

    def fake_prefetch(harness: str, version: str) -> None:
        events.append(("cache", harness, version))

    async def fake_run_eval(environment, config):
        events.append("eval")
        return [
            SimpleNamespace(
                id="trace-1",
                error=None,
                reward=1,
                rewards={},
                usage=None,
                calls=[],
                num_input_tokens=1,
                num_output_tokens=2,
                num_total_tokens=3,
            )
        ]

    ticks = iter([10.0, 13.0])
    monkeypatch.setattr(definitions, "ensure_harness_cached", fake_prefetch)
    monkeypatch.setattr(definitions, "run_eval", fake_run_eval)
    monkeypatch.setattr(definitions, "reset_resource_usage", lambda: None)
    monkeypatch.setattr(definitions, "enable_docker_resource_monitoring", lambda: None)
    monkeypatch.setattr(definitions, "consume_resource_usage", lambda: {})
    monkeypatch.setattr(definitions.time, "perf_counter", lambda: next(ticks))

    row = _execute_rollout(
        {
            "key": "rollout_000001",
            "model": "model-a",
            "harness": "omp_agent",
            "harness_version": "17.2.10",
            "dataset": "terminal-bench/terminal-bench-2-1",
            "task_id": "task-a",
            "rollout": 1,
            "runtime": "docker",
            "container_cpus": 8.0,
            "container_memory_gb": 18.0,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_var": "OPENROUTER_API_KEY",
            "max_tokens": None,
            "temperature": None,
            "rollout_timeout_seconds": 5_400.0,
            "rollout_retries": 0,
            "output_dir": str(tmp_path),
            "dry_run": False,
        },
        "run-1",
    )

    assert events == [("cache", "omp_agent", "17.2.10"), "eval"]
    assert row["runtime_seconds"] == 3.0


def test_dry_run_expands_harness_defaults(tmp_path: Path) -> None:
    result = terminal_bench_rollouts.execute_in_process(
        run_config={
            "ops": {
                "plan_rollouts": {
                    "config": {
                        "models": ["model-a"],
                        "harnesses": [
                            {"id": "codex_agent"},
                            {"id": "claude_code_agent"},
                            {"id": "hermes_agent"},
                            {"id": "opencode"},
                            {"id": "omp_agent"},
                            {"id": "prime_agent"},
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
        ("codex_agent", "0.137.0"),
        ("claude_code_agent", "2.1.226"),
        ("hermes_agent", "0.20.0"),
        ("opencode", "1.18.1"),
        ("omp_agent", "17.2.10"),
        ("prime_agent", "0.7.1"),
    }
