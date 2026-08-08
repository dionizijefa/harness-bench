import asyncio
import datetime as dt
import json
import os
import shlex
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Literal

import dagster as dg
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset

from harness_bloat_bench.resource_monitor import (
    consume_resource_usage,
    enable_docker_resource_monitoring,
    reset_resource_usage,
)

HarnessId = Literal[
    "codex",
    "codex_agent",
    "claude_code_agent",
    "hermes_agent",
    "opencode",
    "pi",
    "omp_agent",
    "prime_agent",
]
DEFAULT_HARNESS_VERSIONS: dict[HarnessId, str] = {
    "codex": "0.137.0",
    "codex_agent": "0.137.0",
    "claude_code_agent": "2.1.226",
    "hermes_agent": "0.20.0",
    "opencode": "1.18.1",
    "pi": "0.80.7",
    "omp_agent": "16.5.2",
    "prime_agent": "0.7.1",
}
REMOTE_RESULT_PREFIX = "__HARNESS_BLOAT_RESULT__="
RUN_TYPE_TAG = "harness_bloat/run_type"
DRY_RUN_TAG = "harness_bloat/dry_run"

# Terminal-Bench tasks that require sending images to the model. Keep these out of
# every rollout matrix because the benchmark's default text-only models cannot run
# them through OpenRouter.
IMAGE_INPUT_TASK_IDS = frozenset(
    {
        "build-pov-ray",
        "chess-best-move",
        "code-from-image",
        "extract-moves-from-video",
        "gcode-to-text",
        "install-windows-3.11",
        "make-doom-for-mips",
        "path-tracing",
        "raman-fitting",
        "sam-cell-seg",
        "video-processing",
    }
)


class HarnessSpec(dg.Config):
    id: HarnessId = "codex"
    version: str | None = None


class SSHExecutionConfig(dg.Config):
    host: str
    project_dir: str
    ssh_options: list[str] = []
    copy_artifacts: bool = True


def _configured_remote() -> dict | None:
    config_path = os.environ.get("HARNESS_BLOAT_REMOTE_CONFIG")
    if not config_path or not Path(config_path).is_file():
        return None
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load remote config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise RuntimeError(f"remote config {config_path} must contain a JSON object")
    return config


def _remote_dict(remote: SSHExecutionConfig | dict | None) -> dict | None:
    if remote is None:
        return None
    return remote if isinstance(remote, dict) else remote.model_dump()


class MatrixConfig(dg.Config):
    models: list[str] = ["~deepseek/deepseek-v4-flash-latest"]
    # Empty means the default Codex harness. Dagster requires nested-config list
    # defaults to be raw dicts, so resolve the semantic default in _harness_specs.
    harnesses: list[HarnessSpec] = []
    # Compatibility with the original Codex-only launch schema. New configs should
    # pair ids and versions through ``harnesses`` instead.
    harness_versions: list[str] = []
    task_ids: list[str] = ["crack-7z-hash"]
    num_rollouts: int = 1
    dataset: str = "terminal-bench/terminal-bench-2-1"
    runtime: Literal["docker", "prime"] = "docker"
    container_cpus: float | None = 8.0
    container_memory_gb: float | None = 18.0
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_var: str = "OPENROUTER_API_KEY"
    max_tokens: int | None = None
    temperature: float | None = None
    rollout_retries: int = 0
    output_dir: str = "outputs"
    dry_run: bool = False
    remote: SSHExecutionConfig | None = _configured_remote()


def _classification_tags(dry_run: bool) -> dict[str, str]:
    return {
        RUN_TYPE_TAG: "test" if dry_run else "real",
        DRY_RUN_TAG: str(dry_run).lower(),
    }


def _harness_specs(config: MatrixConfig) -> list[tuple[HarnessId, str]]:
    if config.harness_versions:
        default_only = not config.harnesses or (
            len(config.harnesses) == 1
            and config.harnesses[0].id == "codex"
            and config.harnesses[0].version in (None, DEFAULT_HARNESS_VERSIONS["codex"])
        )
        if not default_only:
            raise dg.Failure(
                "harness_versions is the legacy Codex-only option; use "
                "harnesses: [{id: ..., version: ...}] for multiple harnesses"
            )
        return list(
            dict.fromkeys(("codex", version) for version in config.harness_versions)
        )

    entries = config.harnesses or [HarnessSpec()]
    resolved = [
        (entry.id, entry.version or DEFAULT_HARNESS_VERSIONS[entry.id])
        for entry in entries
    ]
    return list(dict.fromkeys(resolved))


def _task_ids(config: MatrixConfig) -> list[str]:
    requested = list(
        dict.fromkeys(task_id.rsplit("/", 1)[-1] for task_id in config.task_ids)
    )
    runnable_requested = [
        task_id for task_id in requested if task_id not in IMAGE_INPUT_TASK_IDS
    ]
    if config.remote is not None:
        if not requested:
            raise dg.Failure(
                "remote execution requires explicit task_ids; task discovery runs "
                "on the local Dagster host"
            )
        return runnable_requested
    if config.dry_run and requested:
        return runnable_requested
    # An explicit list containing only excluded tasks must stay empty rather than
    # being interpreted by Harbor as a request to discover the entire dataset.
    if requested and not runnable_requested:
        return []
    taskset = HarborTaskset(
        HarborConfig(
            id="harbor",
            dataset=config.dataset,
            tasks=runnable_requested or None,
        )
    )
    return [
        task_id
        for task in taskset.load()
        if (task_id := Path(task.data.task_dir).name) not in IMAGE_INPUT_TASK_IDS
    ]


@dg.op(out=dg.DynamicOut(dict))
def plan_rollouts(
    context: dg.OpExecutionContext, config: MatrixConfig
) -> Iterator[dg.DynamicOutput[dict]]:
    run_tags = _classification_tags(config.dry_run)
    context.instance.add_run_tags(context.run_id, run_tags)
    context.log.info("Classified Dagster run as %s", run_tags[RUN_TYPE_TAG])

    if config.num_rollouts < 1:
        raise dg.Failure("num_rollouts must be at least 1")
    if config.runtime == "docker":
        if config.container_cpus is not None and config.container_cpus <= 0:
            raise dg.Failure("container_cpus must be positive or null")
        if config.container_memory_gb is not None and config.container_memory_gb <= 0:
            raise dg.Failure("container_memory_gb must be positive or null")

    cases = product(
        config.models,
        _harness_specs(config),
        _task_ids(config),
        range(1, config.num_rollouts + 1),
    )
    count = 0
    for count, (model, harness_spec, task_id, rollout) in enumerate(cases, start=1):
        harness, version = harness_spec
        key = f"rollout_{count:06d}"
        yield dg.DynamicOutput(
            {
                "key": key,
                "model": model,
                "harness": harness,
                "harness_version": version,
                "task_id": task_id,
                "rollout": rollout,
                "dataset": config.dataset,
                "runtime": config.runtime,
                "container_cpus": config.container_cpus,
                "container_memory_gb": config.container_memory_gb,
                "base_url": config.base_url,
                "api_key_var": config.api_key_var,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "rollout_retries": config.rollout_retries,
                "output_dir": config.output_dir,
                "dry_run": config.dry_run,
                "remote": _remote_dict(config.remote),
            },
            mapping_key=key,
            metadata={
                "task": task_id,
                "model": model,
                "harness": harness,
                "rollout": rollout,
            },
        )
    if count == 0:
        raise dg.Failure("the rollout matrix is empty")
    context.log.info("Planned %d rollouts", count)


def _eval_config(spec: dict, output_dir: Path) -> EvalConfig:
    sampling = {
        key: spec[key] for key in ("max_tokens", "temperature") if spec[key] is not None
    }
    runtime = {"type": spec["runtime"]}
    if spec["runtime"] == "docker":
        if spec.get("container_cpus") is not None:
            runtime["cpu"] = spec["container_cpus"]
        if spec.get("container_memory_gb") is not None:
            runtime["memory"] = spec["container_memory_gb"]

    return EvalConfig.model_validate(
        {
            "model": spec["model"],
            "client": {
                "type": "eval",
                "base_url": spec["base_url"],
                "api_key_var": spec["api_key_var"],
            },
            "sampling": sampling,
            "taskset": {
                "id": "harbor",
                "dataset": spec["dataset"],
                "tasks": [spec["task_id"]],
            },
            "harness": {
                "id": spec.get("harness", "codex"),
                "version": spec["harness_version"],
                "runtime": runtime,
            },
            "retries": {
                "rollout": {"max_retries": spec["rollout_retries"]},
            },
            "num_tasks": 1,
            "num_rollouts": 1,
            "max_concurrent": 1,
            "rich": False,
            "push": False,
            "output_dir": output_dir,
        }
    )


def _execute_rollout(spec: dict, dagster_run_id: str) -> dict:
    output_dir = Path(spec["output_dir"]) / dagster_run_id / spec["key"]
    base = {
        "dagster_run_id": dagster_run_id,
        "rollout_key": spec["key"],
        "model": spec["model"],
        "harness": spec["harness"],
        "harness_version": spec["harness_version"],
        "dataset": spec["dataset"],
        "task_id": spec["task_id"],
        "rollout": spec["rollout"],
        "container_cpu_limit": (
            spec.get("container_cpus") if spec["runtime"] == "docker" else None
        ),
        "container_memory_limit_gb": (
            spec.get("container_memory_gb") if spec["runtime"] == "docker" else None
        ),
        "output_dir": str(output_dir),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if spec["dry_run"]:
        return {**base, "status": "dry_run", "passed": None, "reward": None}

    started = time.perf_counter()
    reset_resource_usage()
    if spec["runtime"] == "docker":
        enable_docker_resource_monitoring()
    traces = asyncio.run(
        run_eval(
            Environment(config := _eval_config(spec, output_dir)),
            config,
        )
    )
    trace = traces[0]
    error = trace.error.model_dump() if trace.error else None
    reward = trace.reward
    row = {
        **base,
        "trace_id": trace.id,
        "status": "error" if error else "passed" if reward >= 1 else "failed",
        "passed": not error and reward >= 1,
        "reward": reward,
        "rewards": trace.rewards,
        "error": error,
        "runtime_seconds": time.perf_counter() - started,
        **_trace_usage_fields(trace),
        **consume_resource_usage(),
    }
    return row


def _trace_usage_fields(trace) -> dict:
    usage = trace.usage
    model_calls = trace.calls
    usage_count = sum(call.usage is not None for call in model_calls)
    usage_source = (
        "provider"
        if usage is not None and usage_count == len(model_calls)
        else "provider_partial"
        if usage is not None
        else "trace_fallback"
    )
    return {
        "input_tokens": usage.input_tokens if usage else trace.num_input_tokens,
        "cached_input_tokens": usage.cached_input_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else trace.num_output_tokens,
        "total_tokens": usage.total_tokens if usage else trace.num_total_tokens,
        "reasoning_tokens": usage.reasoning_tokens if usage else None,
        "model_call_count": len(model_calls),
        "usage_source": usage_source,
        "cost_usd": usage.cost if usage else None,
    }


def _hard_failure_row(
    spec: dict,
    dagster_run_id: str,
    error: Exception,
    runtime_seconds: float,
) -> dict:
    output_dir = Path(spec["output_dir"]) / dagster_run_id / spec["key"]
    return {
        "dagster_run_id": dagster_run_id,
        "rollout_key": spec["key"],
        "model": spec["model"],
        "harness": spec["harness"],
        "harness_version": spec["harness_version"],
        "dataset": spec["dataset"],
        "task_id": spec["task_id"],
        "rollout": spec["rollout"],
        "container_cpu_limit": (
            spec.get("container_cpus") if spec.get("runtime") == "docker" else None
        ),
        "container_memory_limit_gb": (
            spec.get("container_memory_gb")
            if spec.get("runtime") == "docker"
            else None
        ),
        "output_dir": str(output_dir),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "error",
        "passed": False,
        "reward": None,
        "rewards": {},
        "error": {"type": type(error).__name__, "message": str(error)},
        "runtime_seconds": runtime_seconds,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "model_call_count": None,
        "usage_source": None,
        "cost_usd": None,
        "resource_usage_source": None,
        "cpu_seconds": None,
        "peak_memory_bytes": None,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "peak_pids": None,
        "oom_kill_count": None,
    }


def _ssh_command(remote: dict, command: str) -> list[str]:
    return ["ssh", *remote["ssh_options"], "--", remote["host"], command]


def _copy_remote_artifacts(
    context: dg.OpExecutionContext,
    remote: dict,
    remote_output_dir: str,
    local_output_dir: Path,
) -> None:
    local_output_dir.mkdir(parents=True, exist_ok=True)
    context.log.info("Copying remote artifacts into %s", local_output_dir)
    remote_command = (
        f"cd {shlex.quote(remote['project_dir'])} && "
        f"tar -C {shlex.quote(remote_output_dir)} -cf - ."
    )
    ssh_process = subprocess.Popen(
        _ssh_command(remote, remote_command),
        stdout=subprocess.PIPE,
    )
    assert ssh_process.stdout is not None
    tar_process = subprocess.run(
        ["tar", "-xf", "-", "-C", str(local_output_dir)],
        stdin=ssh_process.stdout,
        capture_output=True,
        text=False,
    )
    ssh_process.stdout.close()
    ssh_return_code = ssh_process.wait()
    if ssh_return_code != 0 or tar_process.returncode != 0:
        detail = tar_process.stderr.decode(errors="replace").strip()
        raise dg.Failure(
            "remote rollout finished, but artifact transfer failed: "
            f"ssh={ssh_return_code}, tar={tar_process.returncode}"
            + (f": {detail}" if detail else "")
        )


def _run_remote_rollout(context: dg.OpExecutionContext, spec: dict) -> dict:
    remote = spec["remote"]
    remote_command = (
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"cd {shlex.quote(remote['project_dir'])} && "
        "exec uv run python -u -m harness_bloat_bench.remote_worker"
    )
    request = {"dagster_run_id": context.run_id, "spec": spec}
    if api_key := os.environ.get(spec["api_key_var"]):
        request["api_key"] = api_key

    context.log.info("Starting rollout on SSH host %s", remote["host"])
    process = subprocess.Popen(
        _ssh_command(remote, remote_command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    row = None
    try:
        assert process.stdin is not None
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text.startswith(REMOTE_RESULT_PREFIX):
                row = json.loads(text.removeprefix(REMOTE_RESULT_PREFIX))
            elif text:
                context.log.info("[%s] %s", remote["host"], text)
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()

    if return_code != 0:
        raise dg.Failure(
            f"remote rollout on {remote['host']} exited with code {return_code}"
        )
    if row is None:
        raise dg.Failure("remote rollout exited without returning a result")

    remote_output_dir = row["output_dir"]
    local_output_dir = Path(spec["output_dir"]) / context.run_id / spec["key"]
    if remote["copy_artifacts"] and not spec["dry_run"]:
        _copy_remote_artifacts(
            context, remote, remote_output_dir, local_output_dir
        )
    row["remote_output_dir"] = remote_output_dir
    row["output_dir"] = str(local_output_dir)
    return row


@dg.op(pool="rollouts")
def run_rollout(context: dg.OpExecutionContext, spec: dict) -> dict:
    started = time.perf_counter()
    try:
        row = (
            _run_remote_rollout(context, spec)
            if spec.get("remote")
            else _execute_rollout(spec, context.run_id)
        )
    except Exception as error:
        row = _hard_failure_row(
            spec, context.run_id, error, time.perf_counter() - started
        )
        database_path = _persist_rollout_row(row)
        context.log.error(
            "Persisted failed rollout to %s before re-raising: %s",
            database_path,
            error,
        )
        raise

    database_path = _persist_rollout_row(row)
    if not spec["dry_run"]:
        metadata = {
            "task": spec["task_id"],
            "reward": row["reward"],
            "passed": row["passed"],
            "container_cpu_limit": row.get("container_cpu_limit"),
            "container_memory_limit_gb": row.get("container_memory_limit_gb"),
            "database": dg.MetadataValue.path(str(database_path)),
        }
        if row.get("resource_usage_source"):
            metadata.update(
                {
                    "cpu_seconds": row.get("cpu_seconds"),
                    "peak_memory_gib": row.get("peak_memory_bytes", 0)
                    / (1024**3),
                    "io_read_bytes": row.get("io_read_bytes"),
                    "io_write_bytes": row.get("io_write_bytes"),
                    "peak_pids": row.get("peak_pids"),
                    "oom_kill_count": row.get("oom_kill_count"),
                }
            )
        if spec.get("remote") and not spec["remote"]["copy_artifacts"]:
            metadata["remote_trace"] = dg.MetadataValue.text(
                f"{spec['remote']['host']}:{row['remote_output_dir']}/traces.jsonl"
            )
        else:
            metadata["trace"] = dg.MetadataValue.path(
                str(Path(row["output_dir"]) / "traces.jsonl")
            )
        context.add_output_metadata(metadata)
    return row


def _write_results_db(rows: list[dict], output_root: Path) -> Path:
    path = output_root / "results.sqlite"
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rollout_results (
                    dagster_run_id TEXT NOT NULL,
                    rollout_key TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    harness TEXT NOT NULL,
                    harness_version TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    rollout INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    passed INTEGER,
                    reward REAL,
                    runtime_seconds REAL,
                    input_tokens INTEGER,
                    cached_input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    reasoning_tokens INTEGER,
                    model_call_count INTEGER,
                    usage_source TEXT,
                    cost_usd REAL,
                    container_cpu_limit REAL,
                    container_memory_limit_gb REAL,
                    resource_usage_source TEXT,
                    cpu_seconds REAL,
                    peak_memory_bytes INTEGER,
                    io_read_bytes INTEGER,
                    io_write_bytes INTEGER,
                    peak_pids INTEGER,
                    oom_kill_count INTEGER,
                    trace_id TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    output_dir TEXT NOT NULL,
                    remote_output_dir TEXT,
                    row_json TEXT NOT NULL,
                    PRIMARY KEY (dagster_run_id, rollout_key)
                );
                CREATE INDEX IF NOT EXISTS rollout_results_lookup
                    ON rollout_results (model, harness, task_id, status);
                """
            )
            existing_columns = {
                info[1]
                for info in connection.execute("PRAGMA table_info(rollout_results)")
            }
            for column, column_type in {
                "reasoning_tokens": "INTEGER",
                "model_call_count": "INTEGER",
                "usage_source": "TEXT",
                "cost_usd": "REAL",
                "error_type": "TEXT",
                "error_message": "TEXT",
                "container_cpu_limit": "REAL",
                "container_memory_limit_gb": "REAL",
                "resource_usage_source": "TEXT",
                "cpu_seconds": "REAL",
                "peak_memory_bytes": "INTEGER",
                "io_read_bytes": "INTEGER",
                "io_write_bytes": "INTEGER",
                "peak_pids": "INTEGER",
                "oom_kill_count": "INTEGER",
            }.items():
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE rollout_results ADD COLUMN {column} {column_type}"
                    )

            columns = (
                "dagster_run_id",
                "rollout_key",
                "timestamp",
                "model",
                "harness",
                "harness_version",
                "dataset",
                "task_id",
                "rollout",
                "status",
                "passed",
                "reward",
                "runtime_seconds",
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "model_call_count",
                "usage_source",
                "cost_usd",
                "container_cpu_limit",
                "container_memory_limit_gb",
                "resource_usage_source",
                "cpu_seconds",
                "peak_memory_bytes",
                "io_read_bytes",
                "io_write_bytes",
                "peak_pids",
                "oom_kill_count",
                "trace_id",
                "error_type",
                "error_message",
                "output_dir",
                "remote_output_dir",
                "row_json",
            )
            placeholders = ", ".join("?" for _ in columns)
            column_names = ", ".join(columns)
            connection.executemany(
                f"INSERT OR REPLACE INTO rollout_results ({column_names}) "
                f"VALUES ({placeholders})",
                [_database_values(row) for row in rows],
            )
    finally:
        connection.close()
    return path


def _database_values(row: dict) -> tuple:
    error = row.get("error")
    error_type = (
        error.get("type") or error.get("kind") or error.get("error_type")
        if isinstance(error, dict)
        else None
    )
    error_message = (
        error.get("message") or error.get("detail")
        if isinstance(error, dict)
        else None
    )
    return (
        row["dagster_run_id"],
        row["rollout_key"],
        row["timestamp"],
        row["model"],
        row["harness"],
        row["harness_version"],
        row["dataset"],
        row["task_id"],
        row["rollout"],
        row["status"],
        None if row.get("passed") is None else int(row["passed"]),
        row.get("reward"),
        row.get("runtime_seconds"),
        row.get("input_tokens"),
        row.get("cached_input_tokens"),
        row.get("output_tokens"),
        row.get("total_tokens"),
        row.get("reasoning_tokens"),
        row.get("model_call_count"),
        row.get("usage_source"),
        row.get("cost_usd"),
        row.get("container_cpu_limit"),
        row.get("container_memory_limit_gb"),
        row.get("resource_usage_source"),
        row.get("cpu_seconds"),
        row.get("peak_memory_bytes"),
        row.get("io_read_bytes"),
        row.get("io_write_bytes"),
        row.get("peak_pids"),
        row.get("oom_kill_count"),
        row.get("trace_id"),
        error_type,
        error_message,
        row["output_dir"],
        row.get("remote_output_dir"),
        json.dumps(row, sort_keys=True),
    )


def _persist_rollout_row(row: dict) -> Path:
    output_root = Path(row["output_dir"]).parents[1]
    output_root.mkdir(parents=True, exist_ok=True)
    return _write_results_db([row], output_root)


@dg.op
def write_results(context: dg.OpExecutionContext, rows: list[dict]) -> str:
    if not rows:
        raise dg.Failure("no rollout results were produced")
    run_dir = Path(rows[0]["output_dir"]).parent
    database_path = _write_results_db(rows, run_dir.parent)
    measured = [row for row in rows if row.get("resource_usage_source")]
    metadata = {
        "database": dg.MetadataValue.path(str(database_path)),
        "dagster_run_id": rows[0]["dagster_run_id"],
        "rollouts": len(rows),
        "passed": sum(row.get("passed") is True for row in rows),
    }
    if measured:
        metadata.update(
            {
                "measured_rollouts": len(measured),
                "total_cpu_seconds": sum(row.get("cpu_seconds", 0) for row in measured),
                "max_peak_memory_gib": max(
                    row.get("peak_memory_bytes", 0) for row in measured
                )
                / (1024**3),
                "total_oom_kills": sum(
                    row.get("oom_kill_count", 0) for row in measured
                ),
            }
        )
    context.add_output_metadata(metadata)
    return str(database_path)


rollout_executor = dg.multiprocess_executor.configured(
    lambda config: {"max_concurrent": config["max_concurrent"]},
    config_schema={"max_concurrent": dg.Field(int, default_value=8)},
)


@dg.job(executor_def=rollout_executor)
def terminal_bench_rollouts() -> None:
    write_results(plan_rollouts().map(run_rollout).collect())


defs = dg.Definitions(jobs=[terminal_bench_rollouts])
