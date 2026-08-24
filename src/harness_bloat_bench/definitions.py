import asyncio
import datetime as dt
import json
import os
import re
import shlex
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Literal

import dagster as dg
from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset

from harness_bloat_bench.harness_prefetch import ensure_harness_cached
from harness_bloat_bench.resource_monitor import (
    consume_resource_usage,
    enable_docker_resource_monitoring,
    reset_resource_usage,
)
from harness_bloat_bench.state import (
    CampaignConflict,
    active_combinations,
    bind_combination_run,
    combination_id,
    create_campaign,
    launchable_combinations,
    mark_combination_queued,
    mark_task_running,
    mark_task_terminal,
    new_campaign_id,
    reconcile_combination,
    state_database_path,
    task_execution_id,
    touch_task,
    validate_campaign_id,
    validate_combination_config,
)

HarnessId = Literal[
    "codex_agent",
    "claude_code_agent",
    "deepseek_harness",
    "hermes_agent",
    "opencode",
    "omp_agent",
    "pi_agent",
    "pi_rlm_runtime",
    "prime_agent",
]


class ReasoningEffort(str, Enum):
    none = "none"
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


DEFAULT_HARNESS_VERSIONS: dict[HarnessId, str] = {
    "codex_agent": "0.137.0",
    "claude_code_agent": "2.1.226",
    "deepseek_harness": "0.1.1-rc.2",
    "hermes_agent": "0.20.0",
    "opencode": "1.18.1",
    "omp_agent": "17.2.10",
    "pi_agent": "0.84.2",
    "pi_rlm_runtime": "0.1.1",
    "prime_agent": "0.7.1",
}
REMOTE_RESULT_PREFIX = "__HARNESS_BLOAT_RESULT__="
REMOTE_JOB_RESULT_PREFIX = "__HARNESS_BLOAT_REMOTE_JOB__="
RUN_TYPE_TAG = "harness_bloat/run_type"
DRY_RUN_TAG = "harness_bloat/dry_run"
MODEL_TAG = "harness_bloat/model"
HARNESS_TAG = "harness_bloat/harness"
HARNESS_VERSION_TAG = "harness_bloat/harness_version"
HARNESS_MATRIX_TAG = "harness_bloat/harness_matrix"
BATCH_ID_TAG = "harness_bloat/batch_id"
PROGRESS_TAG = "harness_bloat/progress"
REASONING_EFFORT_TAG = "harness_bloat/reasoning_effort"
CAMPAIGN_ID_TAG = "harness_bloat/campaign_id"
COMBINATION_ID_TAG = "harness_bloat/combination_id"
REPETITION_TAG = "harness_bloat/repetition"
TASK_SELECTION_TAG = "harness_bloat/task_selection"
TASK_COUNT_TAG = "harness_bloat/task_count"

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
    id: HarnessId = "codex_agent"
    version: str | None = None


class SSHExecutionConfig(dg.Config):
    host: str
    project_dir: str
    ssh_options: list[str] = []
    copy_artifacts: bool = False
    wall_timeout_seconds: float | None = None
    timeout_grace_seconds: float = 1_800.0
    artifact_copy_timeout_seconds: float = 900.0
    poll_interval_seconds: float = 15.0
    reconnect_delay_seconds: float = 5.0
    ssh_command_timeout_seconds: float = 60.0
    submit_timeout_seconds: float = 300.0


def _configured_remote() -> dict | None:
    config_path = os.environ.get("HARNESS_BLOAT_REMOTE_CONFIG")
    if not config_path or not Path(config_path).is_file():
        return None
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot load remote config {config_path}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise RuntimeError(f"remote config {config_path} must contain a JSON object")
    return config


def _remote_dict(remote: SSHExecutionConfig | dict | None) -> dict | None:
    if remote is None:
        return None
    return remote if isinstance(remote, dict) else remote.model_dump()


def _validate_remote_limits(remote: SSHExecutionConfig | dict | None) -> None:
    resolved = _remote_dict(remote)
    if resolved is None:
        return
    if (
        resolved.get("wall_timeout_seconds") is not None
        and resolved["wall_timeout_seconds"] <= 0
    ):
        raise dg.Failure("remote.wall_timeout_seconds must be positive or null")
    if resolved.get("timeout_grace_seconds", 1_800.0) <= 0:
        raise dg.Failure("remote.timeout_grace_seconds must be positive")
    if resolved.get("artifact_copy_timeout_seconds", 900.0) <= 0:
        raise dg.Failure("remote.artifact_copy_timeout_seconds must be positive")
    for name, default in (
        ("poll_interval_seconds", 15.0),
        ("reconnect_delay_seconds", 5.0),
        ("ssh_command_timeout_seconds", 60.0),
        ("submit_timeout_seconds", 300.0),
    ):
        if resolved.get(name, default) <= 0:
            raise dg.Failure(f"remote.{name} must be positive")


class ExecutionConfig(dg.Config):
    reasoning_effort: ReasoningEffort | None = None
    dataset: str = "terminal-bench/terminal-bench-2-1"
    runtime: Literal["docker", "prime"] = "docker"
    container_cpus: float | None = 8.0
    container_memory_gb: float | None = 18.0
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_var: str = "OPENROUTER_API_KEY"
    max_tokens: int | None = None
    temperature: float | None = None
    rollout_timeout_seconds: float | None = 5_400.0
    rollout_retries: int = 0
    output_dir: str = "outputs"
    run_type: Literal["test", "full"] = "full"
    dry_run: bool = False
    remote: SSHExecutionConfig | None = _configured_remote()


class MatrixConfig(ExecutionConfig):
    """Legacy all-in-one Dagster run configuration."""

    batch_id: str | None = None
    models: list[str] = ["deepseek/deepseek-v4-flash-0731"]
    # Empty means the default Codex Agent harness. Dagster requires nested-config list
    # defaults to be raw dicts, so resolve the semantic default in _harness_specs.
    harnesses: list[HarnessSpec] = []
    # Compatibility with the original Codex Agent-only launch schema. New configs should
    # pair ids and versions through ``harnesses`` instead.
    harness_versions: list[str] = []
    task_ids: list[str] = ["crack-7z-hash"]
    num_rollouts: int = 1


class TaskSelectionConfig(dg.Config):
    mode: Literal["full", "selected"] = "selected"
    ids: list[str] = ["crack-7z-hash"]


class CampaignConfig(ExecutionConfig):
    """Configuration for a persisted campaign expanded into combination runs."""

    campaign_id: str | None = None
    batch_id: str | None = None
    models: list[str] = ["deepseek/deepseek-v4-flash-0731"]
    harnesses: list[HarnessSpec] = []
    repetitions: int = 1
    task_selection: TaskSelectionConfig = TaskSelectionConfig()
    combination_max_concurrent: int = 8


class CombinationRunConfig(ExecutionConfig):
    campaign_id: str
    combination_id: str
    batch_id: str | None = None
    model: str
    harness: HarnessId
    harness_version: str
    repetition: int
    task_selection_mode: Literal["full", "selected"]
    task_ids: list[str]
    state_db_path: str


def _classification_tags(
    config: MatrixConfig, harness_specs: list[tuple[HarnessId, str]]
) -> dict[str, str]:
    models = list(dict.fromkeys(config.models))
    harnesses = list(dict.fromkeys(harness for harness, _ in harness_specs))
    versions = list(dict.fromkeys(version for _, version in harness_specs))
    tags = {
        RUN_TYPE_TAG: "test" if config.dry_run else config.run_type,
        DRY_RUN_TAG: str(config.dry_run).lower(),
        MODEL_TAG: ",".join(models),
        HARNESS_TAG: ",".join(harnesses),
        HARNESS_VERSION_TAG: ",".join(versions),
        HARNESS_MATRIX_TAG: ",".join(
            f"{harness}@{version}" for harness, version in harness_specs
        ),
    }
    if config.batch_id is not None:
        tags[BATCH_ID_TAG] = config.batch_id
    if config.reasoning_effort is not None:
        tags[REASONING_EFFORT_TAG] = config.reasoning_effort.value
    return tags


def _harness_specs(config: MatrixConfig) -> list[tuple[HarnessId, str]]:
    if config.harness_versions:
        default_only = not config.harnesses or (
            len(config.harnesses) == 1
            and config.harnesses[0].id == "codex_agent"
            and config.harnesses[0].version
            in (None, DEFAULT_HARNESS_VERSIONS["codex_agent"])
        )
        if not default_only:
            raise dg.Failure(
                "harness_versions is the legacy Codex Agent-only option; use "
                "harnesses: [{id: ..., version: ...}] for multiple harnesses"
            )
        return list(
            dict.fromkeys(
                ("codex_agent", version) for version in config.harness_versions
            )
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


def _campaign_harness_specs(config: CampaignConfig) -> list[tuple[HarnessId, str]]:
    entries = config.harnesses or [HarnessSpec()]
    return list(
        dict.fromkeys(
            (
                entry.id,
                entry.version or DEFAULT_HARNESS_VERSIONS[entry.id],
            )
            for entry in entries
        )
    )


def _campaign_task_ids(config: CampaignConfig) -> list[str]:
    selection = config.task_selection
    if selection.mode == "selected":
        requested = list(
            dict.fromkeys(task_id.rsplit("/", 1)[-1] for task_id in selection.ids)
        )
        if not requested:
            raise dg.Failure("selected task mode requires at least one task ID")
        runnable = [
            task_id for task_id in requested if task_id not in IMAGE_INPUT_TASK_IDS
        ]
        if not runnable:
            raise dg.Failure("the selected task list contains no runnable tasks")
        return runnable

    taskset = HarborTaskset(
        HarborConfig(id="harbor", dataset=config.dataset, tasks=None)
    )
    task_ids = [
        task_id
        for task in taskset.load()
        if (task_id := Path(task.data.task_dir).name) not in IMAGE_INPUT_TASK_IDS
    ]
    if not task_ids:
        raise dg.Failure("full task discovery returned no runnable tasks")
    return list(dict.fromkeys(task_ids))


def _validate_execution_config(config: ExecutionConfig) -> None:
    if (
        config.rollout_timeout_seconds is not None
        and config.rollout_timeout_seconds <= 0
    ):
        raise dg.Failure("rollout_timeout_seconds must be positive or null")
    _validate_remote_limits(config.remote)
    if config.runtime == "docker":
        if config.container_cpus is not None and config.container_cpus <= 0:
            raise dg.Failure("container_cpus must be positive or null")
        if config.container_memory_gb is not None and config.container_memory_gb <= 0:
            raise dg.Failure("container_memory_gb must be positive or null")


def _combination_run_config(
    config: CampaignConfig,
    *,
    campaign_key: str,
    combination_key: str,
    model: str,
    harness: HarnessId,
    harness_version: str,
    repetition: int,
    task_ids: list[str],
    database_path: Path,
) -> dict:
    common = config.model_dump(
        mode="json",
        exclude={
            "campaign_id",
            "batch_id",
            "models",
            "harnesses",
            "repetitions",
            "task_selection",
            "combination_max_concurrent",
        },
    )
    child = CombinationRunConfig(
        **common,
        campaign_id=campaign_key,
        combination_id=combination_key,
        batch_id=config.batch_id,
        model=model,
        harness=harness,
        harness_version=harness_version,
        repetition=repetition,
        task_selection_mode=config.task_selection.mode,
        task_ids=task_ids,
        state_db_path=str(database_path),
    )
    return {
        "ops": {
            "plan_task_executions": {
                "config": child.model_dump(mode="json"),
            }
        },
        "execution": {
            "config": {"max_concurrent": config.combination_max_concurrent}
        },
    }


def _task_mapping_key(task_id: str, execution_key: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_]", "_", task_id).strip("_") or "task"
    return f"{readable[:48]}__{execution_key.rsplit('_', 1)[-1][:10]}"


@dg.op
def plan_campaign(context: dg.OpExecutionContext, config: CampaignConfig) -> str:
    """Snapshot a campaign and persist every logical task before launching it."""

    _validate_execution_config(config)
    if config.repetitions < 1:
        raise dg.Failure("repetitions must be at least 1")
    if config.combination_max_concurrent < 1:
        raise dg.Failure("combination_max_concurrent must be at least 1")
    if config.rollout_retries != 0:
        raise dg.Failure(
            "campaign execution retries are not implemented; rollout_retries must be 0"
        )
    if config.batch_id is not None and not config.batch_id.strip():
        raise dg.Failure("batch_id must contain at least one non-whitespace character")

    models = list(dict.fromkeys(config.models))
    if not models:
        raise dg.Failure("models must contain at least one model")
    harness_specs = _campaign_harness_specs(config)
    task_ids = _campaign_task_ids(config)
    campaign_key = (
        validate_campaign_id(config.campaign_id)
        if config.campaign_id is not None
        else new_campaign_id()
    )
    database_path = state_database_path()

    planned: list[dict] = []
    for model, harness_spec, repetition in product(
        models,
        harness_specs,
        range(1, config.repetitions + 1),
    ):
        harness, harness_version = harness_spec
        combination_key = combination_id(
            campaign_key,
            model,
            harness,
            harness_version,
            repetition,
        )
        planned.append(
            {
                "combination_id": combination_key,
                "model": model,
                "harness": harness,
                "harness_version": harness_version,
                "repetition": repetition,
                "run_config": _combination_run_config(
                    config,
                    campaign_key=campaign_key,
                    combination_key=combination_key,
                    model=model,
                    harness=harness,
                    harness_version=harness_version,
                    repetition=repetition,
                    task_ids=task_ids,
                    database_path=database_path,
                ),
            }
        )

    manifest = {
        "campaign_id": campaign_key,
        "batch_id": config.batch_id,
        "dataset": config.dataset,
        "models": models,
        "harnesses": [
            {"id": harness, "version": version}
            for harness, version in harness_specs
        ],
        "repetitions": config.repetitions,
        "task_selection_mode": config.task_selection.mode,
        "task_ids": task_ids,
        "execution": config.model_dump(
            mode="json",
            exclude={
                "campaign_id",
                "batch_id",
                "models",
                "harnesses",
                "repetitions",
                "task_selection",
            },
        ),
    }
    try:
        created = create_campaign(
            database_path,
            manifest=manifest,
            planner_run_id=context.run_id,
            combinations=planned,
        )
    except (CampaignConflict, ValueError) as error:
        raise dg.Failure(str(error)) from error

    total_tasks = len(planned) * len(task_ids)
    tags = {
        CAMPAIGN_ID_TAG: campaign_key,
        RUN_TYPE_TAG: "test" if config.dry_run else config.run_type,
        DRY_RUN_TAG: str(config.dry_run).lower(),
        MODEL_TAG: ",".join(models),
        HARNESS_TAG: ",".join(dict.fromkeys(h for h, _ in harness_specs)),
        HARNESS_VERSION_TAG: ",".join(
            dict.fromkeys(version for _, version in harness_specs)
        ),
        HARNESS_MATRIX_TAG: ",".join(
            f"{harness}@{version}" for harness, version in harness_specs
        ),
        TASK_SELECTION_TAG: config.task_selection.mode,
        TASK_COUNT_TAG: str(len(task_ids)),
    }
    if config.batch_id is not None:
        tags[BATCH_ID_TAG] = config.batch_id
    if config.reasoning_effort is not None:
        tags[REASONING_EFFORT_TAG] = config.reasoning_effort.value
    context.instance.add_run_tags(context.run_id, tags)
    context.add_output_metadata(
        {
            "campaign_id": campaign_key,
            "state_database": dg.MetadataValue.path(str(database_path)),
            "created": created,
            "combinations": len(planned),
            "tasks_per_combination": len(task_ids),
            "task_executions": total_tasks,
            "task_selection": config.task_selection.mode,
        }
    )
    context.log.info(
        "%s campaign %s with %d combinations and %d task executions",
        "Created" if created else "Reused",
        campaign_key,
        len(planned),
        total_tasks,
    )
    return campaign_key


@dg.op(out=dg.DynamicOut(dict))
def plan_task_executions(
    context: dg.OpExecutionContext, config: CombinationRunConfig
) -> Iterator[dg.DynamicOutput[dict]]:
    _validate_execution_config(config)
    expected_combination_id = combination_id(
        config.campaign_id,
        config.model,
        config.harness,
        config.harness_version,
        config.repetition,
    )
    if config.combination_id != expected_combination_id:
        raise dg.Failure(
            "combination_id does not match the deterministic combination identity"
        )
    if not config.task_ids:
        raise dg.Failure("combination task list is empty")

    database_path = Path(config.state_db_path)
    validate_combination_config(
        database_path,
        config.combination_id,
        config.model_dump(mode="json"),
    )
    bind_combination_run(
        database_path,
        config.combination_id,
        context.run_id,
        started=True,
    )
    tags = {
        CAMPAIGN_ID_TAG: config.campaign_id,
        COMBINATION_ID_TAG: config.combination_id,
        RUN_TYPE_TAG: "test" if config.dry_run else config.run_type,
        DRY_RUN_TAG: str(config.dry_run).lower(),
        MODEL_TAG: config.model,
        HARNESS_TAG: config.harness,
        HARNESS_VERSION_TAG: config.harness_version,
        HARNESS_MATRIX_TAG: f"{config.harness}@{config.harness_version}",
        REPETITION_TAG: str(config.repetition),
        TASK_SELECTION_TAG: config.task_selection_mode,
        TASK_COUNT_TAG: str(len(config.task_ids)),
        PROGRESS_TAG: _progress_text(0, len(config.task_ids)),
    }
    if config.batch_id is not None:
        tags[BATCH_ID_TAG] = config.batch_id
    if config.reasoning_effort is not None:
        tags[REASONING_EFFORT_TAG] = config.reasoning_effort.value
    context.instance.add_run_tags(context.run_id, tags)

    common = config.model_dump(mode="json")
    for task_id in config.task_ids:
        execution_key = task_execution_id(config.combination_id, task_id)
        spec = {
            **common,
            "key": execution_key,
            "task_execution_id": execution_key,
            "task_id": task_id,
            "rollout": config.repetition,
            "total_rollouts": len(config.task_ids),
        }
        yield dg.DynamicOutput(
            spec,
            mapping_key=_task_mapping_key(task_id, execution_key),
            metadata={
                "campaign_id": config.campaign_id,
                "combination_id": config.combination_id,
                "task_execution_id": execution_key,
                "task": task_id,
                "model": config.model,
                "harness": config.harness,
                "harness_version": config.harness_version,
                "repetition": config.repetition,
            },
        )


@dg.op(out=dg.DynamicOut(dict))
def plan_rollouts(
    context: dg.OpExecutionContext, config: MatrixConfig
) -> Iterator[dg.DynamicOutput[dict]]:
    harness_specs = _harness_specs(config)
    run_tags = _classification_tags(config, harness_specs)
    context.instance.add_run_tags(context.run_id, run_tags)
    context.log.info(
        "Tagged Dagster run as %s (%s; %s)",
        run_tags[RUN_TYPE_TAG],
        run_tags[MODEL_TAG],
        run_tags[HARNESS_MATRIX_TAG],
    )

    if config.num_rollouts < 1:
        raise dg.Failure("num_rollouts must be at least 1")
    if config.batch_id is not None and not config.batch_id.strip():
        raise dg.Failure("batch_id must contain at least one non-whitespace character")
    if (
        config.rollout_timeout_seconds is not None
        and config.rollout_timeout_seconds <= 0
    ):
        raise dg.Failure("rollout_timeout_seconds must be positive or null")
    _validate_remote_limits(config.remote)
    if config.runtime == "docker":
        if config.container_cpus is not None and config.container_cpus <= 0:
            raise dg.Failure("container_cpus must be positive or null")
        if config.container_memory_gb is not None and config.container_memory_gb <= 0:
            raise dg.Failure("container_memory_gb must be positive or null")

    task_ids = _task_ids(config)
    total_rollouts = (
        len(config.models)
        * len(harness_specs)
        * len(task_ids)
        * config.num_rollouts
    )
    if total_rollouts == 0:
        raise dg.Failure("the rollout matrix is empty")
    context.instance.add_run_tags(
        context.run_id, {PROGRESS_TAG: _progress_text(0, total_rollouts)}
    )

    cases = product(
        config.models,
        harness_specs,
        task_ids,
        range(1, config.num_rollouts + 1),
    )
    count = 0
    for count, (model, harness_spec, task_id, rollout) in enumerate(cases, start=1):
        harness, version = harness_spec
        key = f"rollout_{count:06d}"
        yield dg.DynamicOutput(
            {
                "key": key,
                "batch_id": config.batch_id,
                "model": model,
                "reasoning_effort": (
                    config.reasoning_effort.value
                    if config.reasoning_effort is not None
                    else None
                ),
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
                "rollout_timeout_seconds": config.rollout_timeout_seconds,
                "rollout_retries": config.rollout_retries,
                "total_rollouts": total_rollouts,
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
    context.log.info("Planned %d rollouts", count)


def _eval_config(spec: dict, output_dir: Path) -> EvalConfig:
    harness_id = spec.get("harness", "codex_agent")
    sampling = {
        key: spec[key]
        for key in ("max_tokens", "temperature", "reasoning_effort")
        if spec.get(key) is not None
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
                "id": harness_id,
                "version": spec["harness_version"],
                "runtime": runtime,
            },
            "retries": {
                "rollout": {"max_retries": spec["rollout_retries"]},
            },
            "timeout": {
                "rollout": spec.get("rollout_timeout_seconds", 5_400.0),
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
        "campaign_id": spec.get("campaign_id"),
        "combination_id": spec.get("combination_id"),
        "task_execution_id": spec.get("task_execution_id"),
        "batch_id": spec.get("batch_id"),
        "rollout_key": spec["key"],
        "model": spec["model"],
        "reasoning_effort": spec.get("reasoning_effort"),
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
        "state_db_path": spec.get("state_db_path"),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if spec["dry_run"]:
        return {**base, "status": "dry_run", "passed": None, "reward": None}

    # Cache acquisition is deliberately outside runtime_seconds. On a cold host,
    # concurrent rollout processes serialize on the per-release cache lock; none
    # of their benchmark timings include release downloads or dependency builds.
    ensure_harness_cached(spec["harness"], spec["harness_version"])
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
        "campaign_id": spec.get("campaign_id"),
        "combination_id": spec.get("combination_id"),
        "task_execution_id": spec.get("task_execution_id"),
        "batch_id": spec.get("batch_id"),
        "rollout_key": spec["key"],
        "model": spec["model"],
        "reasoning_effort": spec.get("reasoning_effort"),
        "harness": spec["harness"],
        "harness_version": spec["harness_version"],
        "dataset": spec["dataset"],
        "task_id": spec["task_id"],
        "rollout": spec["rollout"],
        "container_cpu_limit": (
            spec.get("container_cpus") if spec.get("runtime") == "docker" else None
        ),
        "container_memory_limit_gb": (
            spec.get("container_memory_gb") if spec.get("runtime") == "docker" else None
        ),
        "output_dir": str(output_dir),
        "state_db_path": spec.get("state_db_path"),
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
    # User-provided command-line options come first because OpenSSH uses the
    # first value it receives for an option. The defaults therefore remain
    # overridable while protecting ordinary runs from half-open TCP sessions.
    return [
        "ssh",
        *remote.get("ssh_options", []),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "--",
        remote["host"],
        command,
    ]


def _remote_wall_timeout_seconds(spec: dict) -> float | None:
    remote = spec["remote"]
    if remote.get("wall_timeout_seconds") is not None:
        return float(remote["wall_timeout_seconds"])
    rollout_timeout = spec.get("rollout_timeout_seconds")
    if rollout_timeout is None:
        return None
    return float(rollout_timeout) + float(remote.get("timeout_grace_seconds", 1_800.0))


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _copy_remote_artifacts(
    context: dg.OpExecutionContext,
    remote: dict,
    remote_output_dir: str,
    local_output_dir: Path,
) -> None:
    local_output_dir.mkdir(parents=True, exist_ok=True)
    context.log.info("Copying remote artifacts into %s", local_output_dir)
    copy_timeout = float(remote.get("artifact_copy_timeout_seconds", 900.0))
    remote_command = (
        f"cd {shlex.quote(remote['project_dir'])} && "
        "exec timeout --signal=TERM --kill-after=30s "
        f"{copy_timeout}s tar -C {shlex.quote(remote_output_dir)} -cf - ."
    )
    ssh_process = subprocess.Popen(
        _ssh_command(remote, remote_command),
        stdout=subprocess.PIPE,
    )
    assert ssh_process.stdout is not None
    try:
        tar_process = subprocess.run(
            ["tar", "-xf", "-", "-C", str(local_output_dir)],
            stdin=ssh_process.stdout,
            capture_output=True,
            text=False,
            timeout=copy_timeout + 120,
            check=False,
        )
        ssh_process.stdout.close()
        ssh_return_code = ssh_process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        _terminate_process(ssh_process)
        raise dg.Failure(
            f"remote artifact transfer exceeded {copy_timeout:g} seconds"
        ) from error
    if ssh_return_code != 0 or tar_process.returncode != 0:
        detail = tar_process.stderr.decode(errors="replace").strip()
        raise dg.Failure(
            "remote rollout finished, but artifact transfer failed: "
            f"ssh={ssh_return_code}, tar={tar_process.returncode}"
            + (f": {detail}" if detail else "")
        )


class RemoteJobTransportError(RuntimeError):
    """A reconnectable failure while invoking the SSH job-control protocol."""


def _remote_job_id(dagster_run_id: str, rollout_key: str) -> str:
    return f"{dagster_run_id}--{rollout_key}"


def _remote_job_command(remote: dict, action: str, job_id: str) -> str:
    return (
        'export PATH="$HOME/.local/bin:$PATH"; '
        f"cd {shlex.quote(remote['project_dir'])} && "
        "exec uv run python -u -m harness_bloat_bench.remote_jobs "
        f"{shlex.quote(action)} {shlex.quote(job_id)}"
    )


def _invoke_remote_job(
    remote: dict,
    action: str,
    job_id: str,
    request: dict | None = None,
) -> dict:
    command = _remote_job_command(remote, action, job_id)
    timeout_seconds = float(remote.get("ssh_command_timeout_seconds", 60.0))
    try:
        completed = subprocess.run(
            _ssh_command(remote, command),
            input=json.dumps(request) if request is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RemoteJobTransportError(
            f"remote {action} command exceeded {timeout_seconds:g} seconds"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RemoteJobTransportError(
            f"remote {action} command exited with code {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith(REMOTE_JOB_RESULT_PREFIX):
            try:
                payload = json.loads(line.removeprefix(REMOTE_JOB_RESULT_PREFIX))
            except json.JSONDecodeError as error:
                raise RemoteJobTransportError(
                    f"remote {action} returned invalid job JSON"
                ) from error
    if not isinstance(payload, dict) or payload.get("job_id") != job_id:
        raise RemoteJobTransportError(
            f"remote {action} exited without returning status for {job_id}"
        )
    return payload


def _remote_failure(job_id: str, payload: dict) -> dg.Failure:
    error = payload.get("error")
    if isinstance(error, dict):
        error_type = error.get("type", "RemoteJobFailure")
        message = error.get("message", "remote job failed")
        detail = f"{error_type}: {message}"
    else:
        detail = str(error or "remote job failed")
    return dg.Failure(f"durable remote job {job_id} failed: {detail}")


def _submit_remote_job(
    context: dg.OpExecutionContext,
    spec: dict,
    job_id: str,
    request: dict,
) -> dict:
    remote = spec["remote"]
    deadline = time.monotonic() + float(remote.get("submit_timeout_seconds", 300.0))
    attempt = 0
    while True:
        attempt += 1
        try:
            return _invoke_remote_job(remote, "submit", job_id, request)
        except RemoteJobTransportError as error:
            if time.monotonic() >= deadline:
                raise dg.Failure(
                    f"could not submit durable remote job {job_id}: {error}"
                ) from error
            if attempt == 1 or attempt % 6 == 0:
                context.log.warning(
                    "Remote submit transport unavailable for %s (attempt %d): %s",
                    job_id,
                    attempt,
                    error,
                )
            time.sleep(float(remote.get("reconnect_delay_seconds", 5.0)))


def _cancel_remote_job(remote: dict, job_id: str) -> None:
    try:
        _invoke_remote_job(remote, "cancel", job_id)
    except RemoteJobTransportError:
        # The remote outer timeout remains the final safety net when cancellation
        # cannot cross the same unavailable transport.
        pass


def _wait_for_remote_job(
    context: dg.OpExecutionContext,
    spec: dict,
    job_id: str,
    initial: dict,
    started: float,
) -> dict:
    remote = spec["remote"]
    wall_timeout = _remote_wall_timeout_seconds(spec)
    deadline = started + wall_timeout if wall_timeout is not None else None
    poll_interval = float(remote.get("poll_interval_seconds", 15.0))
    reconnect_delay = float(remote.get("reconnect_delay_seconds", 5.0))
    payload = initial
    transport_failures = 0
    while True:
        if spec.get("task_execution_id") and spec.get("state_db_path"):
            touch_task(
                Path(spec["state_db_path"]),
                spec["task_execution_id"],
                remote_job_id=job_id,
            )
        state = payload.get("state")
        if state == "completed":
            row = payload.get("row")
            if not isinstance(row, dict):
                raise dg.Failure(
                    f"durable remote job {job_id} completed without a result row"
                )
            return row
        if state in {"failed", "cancelled"}:
            raise _remote_failure(job_id, payload)
        if state not in {"queued", "running"}:
            raise dg.Failure(
                f"durable remote job {job_id} returned unexpected state {state!r}"
            )
        if deadline is not None and time.monotonic() >= deadline:
            _cancel_remote_job(remote, job_id)
            raise dg.Failure(
                f"durable remote job {job_id} exceeded its outer wall-clock limit "
                f"of {wall_timeout:g} seconds"
            )

        time.sleep(poll_interval if transport_failures == 0 else reconnect_delay)
        try:
            payload = _invoke_remote_job(remote, "status", job_id)
            if transport_failures:
                context.log.info(
                    "Reconnected to durable remote job %s after %d failed status checks",
                    job_id,
                    transport_failures,
                )
            transport_failures = 0
        except RemoteJobTransportError as error:
            transport_failures += 1
            if transport_failures == 1 or transport_failures % 12 == 0:
                context.log.warning(
                    "Remote status transport unavailable for %s (%d consecutive): %s",
                    job_id,
                    transport_failures,
                    error,
                )


def _run_remote_rollout(context: dg.OpExecutionContext, spec: dict) -> dict:
    remote = spec["remote"]
    job_id = _remote_job_id(context.run_id, spec["key"])
    request = {"dagster_run_id": context.run_id, "spec": spec}
    if api_key := os.environ.get(spec["api_key_var"]):
        request["api_key"] = api_key

    context.log.info(
        "Submitting durable rollout %s on SSH host %s", job_id, remote["host"]
    )
    started = time.monotonic()
    initial = _submit_remote_job(context, spec, job_id, request)
    try:
        row = _wait_for_remote_job(context, spec, job_id, initial, started)
    except BaseException:
        # Dagster interruption is best-effort because its worker process can be
        # killed outright. When Python does receive the interruption, propagate
        # cancellation to the detached process group before unwinding locally.
        _cancel_remote_job(remote, job_id)
        raise

    remote_output_dir = row["output_dir"]
    local_output_dir = Path(spec["output_dir"]) / context.run_id / spec["key"]
    if remote.get("copy_artifacts", False) and not spec["dry_run"]:
        _copy_remote_artifacts(context, remote, remote_output_dir, local_output_dir)
    row["remote_output_dir"] = remote_output_dir
    row["output_dir"] = str(local_output_dir)
    return row


@dg.op(pool="rollouts")
def run_rollout(context: dg.OpExecutionContext, spec: dict) -> dict:
    started = time.perf_counter()
    execution_key = spec.get("task_execution_id")
    state_path = Path(spec["state_db_path"]) if spec.get("state_db_path") else None
    tracking_started = False
    if execution_key and state_path is not None:
        mark_task_running(
            state_path,
            execution_key,
            dagster_run_id=context.run_id,
            dagster_step_key=context.get_step_execution_context().step.key,
            remote_job_id=(
                _remote_job_id(context.run_id, spec["key"])
                if spec.get("remote")
                else None
            ),
        )
        tracking_started = True
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
        if tracking_started:
            mark_task_terminal(
                state_path,
                execution_key,
                state="error",
                outcome=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        _report_run_progress(
            context, database_path, context.run_id, spec["total_rollouts"]
        )
        context.log.error(
            "Persisted failed rollout to %s before re-raising: %s",
            database_path,
            error,
        )
        raise

    database_path = _persist_rollout_row(row)
    if tracking_started:
        row_error = row.get("error")
        if row.get("status") == "error":
            mark_task_terminal(
                state_path,
                execution_key,
                state="error",
                outcome=None,
                error_type=(
                    row_error.get("type")
                    if isinstance(row_error, dict)
                    else "RolloutTraceError"
                ),
                error_message=(
                    row_error.get("message")
                    if isinstance(row_error, dict)
                    else str(row_error or "rollout trace reported an error")
                ),
            )
        else:
            mark_task_terminal(
                state_path,
                execution_key,
                state="completed",
                outcome=row["status"],
            )
    progress_metadata = _report_run_progress(
        context, database_path, context.run_id, spec["total_rollouts"]
    )
    if tracking_started and row.get("status") == "error":
        detail = row.get("error")
        context.log.error("Rollout trace reported an execution error: %s", detail)
        raise dg.Failure(
            f"task execution {execution_key} completed with a rollout trace error"
        )
    metadata = dict(progress_metadata)
    for key in ("campaign_id", "combination_id", "task_execution_id"):
        if row.get(key) is not None:
            metadata[key] = row[key]
    if row.get("reasoning_effort") is not None:
        metadata["reasoning_effort"] = row["reasoning_effort"]
    if not spec["dry_run"]:
        metadata.update(
            {
                "task": spec["task_id"],
                "reward": row["reward"],
                "passed": row["passed"],
                "container_cpu_limit": row.get("container_cpu_limit"),
                "container_memory_limit_gb": row.get("container_memory_limit_gb"),
                "database": dg.MetadataValue.path(str(database_path)),
            }
        )
        if spec.get("batch_id") is not None:
            metadata["batch_id"] = spec["batch_id"]
        if row.get("resource_usage_source"):
            metadata.update(
                {
                    "cpu_seconds": row.get("cpu_seconds"),
                    "peak_memory_gib": row.get("peak_memory_bytes", 0) / (1024**3),
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


def _write_results_db(
    rows: list[dict],
    output_root: Path,
    *,
    database_path: Path | None = None,
) -> Path:
    path = database_path or output_root / "results.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        with connection:
            # Acquire the writer lock before inspecting or migrating the schema.
            # Mapped rollouts can finish together, and a deferred transaction lets
            # multiple processes all observe the same missing column before one of
            # them adds it.
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rollout_results (
                    dagster_run_id TEXT NOT NULL,
                    campaign_id TEXT,
                    combination_id TEXT,
                    task_execution_id TEXT,
                    batch_id TEXT,
                    rollout_key TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    reasoning_effort TEXT,
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
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rollout_results_lookup "
                "ON rollout_results (model, harness, task_id, status)"
            )
            existing_columns = {
                info[1]
                for info in connection.execute("PRAGMA table_info(rollout_results)")
            }
            for column, column_type in {
                "reasoning_tokens": "INTEGER",
                "reasoning_effort": "TEXT",
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
                "batch_id": "TEXT",
                "campaign_id": "TEXT",
                "combination_id": "TEXT",
                "task_execution_id": "TEXT",
            }.items():
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE rollout_results ADD COLUMN {column} {column_type}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rollout_results_batch "
                "ON rollout_results (batch_id)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS rollout_results_task_execution "
                "ON rollout_results (task_execution_id) "
                "WHERE task_execution_id IS NOT NULL"
            )

            columns = (
                "dagster_run_id",
                "campaign_id",
                "combination_id",
                "task_execution_id",
                "batch_id",
                "rollout_key",
                "timestamp",
                "model",
                "reasoning_effort",
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
        error.get("message") or error.get("detail") if isinstance(error, dict) else None
    )
    return (
        row["dagster_run_id"],
        row.get("campaign_id"),
        row.get("combination_id"),
        row.get("task_execution_id"),
        row.get("batch_id"),
        row["rollout_key"],
        row["timestamp"],
        row["model"],
        row.get("reasoning_effort"),
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
    if row.get("state_db_path"):
        database_path = Path(row["state_db_path"])
        return _write_results_db(
            [row], database_path.parent, database_path=database_path
        )
    output_root = Path(row["output_dir"]).parents[1]
    output_root.mkdir(parents=True, exist_ok=True)
    return _write_results_db([row], output_root)


def _progress_text(completed: int, total: int) -> str:
    percent = 100 * completed / total
    return f"{percent:.0f}% · {completed}/{total} rollouts"


def _report_run_progress(
    context: dg.OpExecutionContext,
    database_path: Path,
    dagster_run_id: str,
    total_rollouts: int,
) -> dict[str, int | float | str]:
    # Serializing the count and tag update against the results database prevents
    # concurrently finishing mapped steps from publishing progress out of order.
    connection = sqlite3.connect(database_path, timeout=30)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("BEGIN IMMEDIATE")
        completed = connection.execute(
            "SELECT COUNT(*) FROM rollout_results WHERE dagster_run_id = ?",
            (dagster_run_id,),
        ).fetchone()[0]
        percent = 100 * completed / total_rollouts
        progress = _progress_text(completed, total_rollouts)
        context.instance.add_run_tags(context.run_id, {PROGRESS_TAG: progress})
        connection.commit()
    finally:
        connection.close()

    context.log.info("Run progress: %s", progress)
    return {
        "progress": progress,
        "progress_percent": percent,
        "completed_rollouts": completed,
        "total_rollouts": total_rollouts,
    }


@dg.op
def write_results(context: dg.OpExecutionContext, rows: list[dict]) -> str:
    if not rows:
        raise dg.Failure("no rollout results were produced")
    run_dir = Path(rows[0]["output_dir"]).parent
    configured_database = (
        Path(rows[0]["state_db_path"]) if rows[0].get("state_db_path") else None
    )
    database_path = _write_results_db(
        rows,
        run_dir.parent,
        database_path=configured_database,
    )
    measured = [row for row in rows if row.get("resource_usage_source")]
    metadata = {
        "database": dg.MetadataValue.path(str(database_path)),
        "dagster_run_id": rows[0]["dagster_run_id"],
        "rollouts": len(rows),
        "passed": sum(row.get("passed") is True for row in rows),
    }
    if rows[0].get("batch_id") is not None:
        metadata["batch_id"] = rows[0]["batch_id"]
    if rows[0].get("reasoning_effort") is not None:
        metadata["reasoning_effort"] = rows[0]["reasoning_effort"]
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


@dg.job
def plan_terminal_bench_campaign() -> None:
    plan_campaign()


@dg.job(executor_def=rollout_executor)
def terminal_bench_combination() -> None:
    write_results(plan_task_executions().map(run_rollout).collect())


def _combination_run_tags(row: dict, run_config: dict) -> dict[str, str]:
    config = run_config["ops"]["plan_task_executions"]["config"]
    tags = {
        CAMPAIGN_ID_TAG: row["campaign_id"],
        COMBINATION_ID_TAG: row["combination_id"],
        RUN_TYPE_TAG: "test" if config["dry_run"] else config["run_type"],
        DRY_RUN_TAG: str(config["dry_run"]).lower(),
        MODEL_TAG: row["model"],
        HARNESS_TAG: row["harness"],
        HARNESS_VERSION_TAG: row["harness_version"],
        HARNESS_MATRIX_TAG: f"{row['harness']}@{row['harness_version']}",
        REPETITION_TAG: str(row["repetition"]),
        TASK_SELECTION_TAG: config["task_selection_mode"],
        TASK_COUNT_TAG: str(len(config["task_ids"])),
        PROGRESS_TAG: _progress_text(0, len(config["task_ids"])),
    }
    if config.get("batch_id") is not None:
        tags[BATCH_ID_TAG] = config["batch_id"]
    if config.get("reasoning_effort") is not None:
        tags[REASONING_EFFORT_TAG] = config["reasoning_effort"]
    return tags


@dg.sensor(
    job=terminal_bench_combination,
    minimum_interval_seconds=5,
    default_status=dg.DefaultSensorStatus.RUNNING,
    description="Launch and reconcile one Dagster run per benchmark combination.",
)
def terminal_bench_campaign_sensor(
    context: dg.SensorEvaluationContext,
) -> Iterator[dg.RunRequest | dg.SkipReason]:
    database_path = state_database_path()
    if not database_path.is_file():
        yield dg.SkipReason(f"state database does not exist yet: {database_path}")
        return

    # Bind queued RunRequests to their physical Dagster IDs and project Dagster
    # terminal states onto tasks that could not report their own final state.
    for combination in active_combinations(database_path):
        run = None
        if combination.get("dagster_run_id"):
            run = context.instance.get_run_by_id(combination["dagster_run_id"])
        if run is None:
            matching = context.instance.get_runs(
                filters=dg.RunsFilter(
                    job_name=terminal_bench_combination.name,
                    tags={COMBINATION_ID_TAG: combination["combination_id"]},
                ),
                limit=1,
            )
            run = matching[0] if matching else None
        if run is not None:
            reconcile_combination(
                database_path,
                combination["combination_id"],
                dagster_run_id=run.run_id,
                dagster_status=run.status.value,
            )

    try:
        launch_limit = int(os.environ.get("HARNESS_BLOAT_CAMPAIGN_LAUNCH_BATCH", "100"))
    except ValueError:
        launch_limit = 100
    launch_limit = max(1, launch_limit)
    pending = launchable_combinations(database_path, limit=launch_limit)
    if not pending:
        yield dg.SkipReason("no campaign combinations are waiting to launch")
        return

    for combination in pending:
        run_config = json.loads(combination["run_config_json"])
        mark_combination_queued(database_path, combination["combination_id"])
        yield dg.RunRequest(
            run_key=combination["combination_id"],
            run_config=run_config,
            tags=_combination_run_tags(combination, run_config),
        )


defs = dg.Definitions(
    jobs=[
        plan_terminal_bench_campaign,
        terminal_bench_combination,
        terminal_bench_rollouts,
    ],
    sensors=[terminal_bench_campaign_sensor],
)
