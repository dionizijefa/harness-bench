import asyncio
import datetime as dt
import json
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

HarnessId = Literal["codex", "opencode", "pi", "omp_agent"]
DEFAULT_HARNESS_VERSIONS: dict[HarnessId, str] = {
    "codex": "0.137.0",
    "opencode": "1.18.1",
    "pi": "0.80.7",
    "omp_agent": "16.5.2",
}


class HarnessSpec(dg.Config):
    id: HarnessId = "codex"
    version: str | None = None


class MatrixConfig(dg.Config):
    models: list[str] = ["qwen/qwen3.7-max"]
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
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_var: str = "OPENROUTER_API_KEY"
    max_tokens: int | None = None
    temperature: float | None = None
    rollout_retries: int = 0
    output_dir: str = "outputs"
    dry_run: bool = False


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
    if config.dry_run and requested:
        return requested
    taskset = HarborTaskset(
        HarborConfig(
            id="harbor",
            dataset=config.dataset,
            tasks=requested or None,
        )
    )
    return [Path(task.data.task_dir).name for task in taskset.load()]


@dg.op(out=dg.DynamicOut(dict))
def plan_rollouts(
    context: dg.OpExecutionContext, config: MatrixConfig
) -> Iterator[dg.DynamicOutput[dict]]:
    if config.num_rollouts < 1:
        raise dg.Failure("num_rollouts must be at least 1")

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
                "base_url": config.base_url,
                "api_key_var": config.api_key_var,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "rollout_retries": config.rollout_retries,
                "output_dir": config.output_dir,
                "dry_run": config.dry_run,
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
                "runtime": {"type": spec["runtime"]},
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


@dg.op(pool="rollouts")
def run_rollout(context: dg.OpExecutionContext, spec: dict) -> dict:
    output_dir = Path(spec["output_dir"]) / context.run_id / spec["key"]
    base = {
        "dagster_run_id": context.run_id,
        "model": spec["model"],
        "harness": spec["harness"],
        "harness_version": spec["harness_version"],
        "dataset": spec["dataset"],
        "task_id": spec["task_id"],
        "rollout": spec["rollout"],
        "output_dir": str(output_dir),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if spec["dry_run"]:
        return {**base, "status": "dry_run", "passed": None, "reward": None}

    started = time.perf_counter()
    traces = asyncio.run(
        run_eval(
            Environment(config := _eval_config(spec, output_dir)),
            config,
        )
    )
    trace = traces[0]
    usage = trace.usage
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
        "input_tokens": usage.input_tokens if usage else trace.num_input_tokens,
        "cached_input_tokens": usage.cached_input_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else trace.num_output_tokens,
        "total_tokens": usage.total_tokens if usage else trace.num_total_tokens,
    }
    context.add_output_metadata(
        {
            "task": spec["task_id"],
            "reward": reward,
            "passed": row["passed"],
            "trace": dg.MetadataValue.path(str(output_dir / "traces.jsonl")),
        }
    )
    return row


@dg.op
def write_results(context: dg.OpExecutionContext, rows: list[dict]) -> str:
    if not rows:
        raise dg.Failure("no rollout results were produced")
    output_dir = Path(rows[0]["output_dir"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    context.add_output_metadata(
        {
            "path": dg.MetadataValue.path(str(path)),
            "rollouts": len(rows),
            "passed": sum(row.get("passed") is True for row in rows),
        }
    )
    return str(path)


@dg.job(executor_def=dg.multiprocess_executor)
def terminal_bench_rollouts() -> None:
    write_results(plan_rollouts().map(run_rollout).collect())


defs = dg.Definitions(jobs=[terminal_bench_rollouts])
