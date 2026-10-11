from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .vba_rubric import DATASET, TASK_ID


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(StrictConfig):
    model: str
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_output_tokens: int = Field(default=8192, ge=64, le=32768)
    max_calls: int = Field(default=150, ge=1, le=500)
    api: Literal["chat", "responses"] = "chat"


class HarnessSpec(StrictConfig):
    id: Literal["hermes_agent", "opencode", "omp_agent", "claude_code_agent", "codex_agent"]
    version: str


class SandboxSpec(StrictConfig):
    image: str = "ubuntu:24.04"
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64"
    worker_cpus: float = Field(default=2, gt=0)
    worker_memory_gb: float = Field(default=3, gt=0)
    task_memory_gb: dict[str, float] = Field(default_factory=dict)
    coordinator_cpus: float = Field(default=1, gt=0)
    coordinator_memory_gb: float = Field(default=2, gt=0)
    shared_cpus: float = Field(default=4, gt=0)
    shared_memory_gb: float = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_memory(self):
        if any(value <= 0 for value in self.task_memory_gb.values()):
            raise ValueError("task memory limits must be positive")
        return self


class TeamConfig(StrictConfig):
    schema_version: Literal[1] = 1
    benchmark_type: Literal["multiagent"] = "multiagent"
    execution_mode: Literal["shared_container", "isolated_tasks"] = "shared_container"
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    dataset: str = DATASET
    task_ids: list[str] = Field(default_factory=lambda: [TASK_ID], min_length=1)
    scoring: Literal["vba_subtasks_v1", "task"] = "vba_subtasks_v1"
    harness: HarnessSpec
    orchestrator: ModelSpec
    executor: ModelSpec
    max_parallel_workers: int | None = Field(default=None, ge=1)
    batch_timeout_seconds: float = Field(default=4500, gt=0, le=29700)
    worker_timeout_seconds: float = Field(default=900, gt=0, le=28800)
    grading_reserve_seconds: float = Field(default=3600, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    sandbox: SandboxSpec = Field(default_factory=SandboxSpec)

    @model_validator(mode="before")
    @classmethod
    def preserve_historical_scoring(cls, value):
        # Frozen schema-v1 configs from the 2.1 pilot predate this field.
        if isinstance(value, dict) and "scoring" not in value and value.get("dataset", DATASET) != DATASET:
            return {**value, "scoring": "task"}
        return value

    @model_validator(mode="after")
    def validate_team(self):
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task IDs must be unique")
        if any(not task or "/" in task or ".." in task for task in self.task_ids):
            raise ValueError("task IDs must be Harbor task names")
        if self.grading_reserve_seconds >= self.batch_timeout_seconds:
            raise ValueError("grading reserve must be shorter than the batch deadline")
        # Shared helpers only do agent work; grading happens after they stop.
        # Legacy isolated workers include their own grader in their deadline.
        if self.execution_mode == "isolated_tasks" and self.grading_reserve_seconds >= self.worker_timeout_seconds:
            raise ValueError("grading reserve must be shorter than the isolated worker deadline")
        if self.scoring == "vba_subtasks_v1":
            if self.dataset != DATASET or self.task_ids != [TASK_ID] or self.execution_mode != "shared_container":
                raise ValueError("vba_subtasks_v1 requires shared_container and the pinned Terminal-Bench 4.0 vba-userform-port task")
        elif self.execution_mode == "shared_container":
            supported = {"pypi-server", "fix-code-vulnerability", "qemu-startup", "sanitize-git-repo"}
            if self.dataset != "terminal-bench/terminal-bench-2-1" or set(self.task_ids) != supported:
                raise ValueError("Legacy task scoring in shared_container requires the four-task Terminal-Bench 2.1 pilot")
        if self.scoring == "task" and self.harness.id not in ("hermes_agent", "opencode"):
            raise ValueError("Legacy multiagent protocols support only Hermes and OpenCode")
        return self


def load_config(path: Path) -> TeamConfig:
    return TeamConfig.model_validate(json.loads(path.read_text()))
