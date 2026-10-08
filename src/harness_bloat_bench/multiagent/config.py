from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(StrictConfig):
    model: str
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_output_tokens: int = Field(default=8192, ge=64, le=32768)
    max_calls: int = Field(default=150, ge=1, le=500)
    api: Literal["chat", "responses"] = "chat"


class HarnessSpec(StrictConfig):
    id: Literal["hermes_agent", "opencode"]
    version: str


class SandboxSpec(StrictConfig):
    image: str = "ubuntu:24.04"
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64"
    worker_cpus: float = Field(default=2, gt=0)
    worker_memory_gb: float = Field(default=3, gt=0)
    task_memory_gb: dict[str, float] = Field(default_factory=dict)
    coordinator_cpus: float = Field(default=1, gt=0)
    coordinator_memory_gb: float = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_memory(self):
        if any(value <= 0 for value in self.task_memory_gb.values()):
            raise ValueError("task memory limits must be positive")
        return self


class TeamConfig(StrictConfig):
    schema_version: Literal[1] = 1
    benchmark_type: Literal["multiagent"] = "multiagent"
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    dataset: str = "terminal-bench/terminal-bench-2-1"
    task_ids: list[str] = Field(min_length=1)
    harness: HarnessSpec
    orchestrator: ModelSpec
    executor: ModelSpec
    max_parallel_workers: int = Field(default=2, ge=1, le=16)
    batch_timeout_seconds: float = Field(default=1200, gt=0, le=1200)
    worker_timeout_seconds: float = Field(default=1200, gt=0, le=1200)
    grading_reserve_seconds: float = Field(default=90, gt=0)
    max_cost_usd: float = Field(default=4, gt=0)
    sandbox: SandboxSpec = Field(default_factory=SandboxSpec)

    @model_validator(mode="after")
    def validate_team(self):
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task IDs must be unique")
        if any(not task or "/" in task or ".." in task for task in self.task_ids):
            raise ValueError("task IDs must be Harbor task names")
        if self.grading_reserve_seconds >= min(self.batch_timeout_seconds, self.worker_timeout_seconds):
            raise ValueError("grading reserve must be shorter than both lifecycle deadlines")
        return self


def load_config(path: Path) -> TeamConfig:
    return TeamConfig.model_validate(json.loads(path.read_text()))
