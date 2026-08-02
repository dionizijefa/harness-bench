import json
from pathlib import Path

import pytest

from harness_bloat_bench.definitions import _eval_config, terminal_bench_rollouts


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
    path = Path(result.output_for_node("write_results"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    assert {row["model"] for row in rows} == {"model-a", "model-b"}
    assert {row["task_id"] for row in rows} == {"task-a"}
    assert all(row["status"] == "dry_run" for row in rows)


def test_eval_config_uses_v1_components(tmp_path: Path) -> None:
    config = _eval_config(
        {
            "model": "qwen/qwen3.7-max",
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
            "model": "qwen/qwen3.7-max",
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
    path = Path(result.output_for_node("write_results"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {(row["harness"], row["harness_version"]) for row in rows} == {
        ("codex", "0.137.0"),
        ("opencode", "1.18.1"),
        ("pi", "0.80.7"),
        ("omp_agent", "16.5.2"),
    }
