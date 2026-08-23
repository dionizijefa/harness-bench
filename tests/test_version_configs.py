from pathlib import Path

import dagster as dg
import yaml

from harness_bloat_bench.definitions import terminal_bench_rollouts


PROJECT_ROOT = Path(__file__).parents[1]


def _matrix_versions(filename: str) -> list[tuple[str, str]]:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / filename).read_text())
    dg.validate_run_config(terminal_bench_rollouts, config)
    harnesses = config["ops"]["plan_rollouts"]["config"]["harnesses"]
    return [(entry["id"], entry["version"]) for entry in harnesses]


def test_codex_agent_history_matrix() -> None:
    versions = [
        "0.78.0",
        "0.88.0",
        "0.94.0",
        "0.102.0",
        "0.110.0",
        "0.118.0",
        "0.124.0",
        "0.134.0",
        "0.140.0",
        "0.147.0",
    ]

    assert _matrix_versions("codex-agent-versions.yaml") == [
        ("codex_agent", version) for version in versions
    ]


def test_omp_agent_history_matrix() -> None:
    versions = [
        "11.3.0",
        "11.13.1",
        "12.6.0",
        "12.16.0",
        "13.6.2",
        "13.14.2",
        "14.4.4",
        "15.4.3",
        "15.12.4",
        "17.2.10",
    ]

    assert _matrix_versions("omp-agent-versions.yaml") == [
        ("omp_agent", version) for version in versions
    ]


def test_prime_agent_history_matrix() -> None:
    versions = [
        "0.2.6",
        "0.2.7",
        "0.2.9",
        "0.3.0",
        "0.3.2",
        "0.3.3",
        "0.4.0",
        "0.5.1",
        "0.6.1",
        "0.7.1",
    ]

    assert _matrix_versions("prime-agent-versions.yaml") == [
        ("prime_agent", version) for version in versions
    ]


def test_claude_code_history_matrix() -> None:
    versions = [
        "2.1.97",
        "2.1.109",
        "2.1.118",
        "2.1.128",
        "2.1.140",
        "2.1.159",
        "2.1.186",
        "2.1.200",
        "2.1.210",
        "2.1.226",
    ]

    assert _matrix_versions("claude-code-versions.yaml") == [
        ("claude_code_agent", version) for version in versions
    ]
