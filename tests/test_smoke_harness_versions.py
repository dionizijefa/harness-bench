from pathlib import Path

from harness_bloat_bench.smoke_harness_versions import (
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    eval_config,
)


def test_versioned_harnesses_share_the_hello_world_smoke_setup(
    tmp_path: Path,
) -> None:
    codex = eval_config("codex_agent", "0.147.0", DEFAULT_MODEL, tmp_path / "codex")
    claude = eval_config(
        "claude_code_agent",
        "2.1.226",
        DEFAULT_MODEL,
        tmp_path / "claude",
    )
    omp = eval_config("omp_agent", "17.2.10", DEFAULT_MODEL, tmp_path / "omp")

    configs = [codex, claude, omp]
    assert {config.model for config in configs} == {
        "~deepseek/deepseek-v4-flash-latest"
    }
    assert {config.client.base_url for config in configs} == {OPENROUTER_BASE_URL}
    assert {config.client.api_key_var for config in configs} == {
        "OPENROUTER_API_KEY"
    }
    assert all(
        config.taskset.model_dump() == codex.taskset.model_dump()
        for config in configs
    )
    assert codex.harness.id == "codex_agent"
    assert claude.harness.id == "claude_code_agent"
    assert omp.harness.id == "omp_agent"
    assert all(
        config.harness.runtime.model_dump() == codex.harness.runtime.model_dump()
        for config in configs
    )
    assert all(config.timeout == codex.timeout for config in configs)
    assert all(config.retries == codex.retries for config in configs)
