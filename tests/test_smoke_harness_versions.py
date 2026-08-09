from pathlib import Path

from harness_bloat_bench.smoke_harness_versions import (
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    eval_config,
)


def test_codex_and_claude_code_share_the_hello_world_smoke_setup(
    tmp_path: Path,
) -> None:
    codex = eval_config("codex_agent", "0.147.0", DEFAULT_MODEL, tmp_path / "codex")
    claude = eval_config(
        "claude_code_agent",
        "2.1.226",
        DEFAULT_MODEL,
        tmp_path / "claude",
    )

    assert codex.model == claude.model == "~deepseek/deepseek-v4-flash-latest"
    assert codex.client.base_url == claude.client.base_url == OPENROUTER_BASE_URL
    assert codex.client.api_key_var == claude.client.api_key_var == (
        "OPENROUTER_API_KEY"
    )
    assert codex.taskset.model_dump() == claude.taskset.model_dump()
    assert codex.harness.id == "codex_agent"
    assert claude.harness.id == "claude_code_agent"
    assert codex.harness.runtime.model_dump() == claude.harness.runtime.model_dump()
    assert codex.timeout == claude.timeout
    assert codex.retries == claude.retries
