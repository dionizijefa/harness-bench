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
    deepseek = eval_config(
        "deepseek_harness",
        "0.1.1-rc.2",
        DEFAULT_MODEL,
        tmp_path / "deepseek",
    )
    pi = eval_config("pi_agent", "0.84.2", DEFAULT_MODEL, tmp_path / "pi")
    pi_rlm = eval_config(
        "pi_rlm_runtime",
        "0.1.1",
        DEFAULT_MODEL,
        tmp_path / "pi-rlm",
    )

    configs = [codex, claude, omp, deepseek, pi, pi_rlm]
    assert {config.model for config in configs} == {
        "deepseek/deepseek-v4-flash-0731"
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
    assert deepseek.harness.id == "deepseek_harness"
    assert pi.harness.id == "pi_agent"
    assert pi_rlm.harness.id == "pi_rlm_runtime"
    assert all(
        config.harness.runtime.model_dump() == codex.harness.runtime.model_dump()
        for config in configs
    )
    assert all(config.timeout == codex.timeout for config in configs)
    assert all(config.retries == codex.retries for config in configs)
