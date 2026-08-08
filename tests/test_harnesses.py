import asyncio
import json
from types import SimpleNamespace
from typing import get_args

import pytest
from verifiers.v1.loaders import harness_config_type, load_harness
from verifiers.v1.runtimes import ProgramResult

from harness_bloat_bench.definitions import DEFAULT_HARNESS_VERSIONS, HarnessId
from harness_bloat_bench.harnesses.claude_code import (
    ClaudeCodeHarness,
    ClaudeCodeHarnessConfig,
    _claude_bin,
)
from harness_bloat_bench.harnesses.claude_code import (
    _install_script as claude_code_install_script,
)
from harness_bloat_bench.harnesses.codex_agent import (
    CodexAgentHarness,
    CodexHarnessConfig,
)
from harness_bloat_bench.harnesses.codex_agent import (
    _install_script as codex_install_script,
)
from harness_bloat_bench.harnesses.codex_agent import _versioned_argv
from harness_bloat_bench.harnesses.hermes_agent import (
    HERMES_BIN,
    HERMES_RELEASE_TAGS,
    HermesAgentHarness,
    HermesAgentHarnessConfig,
)
from harness_bloat_bench.harnesses.hermes_agent import (
    _install_script as hermes_install_script,
)
from harness_bloat_bench.harnesses.omp_agent import (
    OMP_BIN,
    OmpAgentHarness,
    OmpAgentHarnessConfig,
)
from harness_bloat_bench.harnesses.omp_agent import (
    _install_script as omp_install_script,
)
from harness_bloat_bench.harnesses.opencode import (
    OPENCODE_BIN,
    OpenCodeHarness,
    OpenCodeHarnessConfig,
)
from harness_bloat_bench.harnesses.opencode import (
    _install_script as opencode_install_script,
)
from harness_bloat_bench.harnesses.pi import PI_BIN, PiHarness, PiHarnessConfig
from harness_bloat_bench.harnesses.prime_agent import (
    PRIME_AGENT_CLI,
    PRIME_AGENT_KERNEL_VENV,
    PRIME_AGENT_NODE_BIN,
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)
from harness_bloat_bench.harnesses.prime_agent import (
    _install_script as prime_agent_install_script,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.program: tuple[list[str], dict[str, str]] | None = None

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append((argv, env))
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        self.writes[path] = data

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.program = (argv, env)
        return ProgramResult(exit_code=0, stdout="", stderr="")


def trace(prompt: str = "Fix the tests", system_prompt: str = "Be precise"):
    data = SimpleNamespace(prompt=prompt, system_prompt=system_prompt)
    rollout_trace = SimpleNamespace(
        id="trace-123",
        task=SimpleNamespace(data=data),
        stop_condition=None,
        stop_reason=None,
    )
    rollout_trace.stop = lambda reason: setattr(rollout_trace, "stop_reason", reason)
    return rollout_trace


def context():
    return SimpleNamespace(
        model="~deepseek/deepseek-v4-flash-latest",
        sampling=SimpleNamespace(reasoning_effort=None),
    )


def test_every_declared_harness_has_a_default_version() -> None:
    assert set(get_args(HarnessId)) == set(DEFAULT_HARNESS_VERSIONS)


@pytest.mark.parametrize("harness_id", get_args(HarnessId))
def test_every_harness_resolves_sets_up_and_runs(harness_id: str) -> None:
    """Exercise the production plugin boundary without network or model calls."""
    runtime = FakeRuntime()
    rollout_trace = trace()
    config_class = harness_config_type(harness_id)
    config = config_class.model_validate(
        {"id": harness_id, "version": DEFAULT_HARNESS_VERSIONS[harness_id]}
    )
    harness = load_harness(config)
    mcp_urls = (
        {"task-tools": "http://127.0.0.1:9001/mcp"} if harness.SUPPORTS_MCP else {}
    )

    async def exercise() -> None:
        await harness.setup(runtime)
        await harness.run(
            context(),
            rollout_trace,
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            mcp_urls,
        )

    asyncio.run(exercise())

    assert rollout_trace.stop_reason == "agent_completed"
    assert runtime.commands, f"{harness_id} setup did not invoke the runtime"
    assert runtime.program is not None, f"{harness_id} did not launch a program"
    argv, _ = runtime.program
    assert argv and argv[0]
    assert "session-secret" not in argv


CODEX_HISTORY_VERSIONS = [
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


@pytest.mark.parametrize("version", CODEX_HISTORY_VERSIONS)
def test_requested_codex_versions_use_pinned_release_artifacts(version: str) -> None:
    script = codex_install_script(version)

    assert f"VERSION={version}" in script
    assert f"releases/download/rust-v{version}/codex-" in script
    assert 'current="$($BIN --version 2>/dev/null || true)"' in script
    assert '"codex-cli $VERSION"' in script


@pytest.mark.parametrize(
    ("version", "expected_features", "expected_v2_config"),
    [
        ("0.78.0", set(), None),
        ("0.94.0", {"apps"}, None),
        ("0.102.0", {"apps", "multi_agent"}, None),
        ("0.110.0", {"apps", "plugins", "multi_agent"}, None),
        (
            "0.118.0",
            {"apps", "plugins", "multi_agent"},
            "features.multi_agent_v2=false",
        ),
        (
            "0.124.0",
            {"apps", "plugins", "multi_agent"},
            "features.multi_agent_v2.enabled=false",
        ),
    ],
)
def test_codex_launch_tracks_versioned_feature_surface(
    version: str,
    expected_features: set[str],
    expected_v2_config: str | None,
) -> None:
    original = [
        "codex",
        "exec",
        "--disable",
        "apps",
        "--disable",
        "plugins",
        "--disable",
        "multi_agent",
        "-c",
        "features.multi_agent_v2.enabled=false",
        "--",
        "Fix the tests",
    ]

    argv = _versioned_argv(original, version)
    disabled = {
        argv[index + 1]
        for index, arg in enumerate(argv[:-1])
        if arg == "--disable"
    }
    configs = {
        argv[index + 1] for index, arg in enumerate(argv[:-1]) if arg == "-c"
    }

    assert disabled == expected_features
    assert configs == ({expected_v2_config} if expected_v2_config else set())


def test_codex_harness_applies_versioned_launch_command() -> None:
    runtime = FakeRuntime()
    harness = CodexAgentHarness(
        CodexHarnessConfig(id="codex_agent", version="0.78.0")
    )

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {},
        )
    )

    assert runtime.program is not None
    argv, _ = runtime.program
    assert "--disable" not in argv
    assert not any("multi_agent_v2" in arg for arg in argv)


CLAUDE_CODE_HISTORY_VERSIONS = [
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


@pytest.mark.parametrize("version", CLAUDE_CODE_HISTORY_VERSIONS)
def test_requested_claude_code_versions_use_pinned_official_installer(
    version: str,
) -> None:
    script = claude_code_install_script(version)

    assert "https://claude.ai/install.sh" in script
    assert f"bash -s {version}" in script


def test_claude_code_routes_deepseek_through_anthropic_interception() -> None:
    runtime = FakeRuntime()
    harness = ClaudeCodeHarness(
        ClaudeCodeHarnessConfig(id="claude_code_agent", disabled_tools=["WebFetch"])
    )

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {"task-tools": "http://127.0.0.1:9001/mcp"},
        )
    )

    assert runtime.program is not None
    argv, env = runtime.program
    mcp_path = "/tmp/vf-claude-code-state-trace-123/mcp.json"
    mcp = json.loads(runtime.writes[mcp_path])
    model = "~deepseek/deepseek-v4-flash-latest"
    assert argv[0] == _claude_bin("2.1.226")
    assert "--dangerously-skip-permissions" in argv
    assert argv[argv.index("--model") + 1] == model
    assert argv[argv.index("--append-system-prompt") + 1] == "Be precise"
    assert argv[argv.index("--disallowedTools") + 1] == "WebFetch"
    assert argv[-2:] == ["--", "Fix the tests"]
    assert mcp["mcpServers"]["task-tools"] == {
        "type": "http",
        "url": "http://127.0.0.1:9001/mcp",
    }
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "session-secret"
    assert env["ANTHROPIC_MODEL"] == model
    assert env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == model
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == model
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == model
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == model
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == model


PRIME_AGENT_HISTORY_VERSIONS = [
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


@pytest.mark.parametrize("version", PRIME_AGENT_HISTORY_VERSIONS)
def test_requested_prime_agent_versions_use_pinned_release_artifacts(
    version: str,
) -> None:
    script = prime_agent_install_script(version)

    assert f"VERSION={version}" in script
    assert f"releases/download/v{version}/prime-agent-{version}.tgz" in script
    assert "PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=1" in script
    assert "nodejs.org/dist/v$NODE_VERSION" in script


def test_prime_agent_launch_keeps_stock_rlm_surface() -> None:
    runtime = FakeRuntime()
    harness = PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime_agent"))

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {},
        )
    )

    assert runtime.program is not None
    argv, env = runtime.program
    models_path = "/tmp/vf-prime-agent-state-trace-123/models.json"
    models = json.loads(runtime.writes[models_path])
    provider = models["providers"]["verifiers"]
    assert argv[:2] == [PRIME_AGENT_NODE_BIN, PRIME_AGENT_CLI]
    assert argv[argv.index("--provider") + 1] == "verifiers"
    assert argv[argv.index("--model") + 1] == "model"
    assert argv[argv.index("--thinking") + 1] == "high"
    assert argv[argv.index("--append-system-prompt") + 1] == "Be precise"
    assert argv[-3:] == ["--print", "--", "Fix the tests"]
    assert provider["baseUrl"] == "http://127.0.0.1:9000/v1"
    assert provider["apiKey"] == "VF_INTERCEPT_KEY"
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["PRIME_AGENT_CODING_AGENT_DIR"] == (
        "/tmp/vf-prime-agent-state-trace-123"
    )
    assert env["PRIME_AGENT_KERNEL_VENV"] == PRIME_AGENT_KERNEL_VENV


def test_opencode_launch_uses_isolated_config() -> None:
    runtime = FakeRuntime()
    harness = OpenCodeHarness(
        OpenCodeHarnessConfig(id="opencode", disabled_tools=["websearch"])
    )

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {"task-tools": "http://127.0.0.1:9001/mcp"},
        )
    )

    assert runtime.program is not None
    argv, env = runtime.program
    config_path = "/tmp/vf-opencode-state-trace-123/xdg-config/opencode/config.json"
    config = json.loads(runtime.writes[config_path])
    assert argv[0] == OPENCODE_BIN
    assert argv[-1] == "Be precise\n\nFix the tests"
    assert argv == [
        OPENCODE_BIN,
        "run",
        "--model",
        "verifiers/model",
        "Be precise\n\nFix the tests",
    ]
    assert config["provider"]["verifiers"]["options"]["baseURL"].endswith("/v1")
    assert config["provider"]["verifiers"]["options"]["apiKey"] == (
        "{env:VF_INTERCEPT_KEY}"
    )
    assert config["tools"] == {"websearch": False}
    assert config["mcp"]["task-tools"]["type"] == "remote"
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["OPENCODE_CONFIG"] == config_path
    assert b"session-secret" not in runtime.writes[config_path]


@pytest.mark.parametrize(
    "version",
    [
        "0.1.196",
        "0.3.133",
        "0.5.29",
        "0.7.9",
        "0.9.11",
        "0.11.8",
        "0.13.9",
        "0.15.31",
        "1.1.65",
        "1.3.17",
        "1.14.51",
        "1.16.2",
        "1.18.13",
    ],
)
def test_requested_opencode_versions_use_pinned_npm_artifacts(version: str) -> None:
    script = opencode_install_script(version)

    assert "registry.npmjs.org/$package/-/$package-$ARTIFACT_VERSION.tgz" in script
    assert f"VERSION={version}" in script
    if version == "0.1.196":
        assert "ARTIFACT_VERSION=0.1.195" in script
    else:
        assert f"ARTIFACT_VERSION={version}" in script


@pytest.mark.parametrize(
    ("version", "expected_permission"),
    [
        ("0.1.196", None),
        ("0.3.133", {"edit": "allow", "bash": "allow"}),
        (
            "0.5.29",
            {"edit": "allow", "bash": "allow", "webfetch": "allow"},
        ),
        ("1.18.13", {"*": "allow"}),
    ],
)
def test_opencode_config_tracks_versioned_schema(
    version: str, expected_permission: dict[str, str] | None
) -> None:
    runtime = FakeRuntime()
    harness = OpenCodeHarness(OpenCodeHarnessConfig(id="opencode", version=version))

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {},
        )
    )

    config_path = "/tmp/vf-opencode-state-trace-123/xdg-config/opencode/config.json"
    config = json.loads(runtime.writes[config_path])
    assert config.get("permission") == expected_permission
    if version == "0.1.196":
        assert config["autoshare"] is False
        assert config["provider"]["verifiers"]["options"]["apiKey"] == (
            "session-secret"
        )
    else:
        assert config["share"] == "disabled"
        assert config["provider"]["verifiers"]["options"]["apiKey"] == (
            "{env:VF_INTERCEPT_KEY}"
        )


def test_pi_launch_uses_full_coding_toolset_and_ephemeral_state() -> None:
    runtime = FakeRuntime()
    harness = PiHarness(PiHarnessConfig(id="pi", disabled_tools=["write"]))

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {},
        )
    )

    assert runtime.program is not None
    argv, env = runtime.program
    models_path = "/tmp/vf-pi-agent-trace-123/models.json"
    models = json.loads(runtime.writes[models_path])
    provider = models["providers"]["verifiers"]
    assert argv[0] == PI_BIN
    assert argv[argv.index("--tools") + 1] == "read,bash,edit,grep,find,ls"
    assert argv[argv.index("--thinking") + 1] == "medium"
    assert argv[argv.index("--append-system-prompt") + 1] == "Be precise"
    assert argv[-2:] == ["--print", "Fix the tests"]
    assert "session-secret" not in argv
    assert provider["baseUrl"].endswith("/v1")
    assert provider["apiKey"] == "$VF_INTERCEPT_KEY"
    assert provider["models"][0]["maxTokens"] == 32_000
    assert env["PI_CODING_AGENT_DIR"] == "/tmp/vf-pi-agent-trace-123"
    assert env["PI_OFFLINE"] == "1"


def test_omp_launch_keeps_stock_agent_and_wires_mcp() -> None:
    runtime = FakeRuntime()
    harness = OmpAgentHarness(OmpAgentHarnessConfig(id="omp_agent"))

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {"task-tools": "http://127.0.0.1:9001/mcp"},
        )
    )

    assert runtime.program is not None
    argv, env = runtime.program
    models = json.loads(runtime.writes["/tmp/vf-omp-agent-trace-123/models.yml"])
    mcp = json.loads(runtime.writes["/tmp/vf-omp-agent-trace-123/mcp.json"])
    assert argv[0] == OMP_BIN
    assert "--auto-approve" in argv
    assert argv[argv.index("--thinking") + 1] == "high"
    assert argv[-3:] == ["--print", "--", "Fix the tests"]
    assert "session-secret" not in argv
    assert models["providers"]["verifiers"]["apiKey"] == "VF_INTERCEPT_KEY"
    assert mcp["mcpServers"]["task-tools"] == {
        "enabled": True,
        "type": "http",
        "url": "http://127.0.0.1:9001/mcp",
    }
    assert env["PI_CODING_AGENT_DIR"] == "/tmp/vf-omp-agent-trace-123"


OMP_HISTORY_VERSIONS = [
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


@pytest.mark.parametrize("version", OMP_HISTORY_VERSIONS)
def test_requested_omp_versions_use_pinned_release_artifacts(version: str) -> None:
    script = omp_install_script(version)

    assert f"VERSION={version}" in script
    assert 'releases/download/v$VERSION/omp-linux-$arch' in script


@pytest.mark.parametrize(
    ("version", "supports_auto_approve"),
    [("15.4.3", False), ("15.12.4", True), ("17.2.10", True)],
)
def test_omp_launch_tracks_versioned_approval_flag(
    version: str, supports_auto_approve: bool
) -> None:
    runtime = FakeRuntime()
    harness = OmpAgentHarness(OmpAgentHarnessConfig(id="omp_agent", version=version))

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {},
        )
    )

    assert runtime.program is not None
    argv, _ = runtime.program
    assert ("--auto-approve" in argv) is supports_auto_approve


def test_hermes_launch_uses_isolated_config_and_wires_mcp() -> None:
    runtime = FakeRuntime()
    harness = HermesAgentHarness(HermesAgentHarnessConfig(id="hermes_agent"))

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {"task-tools": "http://127.0.0.1:9001/mcp"},
        )
    )

    assert runtime.program is not None
    argv, env = runtime.program
    config_path = "/tmp/vf-hermes-agent-trace-123/config.yaml"
    config = json.loads(runtime.writes[config_path])
    assert argv[0] == HERMES_BIN
    assert argv[:2] == [HERMES_BIN, "chat"]
    assert argv[argv.index("--provider") + 1] == "custom"
    assert argv[argv.index("--reasoning") + 1] == "medium"
    assert argv[argv.index("--toolsets") + 1] == "terminal,file,code_execution"
    assert argv[argv.index("--query") + 1] == "Fix the tests"
    assert "--quiet" in argv
    assert "--yolo" in argv
    assert "session-secret" not in argv
    assert config["model"] == {
        "api_key": "${VF_INTERCEPT_KEY}",
        "base_url": "http://127.0.0.1:9000/v1",
        "context_length": 128_000,
        "default": "~deepseek/deepseek-v4-flash-latest",
        "max_tokens": 32_000,
        "provider": "custom",
    }
    assert config["agent"]["system_prompt"] == "Be precise"
    assert config["agent"]["reasoning_effort"] == "medium"
    assert config["mcp_servers"]["task-tools"] == {
        "enabled": True,
        "url": "http://127.0.0.1:9001/mcp",
    }
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["OPENAI_API_KEY"] == "session-secret"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert env["HERMES_INFERENCE_PROVIDER"] == "custom"
    assert env["HERMES_HOME"] == "/tmp/vf-hermes-agent-trace-123"
    assert env["HERMES_SKIP_NODE_BOOTSTRAP"] == "1"
    assert env["HOME"] == "/tmp/vf-hermes-agent-trace-123/home"
    assert "session-secret" not in runtime.writes[config_path].decode()


@pytest.mark.parametrize(("version", "release_tag"), HERMES_RELEASE_TAGS.items())
def test_requested_hermes_versions_use_pinned_source_tags(
    version: str, release_tag: str
) -> None:
    script = hermes_install_script(version)

    assert f"VERSION={version}" in script
    assert f"RELEASE_TAG=v{release_tag}" in script
    assert "hermes-agent/$RELEASE_TAG/scripts/install.sh" in script
    assert '--branch "$RELEASE_TAG"' in script
    assert "git clone --depth 1 --branch" in script
    assert "skip optional system packages" in script
    assert "skip optional Node dependencies" in script
    assert "--skip-setup" in script
    assert ("--skip-browser" in script) == (
        tuple(map(int, version.split("."))) >= (0, 14, 0)
    )
    assert ("--no-skills" in script) == (
        tuple(map(int, version.split("."))) >= (0, 16, 0)
    )
    assert ("--non-interactive" in script) == (
        tuple(map(int, version.split("."))) >= (0, 16, 0)
    )
    assert 'current" = "$RELEASE_TAG"' in script
    assert "venv/bin/hermes" in script
    assert 'exec "$SOURCE/venv/bin/python" "$SOURCE/hermes"' in script


@pytest.mark.parametrize(
    ("version", "expected_flags"),
    [
        ("0.2.0", set()),
        ("0.8.0", {"--max-turns"}),
        ("0.12.0", {"--max-turns", "--provider"}),
        ("0.20.0", {"--max-turns", "--provider", "--reasoning"}),
    ],
)
def test_hermes_launch_tracks_versioned_cli_surface(
    version: str, expected_flags: set[str]
) -> None:
    runtime = FakeRuntime()
    harness = HermesAgentHarness(
        HermesAgentHarnessConfig(id="hermes_agent", version=version)
    )

    asyncio.run(
        harness.launch(
            context(),
            trace(),
            runtime,
            "http://127.0.0.1:9000/v1",
            "session-secret",
            {},
        )
    )

    assert runtime.program is not None
    argv, _ = runtime.program
    for flag in ("--max-turns", "--provider", "--reasoning"):
        assert (flag in argv) == (flag in expected_flags)
    config_path = "/tmp/vf-hermes-agent-trace-123/config.yaml"
    config = json.loads(runtime.writes[config_path])
    assert config["agent"]["max_turns"] == 90
    assert config["agent"]["reasoning_effort"] == "medium"
    legacy_config_path = (
        "/tmp/vf-hermes-agent-trace-123/home/.hermes/config.yaml"
    )
    assert (legacy_config_path in runtime.writes) == (version == "0.2.0")
