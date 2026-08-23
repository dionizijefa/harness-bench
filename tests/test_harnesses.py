import asyncio
import json
from types import SimpleNamespace
from typing import get_args

import pytest
from verifiers.v1.loaders import harness_config_type, load_harness
from verifiers.v1.runtimes import ProgramResult

from harness_bloat_bench.definitions import (
    DEFAULT_HARNESS_VERSIONS,
    HarnessId,
)
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
    _versioned_argv,
)
from harness_bloat_bench.harnesses.codex_agent import (
    _install_script as codex_install_script,
)
from harness_bloat_bench.harnesses.hermes_agent import (
    HERMES_BIN,
    HERMES_PYTHON_DEPENDENCY_DIR,
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
from harness_bloat_bench.harnesses.prime_agent import (
    PRIME_AGENT_CLI,
    PRIME_AGENT_KERNEL_PYTHON,
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
        client=SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            api_key="openrouter-secret",
            headers={},
        ),
        sampling=SimpleNamespace(reasoning_effort=None),
    )


def test_every_declared_harness_has_a_default_version() -> None:
    expected = {
        "codex_agent",
        "claude_code_agent",
        "hermes_agent",
        "opencode",
        "omp_agent",
        "prime_agent",
    }
    assert set(get_args(HarnessId)) == expected
    assert set(DEFAULT_HARNESS_VERSIONS) == expected


@pytest.mark.parametrize("harness_id", get_args(HarnessId))
def test_every_harness_resolves_sets_up_and_runs(
    harness_id: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Exercise the production plugin boundary without network or model calls."""
    runtime = FakeRuntime()
    rollout_trace = trace()
    config_class = harness_config_type(harness_id)
    config = config_class.model_validate(
        {"id": harness_id, "version": DEFAULT_HARNESS_VERSIONS[harness_id]}
    )
    harness = load_harness(config)
    cached_file = tmp_path / "cached-binary"
    cached_file.write_bytes(b"cached harness binary")
    cached_tree = tmp_path / "cached-tree"
    cached_tree.mkdir()
    (cached_tree / "artifact").write_bytes(b"cached harness tree")
    cache_patches = {
        "claude_code_agent": (
            "harness_bloat_bench.harnesses.claude_code.ensure_claude_cached",
            cached_file,
        ),
        "codex_agent": (
            "harness_bloat_bench.harnesses.codex_agent.ensure_codex_cached",
            cached_file,
        ),
        "hermes_agent": (
            "harness_bloat_bench.harnesses.hermes_agent.ensure_docker_built_tree",
            cached_tree,
        ),
        "omp_agent": (
            "harness_bloat_bench.harnesses.omp_agent.ensure_omp_cached",
            cached_file,
        ),
        "opencode": (
            "harness_bloat_bench.harnesses.opencode.ensure_opencode_cached",
            cached_file,
        ),
        "prime_agent": (
            "harness_bloat_bench.harnesses.prime_agent.ensure_docker_built_tree",
            cached_tree,
        ),
    }
    target, cached = cache_patches[harness_id]
    monkeypatch.setattr(target, lambda *args, **kwargs: cached)
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

    setup_commands = "\n".join(" ".join(argv) for argv, _ in runtime.commands)
    for network_install_marker in (
        "curl ",
        "git clone",
        "npm install",
        "registry.npmjs.org",
        "downloads.claude.ai",
        "github.com/can1357/oh-my-pi/releases",
    ):
        assert network_install_marker not in setup_commands
    if harness_id in {"claude_code_agent", "opencode", "prime_agent"}:
        assert "chmod " not in setup_commands
    if harness_id == "prime_agent":
        assert "flock " not in setup_commands

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
    assert "github.com" not in script
    assert "curl" not in script
    assert 'current="$($BIN --version 2>/dev/null || true)"' in script
    assert '"codex-cli $VERSION"' in script


@pytest.mark.parametrize("version", CODEX_HISTORY_VERSIONS)
def test_codex_launch_leaves_feature_defaults_unset(version: str) -> None:
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
    assert "--disable" not in argv
    assert not any("multi_agent_v2" in arg for arg in argv)


def test_codex_harness_applies_versioned_launch_command() -> None:
    runtime = FakeRuntime()
    harness = CodexAgentHarness(CodexHarnessConfig(id="codex_agent", version="0.78.0"))

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
def test_requested_claude_code_versions_verify_cached_binary(
    version: str,
) -> None:
    script = claude_code_install_script(version)

    assert "claude.ai" not in script
    assert "curl" not in script
    assert version in script
    assert "chmod " not in script


def test_claude_code_routes_deepseek_through_anthropic_interception() -> None:
    runtime = FakeRuntime()
    ctx = context()
    harness = ClaudeCodeHarness(
        ClaudeCodeHarnessConfig(id="claude_code_agent", disabled_tools=["WebFetch"])
    )

    asyncio.run(
        harness.launch(
            ctx,
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
    for default_override in (
        "--bare",
        "--no-session-persistence",
        "--output-format",
        "--strict-mcp-config",
    ):
        assert default_override not in argv
    assert mcp["mcpServers"]["task-tools"] == {
        "type": "http",
        "url": "http://127.0.0.1:9001/mcp",
    }
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert env["ANTHROPIC_API_KEY"] == "session-secret"
    assert set(env) == {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "HOME",
        "IS_SANDBOX",
    }
    assert ctx.client.base_url == "https://openrouter.ai/api"
    assert ctx.client.headers["Authorization"] == "Bearer openrouter-secret"


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
    assert 'PATH="$NODE_DIR/bin:$PATH"' in script


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
    for default_override in ("--thinking", "--no-session", "--offline"):
        assert default_override not in argv
    assert argv[argv.index("--append-system-prompt") + 1] == "Be precise"
    assert argv[-3:] == ["--print", "--", "Fix the tests"]
    assert provider == {
        "api": "openai-completions",
        "apiKey": "VF_INTERCEPT_KEY",
        "baseUrl": "http://127.0.0.1:9000/v1",
        "models": [{"id": "model"}],
    }
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["PRIME_AGENT_CODING_AGENT_DIR"] == (
        "/tmp/vf-prime-agent-state-trace-123"
    )
    assert env["PRIME_AGENT_KERNEL_VENV"] == PRIME_AGENT_KERNEL_VENV
    assert env["PRIME_AGENT_KERNEL_PYTHON"] == PRIME_AGENT_KERNEL_PYTHON
    assert set(env) == {
        "PRIME_AGENT_CODING_AGENT_DIR",
        "PRIME_AGENT_KERNEL_PYTHON",
        "PRIME_AGENT_KERNEL_VENV",
        "VF_INTERCEPT_KEY",
    }


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
        "--auto",
        "--model",
        "verifiers/model",
        "Be precise\n\nFix the tests",
    ]
    assert config["provider"]["verifiers"]["options"]["baseURL"].endswith("/v1")
    assert config["provider"]["verifiers"]["options"]["apiKey"] == (
        "{env:VF_INTERCEPT_KEY}"
    )
    assert config["provider"]["verifiers"]["models"] == {"model": {}}
    assert "npm" not in config["provider"]["verifiers"]
    assert config["tools"] == {"websearch": False}
    assert config["mcp"]["task-tools"] == {
        "type": "remote",
        "url": "http://127.0.0.1:9001/mcp",
    }
    assert set(config).isdisjoint(
        {
            "$schema",
            "autoupdate",
            "enabled_providers",
            "model",
            "permission",
            "share",
            "small_model",
        }
    )
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["OPENCODE_CONFIG"] == config_path
    assert set(env) == {"OPENCODE_CONFIG", "VF_INTERCEPT_KEY"}
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
def test_requested_opencode_versions_verify_cached_artifacts(version: str) -> None:
    script = opencode_install_script(version)

    assert "registry.npmjs.org" not in script
    assert "curl" not in script
    assert f"VERSION={version}" in script
    if version == "0.1.196":
        assert "ARTIFACT_VERSION=0.1.195" in script
    else:
        assert f"ARTIFACT_VERSION={version}" in script
    assert "chmod " not in script


@pytest.mark.parametrize(
    ("version", "expected_permission", "expected_auto"),
    [
        ("0.1.196", None, False),
        ("0.3.133", {"edit": "allow", "bash": "allow"}, False),
        (
            "0.5.29",
            {"edit": "allow", "bash": "allow", "webfetch": "allow"},
            False,
        ),
        ("1.18.13", None, True),
    ],
)
def test_opencode_config_tracks_versioned_schema(
    version: str,
    expected_permission: dict[str, str] | None,
    expected_auto: bool,
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
    assert runtime.program is not None
    argv, env = runtime.program
    assert config.get("permission") == expected_permission
    assert ("--auto" in argv) is expected_auto
    assert ("npm" in config["provider"]["verifiers"]) is version.startswith("0.")
    assert "share" not in config
    assert "autoshare" not in config
    if version == "0.1.196":
        assert config["provider"]["verifiers"]["options"]["apiKey"] == (
            "session-secret"
        )
        assert "VF_INTERCEPT_KEY" not in env
        assert "OPENCODE_CONFIG" not in env
        assert env["XDG_CONFIG_HOME"].endswith("/xdg-config")
    else:
        assert config["provider"]["verifiers"]["options"]["apiKey"] == (
            "{env:VF_INTERCEPT_KEY}"
        )


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
    for default_override in ("--thinking", "--no-session", "--no-title"):
        assert default_override not in argv
    assert argv[-3:] == ["--print", "--", "Fix the tests"]
    assert "session-secret" not in argv
    provider = models["providers"]["verifiers"]
    assert provider == {
        "api": "openai-completions",
        "apiKey": "VF_INTERCEPT_KEY",
        "baseUrl": "http://127.0.0.1:9000/v1",
        "models": [{"id": "model"}],
    }
    assert mcp["mcpServers"]["task-tools"] == {
        "type": "http",
        "url": "http://127.0.0.1:9001/mcp",
    }
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["PI_CODING_AGENT_DIR"] == "/tmp/vf-omp-agent-trace-123"
    assert set(env) == {"PI_CODING_AGENT_DIR", "VF_INTERCEPT_KEY"}


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
    assert "github.com" not in script
    assert "curl" not in script
    assert "cached OMP binary version mismatch" in script


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
    for default_override in (
        "--max-turns",
        "--provider",
        "--quiet",
        "--reasoning",
        "--toolsets",
    ):
        assert default_override not in argv
    assert argv[argv.index("--query") + 1] == "Fix the tests"
    assert "--yolo" in argv
    assert "session-secret" not in argv
    assert config["model"] == {
        "api_key": "${VF_INTERCEPT_KEY}",
        "base_url": "http://127.0.0.1:9000/v1",
        "provider": "custom",
    }
    assert config["agent"] == {"system_prompt": "Be precise"}
    assert config["mcp_servers"]["task-tools"] == {"url": "http://127.0.0.1:9001/mcp"}
    assert set(config).isdisjoint({"approvals", "auxiliary", "display"})
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["HERMES_HOME"] == "/tmp/vf-hermes-agent-trace-123"
    assert env["HERMES_SKIP_NODE_BOOTSTRAP"] == "1"
    assert env["LD_LIBRARY_PATH"] == HERMES_PYTHON_DEPENDENCY_DIR
    assert set(env) == {
        "HERMES_HOME",
        "HERMES_SKIP_NODE_BOOTSTRAP",
        "LD_LIBRARY_PATH",
        "VF_INTERCEPT_KEY",
    }
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


@pytest.mark.parametrize("version", ["0.2.0", "0.8.0", "0.12.0", "0.20.0"])
def test_hermes_launch_leaves_behavior_defaults_unset(version: str) -> None:
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
    argv, env = runtime.program
    for flag in ("--max-turns", "--provider", "--reasoning"):
        assert flag not in argv
    config_path = "/tmp/vf-hermes-agent-trace-123/config.yaml"
    config = json.loads(runtime.writes[config_path])
    assert config["agent"] == {"system_prompt": "Be precise"}
    if version == "0.2.0":
        assert "model" not in config
    else:
        assert config["model"] == {
            "api_key": "${VF_INTERCEPT_KEY}",
            "base_url": "http://127.0.0.1:9000/v1",
            "provider": "custom",
        }
    legacy_config_path = "/tmp/vf-hermes-agent-trace-123/home/.hermes/config.yaml"
    assert (legacy_config_path in runtime.writes) == (version == "0.2.0")
    for key in (
        "HOME",
        "HERMES_INFERENCE_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        assert (key in env) == (version == "0.2.0")
    assert ("VF_INTERCEPT_KEY" in env) == (version != "0.2.0")
