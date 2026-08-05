import asyncio
import json
from types import SimpleNamespace
from typing import get_args

import pytest
from verifiers.v1.loaders import harness_config_type, load_harness
from verifiers.v1.runtimes import ProgramResult

from harness_bloat_bench.definitions import DEFAULT_HARNESS_VERSIONS, HarnessId
from harness_bloat_bench.harnesses.hermes_agent import (
    HERMES_BIN,
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
from harness_bloat_bench.harnesses.opencode import (
    OPENCODE_BIN,
    OpenCodeHarness,
    OpenCodeHarnessConfig,
)
from harness_bloat_bench.harnesses.opencode import (
    _install_script as opencode_install_script,
)
from harness_bloat_bench.harnesses.pi import PI_BIN, PiHarness, PiHarnessConfig


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
        "--",
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
    assert config["mcp_servers"]["task-tools"] == {
        "enabled": True,
        "url": "http://127.0.0.1:9001/mcp",
    }
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert env["HERMES_HOME"] == "/tmp/vf-hermes-agent-trace-123"
    assert "session-secret" not in runtime.writes[config_path].decode()


def test_hermes_install_is_pinned_and_noninteractive() -> None:
    script = hermes_install_script("v2026.8.3")

    assert "hermes-agent/v$VERSION/scripts/install.sh" in script
    assert '--branch "v$VERSION"' in script
    assert "--skip-setup" in script
    assert "--skip-browser" in script
    assert "--no-skills" in script
    assert "--non-interactive" in script
