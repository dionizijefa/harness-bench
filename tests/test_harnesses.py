import asyncio
import json
from types import SimpleNamespace

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
from harness_bloat_bench.harnesses.pi import PI_BIN, PiHarness, PiHarnessConfig
from verifiers.v1.runtimes import ProgramResult


class FakeRuntime:
    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}
        self.program: tuple[list[str], dict[str, str]] | None = None

    async def write(self, path: str, data: bytes) -> None:
        self.writes[path] = data

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.program = (argv, env)
        return ProgramResult(exit_code=0, stdout="", stderr="")


def trace(prompt: str = "Fix the tests", system_prompt: str = "Be precise"):
    data = SimpleNamespace(prompt=prompt, system_prompt=system_prompt)
    return SimpleNamespace(id="trace-123", task=SimpleNamespace(data=data))


def context():
    return SimpleNamespace(model="~deepseek/deepseek-v4-flash-latest")


def test_opencode_launch_uses_isolated_inline_config() -> None:
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
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert argv[0] == OPENCODE_BIN
    assert argv[-1] == "Be precise\n\nFix the tests"
    assert "--auto" in argv
    assert config["provider"]["verifiers"]["options"]["baseURL"].endswith("/v1")
    assert config["provider"]["verifiers"]["options"]["apiKey"] == (
        "{env:VF_INTERCEPT_KEY}"
    )
    assert config["tools"] == {"websearch": False}
    assert config["mcp"]["task-tools"]["type"] == "remote"
    assert env["VF_INTERCEPT_KEY"] == "session-secret"
    assert "session-secret" not in env["OPENCODE_CONFIG_CONTENT"]


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
