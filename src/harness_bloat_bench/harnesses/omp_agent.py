"""Oh My Pi (OMP) coding-agent harness."""

import asyncio
import logging
import shlex
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from harness_bloat_bench.harness_cache import ensure_omp_cached, stage_cached_file

from ._common import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
    INTERCEPT_KEY_VAR,
    INTERCEPT_MODEL,
    INTERCEPT_PROVIDER,
    json_bytes,
    openai_compat_model_config,
    release_version,
    run_install,
    shell_assignment,
)

logger = logging.getLogger(__name__)

OMP_DIR = "/tmp/vf-omp"
OMP_BIN = f"{OMP_DIR}/bin/omp"
ThinkingLevel = Literal[
    "off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"
]


def _version_tuple(version: str) -> tuple[int, ...]:
    normalized = release_version(version)
    try:
        return tuple(int(part) for part in normalized.split("."))
    except ValueError as error:
        raise ValueError(f"OMP version must be numeric: {version}") from error


def _install_script(version: str) -> str:
    version = release_version(version)
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
BIN={shlex.quote(OMP_BIN)}

chmod 755 "$BIN"
current="$($BIN --version 2>&1 || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"omp/$VERSION"|*" $VERSION"*) exit 0 ;;
    *) echo "cached OMP binary version mismatch: expected $VERSION, got $current" >&2; exit 1 ;;
esac
"""


class OmpAgentHarnessConfig(HarnessConfig):
    version: str = "17.2.10"
    """Oh My Pi release to install, pinned for reproducibility."""

    thinking: ThinkingLevel | None = "high"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS


class OmpAgentHarness(Harness[OmpAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info("omp-agent: loading cached OMP %s", self.config.version)
        cached_binary = await asyncio.to_thread(ensure_omp_cached, self.config.version)
        await stage_cached_file(runtime, cached_binary, OMP_BIN)
        await run_install(
            runtime, "OMP", self.config.version, _install_script(self.config.version)
        )

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError("OMP does not support disabling arbitrary tools")
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("OMP requires a task prompt")

        agent_dir = f"/tmp/vf-omp-agent-{trace.id}"
        models = openai_compat_model_config(
            endpoint=endpoint,
            model_name=ctx.model,
            context_window=self.config.context_window,
            max_tokens=self.config.max_tokens,
            # OMP treats this as an environment-variable name before a literal.
            api_key=INTERCEPT_KEY_VAR,
        )
        # JSON is valid YAML, while avoiding a new YAML dependency in this adapter.
        await runtime.write(f"{agent_dir}/models.yml", json_bytes(models))
        if mcp_urls:
            mcp = {
                "mcpServers": {
                    name: {"type": "http", "url": url, "enabled": True}
                    for name, url in mcp_urls.items()
                }
            }
            await runtime.write(f"{agent_dir}/mcp.json", json_bytes(mcp))

        argv = [
            OMP_BIN,
            "--no-session",
            "--no-title",
            "--provider",
            INTERCEPT_PROVIDER,
            "--model",
            INTERCEPT_MODEL,
        ]
        if _version_tuple(self.config.version) >= (15, 12, 4):
            argv.append("--auto-approve")
        if self.config.thinking is not None:
            argv += ["--thinking", self.config.thinking]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        argv += ["--print", "--", prompt]

        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "NO_COLOR": "1",
            "PI_CODING_AGENT_DIR": agent_dir,
        }
        return await runtime.run_program(argv, env)
