"""Pi coding-agent harness with a reproducible, practical coding toolset."""

import asyncio
import logging
import shlex
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from harness_bloat_bench.harness_cache import ensure_pi_cached, stage_cached_tree

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

PI_DIR = "/tmp/vf-pi"
PI_BIN = f"{PI_DIR}/current/pi"
PI_CODING_TOOLS = ["read", "bash", "edit", "write", "grep", "find", "ls"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def _install_script(version: str) -> str:
    version = release_version(version)
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
BIN={shlex.quote(PI_BIN)}

chmod 755 "$BIN"
current="$($BIN --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION") exit 0 ;;
    *) echo "cached Pi binary version mismatch: expected $VERSION, got $current" >&2; exit 1 ;;
esac
"""


class PiHarnessConfig(HarnessConfig):
    version: str = "0.80.7"
    """Pi coding-agent release to install, pinned for reproducibility."""

    # Pi enables read/bash/edit/write by default. The three official read-only
    # navigation tools make a materially better out-of-the-box coding harness.
    tools: list[str] = PI_CODING_TOOLS
    thinking: ThinkingLevel | None = "medium"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS


class PiHarness(Harness[PiHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        logger.info("pi: loading cached Pi %s", self.config.version)
        cached_tree = await asyncio.to_thread(ensure_pi_cached, self.config.version)
        await stage_cached_tree(runtime, cached_tree, f"{PI_DIR}/current")
        await run_install(
            runtime, "Pi", self.config.version, _install_script(self.config.version)
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
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("Pi requires a task prompt")

        agent_dir = f"/tmp/vf-pi-agent-{trace.id}"
        models = openai_compat_model_config(
            endpoint=endpoint,
            model_name=ctx.model,
            context_window=self.config.context_window,
            max_tokens=self.config.max_tokens,
            api_key=f"${INTERCEPT_KEY_VAR}",
        )
        await runtime.write(f"{agent_dir}/models.json", json_bytes(models))

        tools = [
            tool
            for tool in self.config.tools
            if tool not in set(self.config.disabled_tools or [])
        ]
        argv = [
            PI_BIN,
            "--no-session",
            "--offline",
            "--approve",
            "--provider",
            INTERCEPT_PROVIDER,
            "--model",
            INTERCEPT_MODEL,
        ]
        argv += ["--tools", ",".join(tools)] if tools else ["--no-tools"]
        if self.config.disabled_tools:
            argv += ["--exclude-tools", ",".join(self.config.disabled_tools)]
        if self.config.thinking is not None:
            argv += ["--thinking", self.config.thinking]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        argv += ["--print", prompt]

        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "NO_COLOR": "1",
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_CODING_AGENT_SESSION_DIR": f"{agent_dir}/sessions",
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }
        return await runtime.run_program(argv, env)
