"""Claude Code harness routed through the Anthropic Messages interceptor."""

import asyncio
import logging
import shlex
from typing import cast
from urllib.parse import urlparse

import verifiers.v1.harnesses.claude_code.harness as stock_claude
from pydantic import Field
from verifiers.v1.clients import EvalClient, ModelContext
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from harness_bloat_bench.harness_cache import ensure_claude_cached, stage_cached_file

from ._common import json_bytes, release_version, run_install

logger = logging.getLogger(__name__)


def _version_tuple(version: str) -> tuple[int, ...]:
    normalized = release_version(version)
    try:
        return tuple(int(part) for part in normalized.split("."))
    except ValueError as error:
        raise ValueError(f"Claude Code version must be numeric: {version}") from error


def _claude_bin(version: str) -> str:
    return stock_claude.CLAUDE_BIN.format(version=release_version(version))


def _install_script(version: str) -> str:
    version = release_version(version)
    _version_tuple(version)
    binary = _claude_bin(version)
    return f"""\
set -eu
current="$({shlex.quote(binary)} --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"$VERSION "*|*" $VERSION") exit 0 ;;
    *) echo "cached Claude Code binary version mismatch: expected {version}, got $current" >&2; exit 1 ;;
esac
""".replace("$VERSION", version)


def _configure_anthropic_upstream(client: EvalClient) -> None:
    """Adapt an OpenAI-style client root for the Anthropic Messages dialect."""
    # EvalClient appends the dialect path (/v1/messages). Matrix endpoints normally
    # include /v1 for OpenAI harnesses, so remove it before Claude makes a request.
    client.base_url = client.base_url.removesuffix("/v1")
    host = urlparse(client.base_url).hostname or ""
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        # OpenRouter's Anthropic Messages skin authenticates with Bearer. Keep the
        # dialect's x-api-key too; both carry the same key and Claude-compatible
        # providers that accept x-api-key continue to work unchanged.
        client.headers["Authorization"] = f"Bearer {client.api_key}"


class ClaudeCodeHarnessConfig(stock_claude.ClaudeCodeHarnessConfig):
    version: str = Field(default="2.1.226", pattern=r"^[A-Za-z0-9._+-]+$")
    """Claude Code release to install, pinned for reproducibility."""


class ClaudeCodeHarness(Harness[ClaudeCodeHarnessConfig]):
    """Add pinned setup, isolated state, and benchmark-wide custom model routing."""

    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_MESSAGE_PROMPT = False

    async def setup(self, runtime: Runtime) -> None:
        version = release_version(self.config.version)
        home = stock_claude.CLAUDE_HOME.format(version=version)
        binary = _claude_bin(version)
        cached_binary = await asyncio.to_thread(ensure_claude_cached, version)
        await stage_cached_file(runtime, cached_binary, binary)
        install = shlex.quote(_install_script(version))
        guarded = (
            f"mkdir -p {shlex.quote(home)} && "
            f'"$(command -v flock || command -v lockf)" '
            f"{shlex.quote(f'{home}/install.lock')} "
            f"bash -o pipefail -c {install}"
        )
        logger.info("claude-code: ensuring Claude Code %s is installed", version)
        await run_install(runtime, "Claude Code", version, guarded)

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
            raise ValueError("Claude Code requires a task prompt")

        _configure_anthropic_upstream(cast(EvalClient, ctx.client))

        state_dir = f"/tmp/vf-claude-code-state-{trace.id}"
        argv = [
            _claude_bin(self.config.version),
            "--print",
            "--bare",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--output-format",
            "text",
            "--model",
            ctx.model,
        ]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        argv += [
            arg
            for tool in self.config.disabled_tools or []
            for arg in ("--disallowedTools", tool)
        ]
        if mcp_urls:
            mcp_path = f"{state_dir}/mcp.json"
            mcp = {
                "mcpServers": {
                    name: {"type": "http", "url": url}
                    for name, url in mcp_urls.items()
                }
            }
            await runtime.write(mcp_path, json_bytes(mcp))
            argv += ["--mcp-config", mcp_path, "--strict-mcp-config"]
        argv += ["--", prompt]

        model = ctx.model
        env = {
            **self.config.resolved_env,
            # AUTH_TOKEN makes Claude send Bearer auth to interception. Explicitly
            # clear API_KEY so no cached Anthropic login mode can take precedence.
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": secret,
            # Claude appends /v1/messages; interception's endpoint already ends in /v1.
            "ANTHROPIC_BASE_URL": endpoint.removesuffix("/v1"),
            "ANTHROPIC_CUSTOM_MODEL_OPTION": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_MODEL": model,
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_UPDATES": "1",
            "HOME": f"{state_dir}/home",
            "IS_SANDBOX": "1",
            "NO_COLOR": "1",
        }
        return await runtime.run_program(argv, env)


__all__ = ["ClaudeCodeHarness", "ClaudeCodeHarnessConfig"]
