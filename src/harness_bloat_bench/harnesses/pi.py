"""Pi coding-agent harness with a reproducible, practical coding toolset."""

import logging
import shlex
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

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
DIR={shlex.quote(PI_DIR)}
BIN={shlex.quote(PI_BIN)}

current="$($BIN --version 2>/dev/null || true)"
if [ -x "$BIN" ] && {{ [ "$current" = "$VERSION" ] || [ "$current" = "v$VERSION" ]; }}; then
    exit 0
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get -o Acquire::Retries=3 update -qq
        apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates tar >/dev/null
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl ca-certificates tar >/dev/null
    else
        echo "Pi needs curl, CA certificates, and tar" >&2
        exit 1
    fi
fi

case "$(uname -m)" in
    aarch64|arm64) arch=arm64 ;;
    x86_64|amd64) arch=x64 ;;
    *) echo "unsupported Pi architecture: $(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$DIR"
archive="$DIR/pi-linux-$arch.tar.gz.tmp"
stage="$DIR/stage"
trap 'rm -f "$archive"; rm -rf "$stage"' EXIT
curl -fsSL "https://github.com/earendil-works/pi-mono/releases/download/v$VERSION/pi-linux-$arch.tar.gz" -o "$archive"
rm -rf "$stage"
mkdir -p "$stage"
tar -xzf "$archive" --strip-components=1 -C "$stage"
rm -rf "$DIR/current"
mv "$stage" "$DIR/current"
chmod 755 "$BIN"
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
        logger.info("pi: ensuring Pi %s is installed", self.config.version)
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
