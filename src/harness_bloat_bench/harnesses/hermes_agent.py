"""Nous Research Hermes Agent harness."""

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
    json_bytes,
    release_version,
    run_install,
    shell_assignment,
)

logger = logging.getLogger(__name__)

HERMES_DIR = "/tmp/vf-hermes"
HERMES_SOURCE_DIR = f"{HERMES_DIR}/source"
HERMES_BIN = f"{HERMES_SOURCE_DIR}/venv/bin/hermes"
ThinkingLevel = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
]


def _install_script(version: str) -> str:
    version = release_version(version)
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
DIR={shlex.quote(HERMES_DIR)}
SOURCE={shlex.quote(HERMES_SOURCE_DIR)}
BIN={shlex.quote(HERMES_BIN)}

current="$(git -C "$SOURCE" describe --tags --exact-match 2>/dev/null || true)"
if [ -x "$BIN" ] && [ "$current" = "v$VERSION" ]; then
    exit 0
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v bash >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get -o Acquire::Retries=3 update -qq
        apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates git bash >/dev/null
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl ca-certificates git bash >/dev/null
    else
        echo "Hermes Agent needs curl, Git, Bash, and CA certificates" >&2
        exit 1
    fi
fi

rm -rf "$SOURCE"
mkdir -p "$DIR"
installer="$DIR/install.sh"
trap 'rm -f "$installer"' EXIT
curl -fsSL \
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/v$VERSION/scripts/install.sh" \
    -o "$installer"
HERMES_HOME="$DIR/install-state" HERMES_INSTALL_DIR="$SOURCE" \
    bash "$installer" \
        --branch "v$VERSION" \
        --skip-setup \
        --skip-browser \
        --no-skills \
        --non-interactive

[ -x "$BIN" ]
"""


class HermesAgentHarnessConfig(HarnessConfig):
    version: str = "2026.8.3"
    """Hermes Agent release tag to install, pinned for reproducibility."""

    thinking: ThinkingLevel | None = "medium"
    max_turns: int = 90
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS


class HermesAgentHarness(Harness[HermesAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "hermes-agent: ensuring Hermes Agent %s is installed",
            self.config.version,
        )
        await run_install(
            runtime,
            "Hermes Agent",
            self.config.version,
            _install_script(self.config.version),
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
            raise ValueError(
                "Hermes Agent supports disabling toolsets, not arbitrary tools"
            )
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("Hermes Agent requires a task prompt")

        state_dir = f"/tmp/vf-hermes-agent-{trace.id}"
        config: dict = {
            "model": {
                "default": ctx.model,
                "provider": "custom",
                "base_url": endpoint,
                # Hermes expands ${...} references when loading YAML. This
                # supplies auth to loopback interception without persisting it.
                "api_key": f"${{{INTERCEPT_KEY_VAR}}}",
                "context_length": self.config.context_window,
                "max_tokens": self.config.max_tokens,
            },
            "agent": {
                "max_turns": self.config.max_turns,
                "system_prompt": system_prompt or "",
            },
            "approvals": {"mode": "off"},
            "display": {
                "compact": True,
                "show_reasoning": False,
                "streaming": False,
            },
            # A one-task rollout does not need a second model call for a title.
            "auxiliary": {"title_generation": {"enabled": False}},
        }
        if mcp_urls:
            config["mcp_servers"] = {
                name: {"url": url, "enabled": True}
                for name, url in mcp_urls.items()
            }
        await runtime.write(f"{state_dir}/config.yaml", json_bytes(config))

        argv = [
            HERMES_BIN,
            "chat",
            "--model",
            ctx.model,
            "--provider",
            "custom",
            "--max-turns",
            str(self.config.max_turns),
            "--quiet",
            "--yolo",
        ]
        if self.config.thinking is not None:
            argv += ["--reasoning", self.config.thinking]
        argv += ["--query", prompt]

        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "HERMES_HOME": state_dir,
            "NO_COLOR": "1",
        }
        return await runtime.run_program(argv, env)


__all__ = ["HermesAgentHarness", "HermesAgentHarnessConfig"]
