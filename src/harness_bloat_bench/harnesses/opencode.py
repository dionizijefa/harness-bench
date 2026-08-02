"""OpenCode CLI harness using an isolated OpenAI-compatible provider."""

import json
import logging
import shlex

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
    release_version,
    run_install,
    shell_assignment,
)

logger = logging.getLogger(__name__)

OPENCODE_DIR = "/tmp/vf-opencode"
OPENCODE_BIN = f"{OPENCODE_DIR}/bin/opencode"


def _install_script(version: str) -> str:
    version = release_version(version)
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
DIR={shlex.quote(OPENCODE_DIR)}
BIN={shlex.quote(OPENCODE_BIN)}

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
        echo "OpenCode needs curl, CA certificates, and tar" >&2
        exit 1
    fi
fi

case "$(uname -m)" in
    aarch64|arm64) stem=opencode-linux-arm64 ;;
    x86_64|amd64) stem=opencode-linux-x64-baseline ;;
    *) echo "unsupported OpenCode architecture: $(uname -m)" >&2; exit 1 ;;
esac

# OpenCode publishes dynamically linked glibc and musl builds. Selecting the
# wrong loader looks like a missing binary even when the archive extracted.
libc_suffix=
for loader in /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do
    if [ -e "$loader" ]; then
        libc_suffix=-musl
        break
    fi
done
asset="$stem$libc_suffix.tar.gz"

mkdir -p "$DIR/bin"
archive="$DIR/$asset.tmp"
trap 'rm -f "$archive"' EXIT
curl -fsSL "https://github.com/anomalyco/opencode/releases/download/v$VERSION/$asset" -o "$archive"
rm -f "$BIN"
tar -xzf "$archive" -C "$DIR/bin"
chmod 755 "$BIN"
"""


class OpenCodeHarnessConfig(HarnessConfig):
    version: str = "1.18.1"
    """OpenCode release to install, pinned for reproducibility."""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    provider_timeout_ms: int = 3_600_000


class OpenCodeHarness(Harness[OpenCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info("opencode: ensuring OpenCode %s is installed", self.config.version)
        await run_install(
            runtime,
            "OpenCode",
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
        _, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("OpenCode requires a task prompt")

        model = {
            "name": ctx.model,
            "limit": {
                "context": self.config.context_window,
                "output": self.config.max_tokens,
            },
        }
        config: dict = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "enabled_providers": [INTERCEPT_PROVIDER],
            "model": f"{INTERCEPT_PROVIDER}/{INTERCEPT_MODEL}",
            # Prevent auxiliary work from silently selecting a catalog model.
            "small_model": f"{INTERCEPT_PROVIDER}/{INTERCEPT_MODEL}",
            "provider": {
                INTERCEPT_PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Verifiers interception",
                    "options": {
                        "baseURL": endpoint,
                        "apiKey": f"{{env:{INTERCEPT_KEY_VAR}}}",
                        "timeout": self.config.provider_timeout_ms,
                    },
                    "models": {INTERCEPT_MODEL: model},
                }
            },
            "permission": {"*": "allow"},
        }
        if self.config.disabled_tools:
            config["tools"] = {tool: False for tool in self.config.disabled_tools}
        if mcp_urls:
            config["mcp"] = {
                name: {"type": "remote", "url": url, "enabled": True}
                for name, url in mcp_urls.items()
            }

        state_dir = f"/tmp/vf-opencode-state-{trace.id}"
        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "NO_COLOR": "1",
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            "OPENCODE_CONFIG_DIR": f"{state_dir}/config",
            "XDG_CACHE_HOME": f"{state_dir}/cache",
            "XDG_CONFIG_HOME": f"{state_dir}/xdg-config",
            "XDG_DATA_HOME": f"{state_dir}/data",
            "XDG_STATE_HOME": f"{state_dir}/state",
        }
        argv = [
            OPENCODE_BIN,
            "run",
            "--model",
            f"{INTERCEPT_PROVIDER}/{INTERCEPT_MODEL}",
            "--agent",
            "build",
            "--auto",
            "--title",
            "Verifiers rollout",
            "--format",
            "json",
            "--",
            prompt,
        ]
        return await runtime.run_program(argv, env)
