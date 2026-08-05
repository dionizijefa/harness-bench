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

# v0.1.196 and v0.1.195 point at the same upstream commit, but the v0.1.196
# publish job did not produce GitHub or npm artifacts. Use the identical
# v0.1.195 build while retaining v0.1.196 as the configured benchmark version.
OPENCODE_ARTIFACT_ALIASES = {"0.1.196": "0.1.195"}


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = release_version(version).split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"OpenCode version must be numeric semver: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _install_script(version: str) -> str:
    version = release_version(version)
    artifact_version = OPENCODE_ARTIFACT_ALIASES.get(version, version)
    supports_musl = _version_tuple(artifact_version) >= (1, 0, 0)
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
{shell_assignment("ARTIFACT_VERSION", artifact_version)}
{shell_assignment("SUPPORTS_MUSL", "1" if supports_musl else "0")}
DIR={shlex.quote(OPENCODE_DIR)}
BIN={shlex.quote(OPENCODE_BIN)}

current="$($BIN --version 2>/dev/null || true)"
if [ -x "$BIN" ] && {{ [ "$current" = "$VERSION" ] || [ "$current" = "v$VERSION" ] || [ "$current" = "$ARTIFACT_VERSION" ] || [ "$current" = "v$ARTIFACT_VERSION" ]; }}; then
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

musl=0
for loader in /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do
    if [ -e "$loader" ]; then
        musl=1
        break
    fi
done
if [ "$musl" = 1 ] && [ "$SUPPORTS_MUSL" = 0 ]; then
    echo "OpenCode $VERSION has no official musl artifact" >&2
    exit 1
fi

case "$(uname -m)" in
    aarch64|arm64)
        package=opencode-linux-arm64
        [ "$musl" = 0 ] || package="$package-musl"
        ;;
    x86_64|amd64)
        if [ "$ARTIFACT_VERSION" = "0.1.195" ]; then
            package=opencode-linux-x64
        else
            package=opencode-linux-x64-baseline
        fi
        [ "$musl" = 0 ] || package="$package-musl"
        ;;
    *) echo "unsupported OpenCode architecture: $(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$DIR/bin"
archive="$DIR/$package.tgz.tmp"
unpack="$DIR/unpack"
trap 'rm -rf "$archive" "$unpack"' EXIT
curl -fsSL "https://registry.npmjs.org/$package/-/$package-$ARTIFACT_VERSION.tgz" -o "$archive"
rm -f "$BIN"
rm -rf "$unpack"
mkdir -p "$unpack"
tar -xzf "$archive" -C "$unpack"
mv "$unpack/package/bin/opencode" "$BIN"
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
            "model": f"{INTERCEPT_PROVIDER}/{INTERCEPT_MODEL}",
            "provider": {
                INTERCEPT_PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Verifiers interception",
                    "options": {
                        "baseURL": endpoint,
                        # Config-variable expansion did not exist in v0.1.x.
                        "apiKey": (
                            secret
                            if _version_tuple(self.config.version) < (0, 2, 0)
                            else f"{{env:{INTERCEPT_KEY_VAR}}}"
                        ),
                        "timeout": self.config.provider_timeout_ms,
                    },
                    "models": {INTERCEPT_MODEL: model},
                }
            },
        }

        version = _version_tuple(self.config.version)
        if version < (0, 3, 0):
            config["autoshare"] = False
        else:
            config["share"] = "disabled"
            # Prevent auxiliary work from silently selecting a catalog model.
            config["small_model"] = f"{INTERCEPT_PROVIDER}/{INTERCEPT_MODEL}"
            config["permission"] = (
                {"*": "allow"}
                if version >= (1, 0, 0)
                else {
                    "edit": "allow",
                    "bash": "allow",
                    **({"webfetch": "allow"} if version >= (0, 5, 0) else {}),
                }
            )
        if version >= (1, 0, 0):
            config["enabled_providers"] = [INTERCEPT_PROVIDER]
        if self.config.disabled_tools and version >= (0, 3, 0):
            config["tools"] = {tool: False for tool in self.config.disabled_tools}
        if mcp_urls:
            config["mcp"] = {
                name: {"type": "remote", "url": url, "enabled": True}
                for name, url in mcp_urls.items()
            }

        state_dir = f"/tmp/vf-opencode-state-{trace.id}"
        config_path = f"{state_dir}/xdg-config/opencode/config.json"
        await runtime.write(config_path, json.dumps(config).encode())
        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "NO_COLOR": "1",
            # OPENCODE_CONFIG is supported by v0.3+; v0.1.x reads this same
            # path as its XDG global config.
            "OPENCODE_CONFIG": config_path,
            "OPENCODE_CONFIG_DIR": f"{state_dir}/xdg-config/opencode",
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
            "--",
            prompt,
        ]
        return await runtime.run_program(argv, env)
