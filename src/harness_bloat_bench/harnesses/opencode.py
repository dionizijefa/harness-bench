"""OpenCode CLI harness using an isolated OpenAI-compatible provider."""

import asyncio
import json
import logging
import shlex

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from harness_bloat_bench.harness_cache import ensure_opencode_cached, stage_cached_file

from ._common import (
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
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
{shell_assignment("ARTIFACT_VERSION", artifact_version)}
BIN={shlex.quote(OPENCODE_BIN)}

current="$($BIN --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"$ARTIFACT_VERSION"|"v$ARTIFACT_VERSION") exit 0 ;;
    *) echo "cached OpenCode binary version mismatch: expected $VERSION, got $current" >&2; exit 1 ;;
esac
"""


class OpenCodeHarnessConfig(HarnessConfig):
    version: str = "1.18.1"
    """OpenCode release to install, pinned for reproducibility."""


class OpenCodeHarness(Harness[OpenCodeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info("opencode: loading cached OpenCode %s", self.config.version)
        musl_check = await runtime.run(
            [
                "sh",
                "-c",
                'for loader in /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do [ ! -e "$loader" ] || exit 0; done; exit 1',
            ],
            {},
        )
        musl = musl_check.exit_code == 0
        artifact_version = OPENCODE_ARTIFACT_ALIASES.get(
            release_version(self.config.version), release_version(self.config.version)
        )
        if musl and _version_tuple(artifact_version) < (1, 0, 0):
            raise RuntimeError(
                f"OpenCode {self.config.version} has no official musl artifact"
            )
        cached_binary = await asyncio.to_thread(
            ensure_opencode_cached,
            self.config.version,
            artifact_version,
            None,
            musl=musl,
        )
        await stage_cached_file(runtime, cached_binary, OPENCODE_BIN)
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

        version = _version_tuple(self.config.version)
        provider: dict = {
            "options": {
                "baseURL": endpoint,
                # Config-variable expansion did not exist in v0.1.x.
                "apiKey": (
                    secret
                    if _version_tuple(self.config.version) < (0, 2, 0)
                    else f"{{env:{INTERCEPT_KEY_VAR}}}"
                ),
            },
            "models": {INTERCEPT_MODEL: {}},
        }
        if version < (1, 0, 0):
            # Pre-v1 treats a missing package as the provider ID and attempts
            # to install `verifiers`; v1+ defaults to this transport itself.
            provider["npm"] = "@ai-sdk/openai-compatible"
        config: dict = {"provider": {INTERCEPT_PROVIDER: provider}}

        # Older releases have no headless approval flag. Their permission
        # entries are required to keep a non-interactive coding run moving.
        if (0, 3, 0) <= version < (1, 18, 0):
            config["permission"] = (
                {"*": "allow"}
                if version >= (1, 0, 0)
                else {
                    "edit": "allow",
                    "bash": "allow",
                    **({"webfetch": "allow"} if version >= (0, 5, 0) else {}),
                }
            )
        if self.config.disabled_tools and version >= (0, 3, 0):
            config["tools"] = {tool: False for tool in self.config.disabled_tools}
        if mcp_urls:
            config["mcp"] = {
                name: {"type": "remote", "url": url} for name, url in mcp_urls.items()
            }

        state_dir = f"/tmp/vf-opencode-state-{trace.id}"
        config_path = f"{state_dir}/xdg-config/opencode/config.json"
        await runtime.write(config_path, json.dumps(config).encode())
        env = {**self.config.resolved_env}
        if version >= (0, 2, 0):
            env[INTERCEPT_KEY_VAR] = secret
        if version < (0, 3, 0):
            env["XDG_CONFIG_HOME"] = f"{state_dir}/xdg-config"
        else:
            env["OPENCODE_CONFIG"] = config_path
        argv = [
            OPENCODE_BIN,
            "run",
            *(["--auto"] if version >= (1, 18, 0) else []),
            "--model",
            f"{INTERCEPT_PROVIDER}/{INTERCEPT_MODEL}",
            prompt,
        ]
        return await runtime.run_program(argv, env)
