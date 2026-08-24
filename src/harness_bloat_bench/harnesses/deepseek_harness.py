"""DeepSeek Harness headless coding-agent adapter."""

import asyncio
import logging
import shlex

from pydantic import Field
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from harness_bloat_bench.harness_cache import (
    ensure_docker_built_tree,
    stage_cached_tree,
)

from ._common import (
    INTERCEPT_KEY_VAR,
    INTERCEPT_MODEL,
    INTERCEPT_PROVIDER,
    json_bytes,
    node_install_script,
    release_version,
    run_install,
    runtime_linux_arch,
    shell_assignment,
)

logger = logging.getLogger(__name__)

DEEPSEEK_HARNESS_DIR = "/tmp/vf-deepseek-harness"
DEEPSEEK_HARNESS_NODE_DIR = f"{DEEPSEEK_HARNESS_DIR}/node"
DEEPSEEK_HARNESS_PREFIX = f"{DEEPSEEK_HARNESS_DIR}/install"
DEEPSEEK_HARNESS_NODE_BIN = f"{DEEPSEEK_HARNESS_NODE_DIR}/bin/node"
DEEPSEEK_HARNESS_NPM_BIN = f"{DEEPSEEK_HARNESS_NODE_DIR}/bin/npm"
DEEPSEEK_HARNESS_CLI = (
    f"{DEEPSEEK_HARNESS_PREFIX}/lib/node_modules/@deepseek-ai/dsh/lib/bin.js"
)


def _install_script(version: str) -> str:
    version = release_version(version)
    package = f"@deepseek-ai/dsh@{version}"
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
DIR={shlex.quote(DEEPSEEK_HARNESS_DIR)}
PREFIX={shlex.quote(DEEPSEEK_HARNESS_PREFIX)}
NODE={shlex.quote(DEEPSEEK_HARNESS_NODE_BIN)}
NPM={shlex.quote(DEEPSEEK_HARNESS_NPM_BIN)}
CLI={shlex.quote(DEEPSEEK_HARNESS_CLI)}

{node_install_script(DEEPSEEK_HARNESS_NODE_DIR)}

current="$($NODE $CLI --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"dsh $VERSION") exit 0 ;;
esac

mkdir -p "$DIR/build-home" "$PREFIX"
PATH="$NODE_DIR/bin:$PATH" \
HOME="$DIR/build-home" \
"$NPM" install -g --prefix "$PREFIX" --no-fund --no-audit --loglevel=error --progress=false {shlex.quote(package)}

test -f "$CLI"
current="$($NODE $CLI --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"dsh $VERSION") ;;
    *) echo "DeepSeek Harness version mismatch: expected $VERSION, got $current" >&2; exit 1 ;;
esac
"""


class DeepSeekHarnessConfig(HarnessConfig):
    version: str = Field(default="0.1.1-rc.2", pattern=r"^[A-Za-z0-9._+-]+$")
    """DeepSeek Harness npm release, pinned for reproducibility."""


class DeepSeekHarness(Harness[DeepSeekHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "deepseek-harness: loading DeepSeek Harness %s", self.config.version
        )
        arch = await runtime_linux_arch(runtime)
        cached_tree = await asyncio.to_thread(
            ensure_docker_built_tree,
            harness="deepseek-harness",
            version=self.config.version,
            source_dir=DEEPSEEK_HARNESS_DIR,
            install_script=_install_script(self.config.version),
            arch=arch,
            bundle_python_runtime=False,
        )
        await stage_cached_tree(runtime, cached_tree, DEEPSEEK_HARNESS_DIR)
        await run_install(
            runtime,
            "DeepSeek Harness",
            self.config.version,
            f"""\
set -eu
test -x {shlex.quote(DEEPSEEK_HARNESS_NODE_BIN)}
test -f {shlex.quote(DEEPSEEK_HARNESS_CLI)}
{shlex.quote(DEEPSEEK_HARNESS_NODE_BIN)} {shlex.quote(DEEPSEEK_HARNESS_CLI)} --version >/dev/null
""",
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
                "DeepSeek Harness does not expose a stable arbitrary-tool denylist"
            )
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("DeepSeek Harness requires a task prompt")

        state_dir = f"/tmp/vf-deepseek-harness-state-{trace.id}"
        patch_path = f"{state_dir}/verifiers.patch.json"
        patches: list[dict] = [
            {
                "id": "llm-pi-ai",
                "config": {
                    "providers": {
                        INTERCEPT_PROVIDER: {
                            "apiKeyEnv": INTERCEPT_KEY_VAR,
                            "api": "openai-completions",
                            "baseURL": endpoint,
                            "compat": {
                                "supportsDeveloperRole": False,
                                "maxTokensField": "max_tokens",
                            },
                            "models": [{"id": INTERCEPT_MODEL}],
                        }
                    }
                },
            },
            {
                "id": "agent-default-model",
                "config": {
                    "provider": INTERCEPT_PROVIDER,
                    "model": INTERCEPT_MODEL,
                },
            },
        ]
        if system_prompt:
            patches.append(
                {
                    "id": "system-prompt",
                    "config": {
                        "persona": (
                            "You are a coding agent powered by the {{model}} model. "
                            "Your working directory is {{cwd}}.\n\n"
                            f"{system_prompt}"
                        )
                    },
                }
            )
        if mcp_urls:
            patches.append(
                {
                    "insert": [
                        {
                            "id": f"verifiers-mcp-{index}",
                            "name": "@deepseek-ai/dsh-mcp-client",
                            "config": {
                                "serverName": name,
                                "transport": "streamable-http",
                                "url": url,
                                "failOnStartupError": True,
                            },
                        }
                        for index, (name, url) in enumerate(mcp_urls.items(), start=1)
                    ]
                }
            )
        await runtime.write(patch_path, json_bytes(patches))

        argv = [
            DEEPSEEK_HARNESS_NODE_BIN,
            DEEPSEEK_HARNESS_CLI,
            "--profile",
            "headless",
            "--patch",
            patch_path,
            prompt,
        ]
        home = f"{state_dir}/home"
        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "DSH_HOME": home,
            "DSH_PERMISSION_MODE": "danger-full-access",
            "DSH_TELEMETRY_DISABLED": "1",
            "HOME": home,
        }
        return await runtime.run_program(argv, env)


__all__ = ["DeepSeekHarness", "DeepSeekHarnessConfig"]
