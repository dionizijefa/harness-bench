"""Pi coding-agent harness."""

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
    openai_compat_model_config,
    release_version,
    run_install,
    runtime_linux_arch,
    shell_assignment,
)

logger = logging.getLogger(__name__)

PI_AGENT_DIR = "/tmp/vf-pi-agent"
PI_AGENT_NODE_DIR = f"{PI_AGENT_DIR}/node"
PI_AGENT_PREFIX = f"{PI_AGENT_DIR}/install"
PI_AGENT_NODE_BIN = f"{PI_AGENT_NODE_DIR}/bin/node"
PI_AGENT_NPM_BIN = f"{PI_AGENT_NODE_DIR}/bin/npm"
PI_AGENT_CLI = (
    f"{PI_AGENT_PREFIX}/lib/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
)


def _install_script(version: str) -> str:
    version = release_version(version)
    package = f"@earendil-works/pi-coding-agent@{version}"
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
DIR={shlex.quote(PI_AGENT_DIR)}
PREFIX={shlex.quote(PI_AGENT_PREFIX)}
NODE={shlex.quote(PI_AGENT_NODE_BIN)}
NPM={shlex.quote(PI_AGENT_NPM_BIN)}
CLI={shlex.quote(PI_AGENT_CLI)}

{node_install_script(PI_AGENT_NODE_DIR)}

current="$($NODE $CLI --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"pi $VERSION") exit 0 ;;
esac

mkdir -p "$DIR/build-home" "$PREFIX"
PATH="$NODE_DIR/bin:$PATH" \
HOME="$DIR/build-home" \
"$NPM" install -g --prefix "$PREFIX" --ignore-scripts --no-fund --no-audit --loglevel=error --progress=false {shlex.quote(package)}

test -f "$CLI"
current="$($NODE $CLI --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"pi $VERSION") ;;
    *) echo "Pi version mismatch: expected $VERSION, got $current" >&2; exit 1 ;;
esac
"""


class PiAgentHarnessConfig(HarnessConfig):
    version: str = Field(default="0.84.2", pattern=r"^[A-Za-z0-9._+-]+$")
    """Pi coding-agent npm release, pinned for reproducibility."""


class PiAgentHarness(Harness[PiAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        logger.info("pi-agent: loading Pi %s", self.config.version)
        arch = await runtime_linux_arch(runtime)
        cached_tree = await asyncio.to_thread(
            ensure_docker_built_tree,
            harness="pi-agent",
            version=self.config.version,
            source_dir=PI_AGENT_DIR,
            install_script=_install_script(self.config.version),
            arch=arch,
            bundle_python_runtime=False,
        )
        await stage_cached_tree(runtime, cached_tree, PI_AGENT_DIR)
        await run_install(
            runtime,
            "Pi",
            self.config.version,
            f"""\
set -eu
test -x {shlex.quote(PI_AGENT_NODE_BIN)}
test -f {shlex.quote(PI_AGENT_CLI)}
{shlex.quote(PI_AGENT_NODE_BIN)} {shlex.quote(PI_AGENT_CLI)} --version >/dev/null
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
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("Pi requires a task prompt")

        state_dir = f"/tmp/vf-pi-agent-state-{trace.id}"
        agent_dir = f"{state_dir}/agent"
        models = openai_compat_model_config(
            endpoint=endpoint,
            api_key=f"${INTERCEPT_KEY_VAR}",
        )
        await runtime.write(f"{agent_dir}/models.json", json_bytes(models))

        argv = [
            PI_AGENT_NODE_BIN,
            PI_AGENT_CLI,
            "--provider",
            INTERCEPT_PROVIDER,
            "--model",
            INTERCEPT_MODEL,
            "--no-session",
            "--no-approve",
            "--offline",
        ]
        if self.config.disabled_tools:
            argv += ["--exclude-tools", ",".join(self.config.disabled_tools)]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        argv += ["--print", prompt]

        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "HOME": state_dir,
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }
        return await runtime.run_program(argv, env)


__all__ = ["PiAgentHarness", "PiAgentHarnessConfig"]
