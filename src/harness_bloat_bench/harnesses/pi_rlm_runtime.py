"""Pi with the pi-rlm-runtime recursive-agent extension."""

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
    install_cached_python_runtime_script,
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
    shell_assignment,
)

logger = logging.getLogger(__name__)

PI_RLM_RUNTIME_DIR = "/tmp/vf-pi-rlm-runtime"
PI_RLM_RUNTIME_NODE_DIR = f"{PI_RLM_RUNTIME_DIR}/node"
PI_RLM_RUNTIME_PREFIX = f"{PI_RLM_RUNTIME_DIR}/install"
PI_RLM_RUNTIME_NODE_BIN = f"{PI_RLM_RUNTIME_NODE_DIR}/bin/node"
PI_RLM_RUNTIME_NPM_BIN = f"{PI_RLM_RUNTIME_NODE_DIR}/bin/npm"
PI_RLM_RUNTIME_PI_CLI = (
    f"{PI_RLM_RUNTIME_PREFIX}/lib/node_modules/"
    "@earendil-works/pi-coding-agent/dist/cli.js"
)
PI_RLM_RUNTIME_EXTENSION = (
    f"{PI_RLM_RUNTIME_PREFIX}/lib/node_modules/pi-rlm-runtime/dist/index.js"
)
PI_RLM_RUNTIME_MANIFEST = (
    f"{PI_RLM_RUNTIME_PREFIX}/lib/node_modules/pi-rlm-runtime/package.json"
)
PI_RLM_RUNTIME_KERNEL_VENV = f"{PI_RLM_RUNTIME_DIR}/kernel-venv"
PI_RLM_RUNTIME_KERNEL_PYTHON = f"{PI_RLM_RUNTIME_KERNEL_VENV}/bin/python"
PI_RLM_RUNTIME_PYTHON_DEPENDENCY_DIR = "/tmp/vf-pi-rlm-runtime-python-libs"
PI_RLM_RUNTIME_PI_VERSION = "0.84.2"
PI_RLM_RUNTIME_IPYKERNEL_VERSION = "7.2.0"


def _install_script(version: str) -> str:
    version = release_version(version)
    pi_package = (
        f"@earendil-works/pi-coding-agent@{PI_RLM_RUNTIME_PI_VERSION}"
    )
    runtime_package = f"pi-rlm-runtime@{version}"
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
{shell_assignment("PI_VERSION", PI_RLM_RUNTIME_PI_VERSION)}
DIR={shlex.quote(PI_RLM_RUNTIME_DIR)}
PREFIX={shlex.quote(PI_RLM_RUNTIME_PREFIX)}
NODE={shlex.quote(PI_RLM_RUNTIME_NODE_BIN)}
NPM={shlex.quote(PI_RLM_RUNTIME_NPM_BIN)}
PI_CLI={shlex.quote(PI_RLM_RUNTIME_PI_CLI)}
RLM_EXTENSION={shlex.quote(PI_RLM_RUNTIME_EXTENSION)}
RLM_MANIFEST={shlex.quote(PI_RLM_RUNTIME_MANIFEST)}
KERNEL_VENV={shlex.quote(PI_RLM_RUNTIME_KERNEL_VENV)}
KERNEL_PYTHON={shlex.quote(PI_RLM_RUNTIME_KERNEL_PYTHON)}

{node_install_script(PI_RLM_RUNTIME_NODE_DIR)}

current_pi="$($NODE $PI_CLI --version 2>/dev/null || true)"
current_runtime="$($NODE -p "require('$RLM_MANIFEST').version" 2>/dev/null || true)"
if [ "$current_pi" = "$PI_VERSION" ] && [ "$current_runtime" = "$VERSION" ] && \
    [ -f "$RLM_EXTENSION" ] && [ -x "$KERNEL_PYTHON" ] && \
    "$KERNEL_PYTHON" -c 'import IPython, ipykernel' >/dev/null 2>&1; then
    exit 0
fi

mkdir -p "$DIR/build-home" "$PREFIX"
PATH="$NODE_DIR/bin:$PATH" \
HOME="$DIR/build-home" \
"$NPM" install -g --prefix "$PREFIX" --no-fund --no-audit --loglevel=error --progress=false \
    {shlex.quote(pi_package)} {shlex.quote(runtime_package)}

python3.11 -m venv "$KERNEL_VENV"
"$KERNEL_PYTHON" -m pip install --disable-pip-version-check --no-cache-dir \
    {shlex.quote(f"ipykernel=={PI_RLM_RUNTIME_IPYKERNEL_VERSION}")}

test -f "$PI_CLI"
test -f "$RLM_EXTENSION"
test "$($NODE $PI_CLI --version)" = "$PI_VERSION"
test "$($NODE -p "require('$RLM_MANIFEST').version")" = "$VERSION"
"$KERNEL_PYTHON" -c 'import IPython, ipykernel'
"""


class PiRlmRuntimeHarnessConfig(HarnessConfig):
    version: str = Field(default="0.1.1", pattern=r"^[A-Za-z0-9._+-]+$")
    """pi-rlm-runtime npm release, pinned with its supported Pi release."""


class PiRlmRuntimeHarness(Harness[PiRlmRuntimeHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "pi-rlm-runtime: loading runtime %s with Pi %s",
            self.config.version,
            PI_RLM_RUNTIME_PI_VERSION,
        )
        cached_tree = await asyncio.to_thread(
            ensure_docker_built_tree,
            harness="pi-rlm-runtime",
            version=self.config.version,
            source_dir=PI_RLM_RUNTIME_DIR,
            install_script=_install_script(self.config.version),
        )
        await stage_cached_tree(runtime, cached_tree, PI_RLM_RUNTIME_DIR)
        await run_install(
            runtime,
            "pi-rlm-runtime",
            self.config.version,
            f"""\
set -eu
{install_cached_python_runtime_script(PI_RLM_RUNTIME_DIR, PI_RLM_RUNTIME_PYTHON_DEPENDENCY_DIR)}
test -x {shlex.quote(PI_RLM_RUNTIME_NODE_BIN)}
test -f {shlex.quote(PI_RLM_RUNTIME_PI_CLI)}
test -f {shlex.quote(PI_RLM_RUNTIME_EXTENSION)}
test -x {shlex.quote(PI_RLM_RUNTIME_KERNEL_PYTHON)}
{shlex.quote(PI_RLM_RUNTIME_NODE_BIN)} {shlex.quote(PI_RLM_RUNTIME_PI_CLI)} --version >/dev/null
LD_LIBRARY_PATH={shlex.quote(PI_RLM_RUNTIME_PYTHON_DEPENDENCY_DIR)} \
    {shlex.quote(PI_RLM_RUNTIME_KERNEL_PYTHON)} -c 'import IPython, ipykernel'
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
            raise ValueError("pi-rlm-runtime requires its built-in IPython tool")
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("pi-rlm-runtime requires a task prompt")

        state_dir = f"/tmp/vf-pi-rlm-runtime-state-{trace.id}"
        agent_dir = f"{state_dir}/agent"
        models = openai_compat_model_config(
            endpoint=endpoint,
            api_key=f"${INTERCEPT_KEY_VAR}",
        )
        await runtime.write(f"{agent_dir}/models.json", json_bytes(models))

        argv = [
            PI_RLM_RUNTIME_NODE_BIN,
            PI_RLM_RUNTIME_PI_CLI,
            "--provider",
            INTERCEPT_PROVIDER,
            "--model",
            INTERCEPT_MODEL,
            "--extension",
            PI_RLM_RUNTIME_EXTENSION,
            "--rlm-runtime",
            "--rlm-runtime-max-depth",
            "4",
            "--no-session",
            "--no-approve",
            "--offline",
        ]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        argv += ["--print", "--", prompt]

        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "HOME": state_dir,
            "LD_LIBRARY_PATH": PI_RLM_RUNTIME_PYTHON_DEPENDENCY_DIR,
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_RLM_RUNTIME_PYTHON": PI_RLM_RUNTIME_KERNEL_PYTHON,
            "PI_TELEMETRY": "0",
        }
        return await runtime.run_program(argv, env)


__all__ = ["PiRlmRuntimeHarness", "PiRlmRuntimeHarnessConfig"]
