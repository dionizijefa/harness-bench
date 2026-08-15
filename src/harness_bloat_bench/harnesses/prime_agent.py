"""PrimeAgent harness with its stock IPython/RLM coding surface."""

import asyncio
import logging
import shlex
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from harness_bloat_bench.harness_cache import (
    ensure_docker_built_tree,
    stage_cached_tree,
)

from ._common import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
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

PRIME_AGENT_DIR = "/tmp/vf-prime-agent"
PRIME_AGENT_NODE_DIR = f"{PRIME_AGENT_DIR}/node"
PRIME_AGENT_PREFIX = f"{PRIME_AGENT_DIR}/install"
PRIME_AGENT_NODE_BIN = f"{PRIME_AGENT_NODE_DIR}/bin/node"
PRIME_AGENT_NPM_BIN = f"{PRIME_AGENT_NODE_DIR}/bin/npm"
PRIME_AGENT_CLI = (
    f"{PRIME_AGENT_PREFIX}/lib/node_modules/prime-agent/dist/bundle/cli.js"
)
PRIME_AGENT_KERNEL_VENV = f"{PRIME_AGENT_DIR}/kernel-venv"
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def _version_tuple(version: str) -> tuple[int, ...]:
    normalized = release_version(version)
    try:
        return tuple(int(part) for part in normalized.split("."))
    except ValueError as error:
        raise ValueError(f"PrimeAgent version must be numeric: {version}") from error


def _install_script(version: str) -> str:
    version = release_version(version)
    _version_tuple(version)
    package_url = (
        "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/"
        f"v{version}/prime-agent-{version}.tgz"
    )
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
DIR={shlex.quote(PRIME_AGENT_DIR)}
PREFIX={shlex.quote(PRIME_AGENT_PREFIX)}
NODE={shlex.quote(PRIME_AGENT_NODE_BIN)}
NPM={shlex.quote(PRIME_AGENT_NPM_BIN)}
CLI={shlex.quote(PRIME_AGENT_CLI)}
KERNEL_VENV={shlex.quote(PRIME_AGENT_KERNEL_VENV)}

{node_install_script(PRIME_AGENT_NODE_DIR)}

current="$($NODE $CLI --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"prime-agent $VERSION"|"$VERSION "*) exit 0 ;;
esac

mkdir -p "$DIR/home" "$PREFIX"
PATH="$NODE_DIR/bin:$PATH" \
HOME="$DIR/home" \
PRIME_AGENT_CODING_AGENT_DIR="$DIR/shared-agent" \
PRIME_AGENT_KERNEL_VENV="$KERNEL_VENV" \
PRIME_AGENT_BOOTSTRAP_KERNEL_ON_INSTALL=1 \
PRIME_AGENT_BOOTSTRAP_TOOLS_ON_INSTALL=1 \
PRIME_AGENT_INSTALL_UV=1 \
"$NPM" install -g --prefix "$PREFIX" --no-fund --no-audit --loglevel=error --progress=false {shlex.quote(package_url)}

if [ ! -f "$CLI" ] || [ ! -x "$KERNEL_VENV/bin/python" ]; then
    echo "PrimeAgent $VERSION did not install a runnable CLI and kernel" >&2
    exit 1
fi
"""


class PrimeAgentHarnessConfig(HarnessConfig):
    version: str = "0.7.1"
    """PrimeAgent release to install, pinned for reproducibility."""

    thinking: ThinkingLevel | None = "high"
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS


class PrimeAgentHarness(Harness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "prime-agent: loading cached PrimeAgent %s", self.config.version
        )
        cached_tree = await asyncio.to_thread(
            ensure_docker_built_tree,
            harness="prime-agent",
            version=self.config.version,
            source_dir=PRIME_AGENT_DIR,
            install_script=_install_script(self.config.version),
            bundle_python_runtime=False,
        )
        await stage_cached_tree(runtime, cached_tree, PRIME_AGENT_DIR)
        script = f"""\
set -eu
test -x {shlex.quote(PRIME_AGENT_NODE_BIN)}
test -f {shlex.quote(PRIME_AGENT_CLI)}
test -x {shlex.quote(f'{PRIME_AGENT_KERNEL_VENV}/bin/python')}
{shlex.quote(PRIME_AGENT_NODE_BIN)} {shlex.quote(PRIME_AGENT_CLI)} --version >/dev/null
"""
        guarded = (
            f"mkdir -p {shlex.quote(PRIME_AGENT_DIR)} && "
            f"flock {shlex.quote(f'{PRIME_AGENT_DIR}/install.lock')} "
            f"sh -c {shlex.quote(script)}"
        )
        await run_install(runtime, "PrimeAgent", self.config.version, guarded)

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
            raise ValueError("PrimeAgent only exposes its built-in IPython tool")
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("PrimeAgent requires a task prompt")

        agent_dir = f"/tmp/vf-prime-agent-state-{trace.id}"
        models = openai_compat_model_config(
            endpoint=endpoint,
            model_name=ctx.model,
            context_window=self.config.context_window,
            max_tokens=self.config.max_tokens,
            api_key=INTERCEPT_KEY_VAR,
        )
        await runtime.write(f"{agent_dir}/models.json", json_bytes(models))

        argv = [
            PRIME_AGENT_NODE_BIN,
            PRIME_AGENT_CLI,
            "--no-session",
            "--offline",
            "--provider",
            INTERCEPT_PROVIDER,
            "--model",
            INTERCEPT_MODEL,
        ]
        if self.config.thinking is not None:
            argv += ["--thinking", self.config.thinking]
        if system_prompt:
            argv += ["--append-system-prompt", system_prompt]
        argv += ["--print", "--", prompt]

        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            "DO_NOT_TRACK": "1",
            "HOME": f"{agent_dir}/home",
            "NO_COLOR": "1",
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PRIME_AGENT_CODING_AGENT_DIR": agent_dir,
            "PRIME_AGENT_KERNEL_VENV": PRIME_AGENT_KERNEL_VENV,
        }
        return await runtime.run_program(argv, env)


__all__ = ["PrimeAgentHarness", "PrimeAgentHarnessConfig"]
