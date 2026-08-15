"""Nous Research Hermes Agent harness."""

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
    install_cached_python_runtime_script,
    stage_cached_tree,
)

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
HERMES_BIN = f"{HERMES_DIR}/bin/hermes"
ThinkingLevel = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
]

# Hermes' public product versions are stored in ``pyproject.toml`` and release
# names, while the corresponding Git tags use calendar versions. Keep benchmark
# labels in the public version scheme and pin each one to its exact source tag.
HERMES_RELEASE_TAGS = {
    "0.2.0": "2026.3.12",
    "0.4.0": "2026.3.23",
    "0.6.0": "2026.3.30",
    "0.8.0": "2026.4.8",
    "0.10.0": "2026.4.16",
    "0.12.0": "2026.4.30",
    "0.14.0": "2026.5.16",
    "0.16.0": "2026.6.5",
    "0.18.2": "2026.7.7.2",
    "0.20.0": "2026.8.3",
}


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = release_version(version).split(".")
    if not parts or not all(part.isdigit() for part in parts):
        raise ValueError(f"Hermes Agent version must be numeric: {version}")
    return tuple(int(part) for part in parts)


def _product_version_tuple(version: str) -> tuple[int, ...]:
    version = release_version(version)
    if version in HERMES_RELEASE_TAGS:
        return _version_tuple(version)
    for product_version, release_tag in HERMES_RELEASE_TAGS.items():
        if version == release_tag:
            return _version_tuple(product_version)
    return _version_tuple(version)


def _install_script(version: str) -> str:
    version = release_version(version)
    release_tag = f"v{HERMES_RELEASE_TAGS.get(version, version)}"
    product_version = _product_version_tuple(version)
    installer_args = ["--skip-setup"]
    if product_version >= (0, 14, 0):
        installer_args.append("--skip-browser")
    if product_version >= (0, 16, 0):
        installer_args.extend(["--no-skills", "--non-interactive"])
    installer_args_text = " ".join(installer_args)
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
{shell_assignment("RELEASE_TAG", release_tag)}
DIR={shlex.quote(HERMES_DIR)}
SOURCE={shlex.quote(HERMES_SOURCE_DIR)}
BIN={shlex.quote(HERMES_BIN)}

current="$(git -C "$SOURCE" describe --tags --exact-match 2>/dev/null || true)"
if [ -x "$BIN" ] && [ "$current" = "$RELEASE_TAG" ]; then
    exit 0
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v bash >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1 || ! command -v xz >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get -o Acquire::Retries=3 update -qq
        apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates git bash tar xz-utils >/dev/null
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl ca-certificates git bash tar xz >/dev/null
    else
        echo "Hermes Agent needs curl, Git, Bash, tar, xz, and CA certificates" >&2
        exit 1
    fi
fi

rm -rf "$SOURCE"
mkdir -p "$DIR"
installer="$DIR/install.sh"
trap 'rm -f "$installer"' EXIT
curl -fsSL \
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/$RELEASE_TAG/scripts/install.sh" \
    -o "$installer"
# v0.12.0 and v0.14.0 predate the upstream shallow-clone optimization. A
# rollout only needs its pinned tag, not the repository's full history.
sed -i 's/git clone --branch /git clone --depth 1 --branch /g' "$installer"
# The benchmark uses the Python CLI and terminal tools, not Hermes' browser,
# media, desktop, or TUI integrations. Avoid installing Node, FFmpeg, browser
# assets, and related workstation packages into every disposable task image.
# Limit these rewrites to main() so helper function definitions remain intact.
sed -i '/^main() {{$/,/^    setup_path$/ {{
    s/^    check_node$/    HAS_NODE=false # skip optional Node runtime/
    s/^    install_system_packages$/    : # skip optional system packages/
    s/^    install_node_deps$/    : # skip optional Node dependencies/
}}' "$installer"
HERMES_HOME="$DIR/install-state" HERMES_INSTALL_DIR="$SOURCE" \
    bash "$installer" \
        --branch "$RELEASE_TAG" \
        {installer_args_text}

mkdir -p "$DIR/bin"
rm -f "$BIN"
if [ -x "$SOURCE/venv/bin/hermes" ]; then
    ln -s "$SOURCE/venv/bin/hermes" "$BIN"
elif [ -x "$SOURCE/venv/bin/python" ] && [ -f "$SOURCE/hermes" ]; then
    cat >"$BIN" <<EOF
#!/bin/sh
unset PYTHONHOME PYTHONPATH
exec "$SOURCE/venv/bin/python" "$SOURCE/hermes" "\\$@"
EOF
    chmod 755 "$BIN"
else
    echo "Hermes Agent $VERSION did not install a runnable CLI" >&2
    exit 1
fi
"""


class HermesAgentHarnessConfig(HarnessConfig):
    version: str = "0.20.0"
    """Hermes Agent release tag to install, pinned for reproducibility."""

    thinking: ThinkingLevel | None = "medium"
    toolsets: tuple[str, ...] = ("terminal", "file", "code_execution")
    max_turns: int = 90
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS


class HermesAgentHarness(Harness[HermesAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "hermes-agent: loading cached Hermes Agent %s",
            self.config.version,
        )
        cached_tree = await asyncio.to_thread(
            ensure_docker_built_tree,
            harness="hermes-agent",
            version=self.config.version,
            source_dir=HERMES_DIR,
            install_script=_install_script(self.config.version),
        )
        await stage_cached_tree(runtime, cached_tree, HERMES_DIR)
        await run_install(
            runtime,
            "Hermes Agent",
            self.config.version,
            f"""\
set -eu
{install_cached_python_runtime_script(HERMES_DIR)}
test -x {shlex.quote(HERMES_BIN)}
{shlex.quote(HERMES_BIN)} --version >/dev/null
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
                "Hermes Agent supports disabling toolsets, not arbitrary tools"
            )
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if prompt is None:
            raise ValueError("Hermes Agent requires a task prompt")

        state_dir = f"/tmp/vf-hermes-agent-{trace.id}"
        product_version = _product_version_tuple(self.config.version)
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
                "reasoning_effort": self.config.thinking or "none",
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
        config_bytes = json_bytes(config)
        await runtime.write(f"{state_dir}/config.yaml", config_bytes)
        # v0.2.0's chat module still reads ~/.hermes directly even though the
        # command router and MCP loader honor HERMES_HOME.
        if product_version < (0, 4, 0):
            await runtime.write(
                f"{state_dir}/home/.hermes/config.yaml",
                config_bytes,
            )

        argv = [
            HERMES_BIN,
            "chat",
            "--model",
            ctx.model,
            "--quiet",
            "--yolo",
        ]
        if self.config.toolsets:
            argv += ["--toolsets", ",".join(self.config.toolsets)]
        # These flags were added over time; the config carries the equivalent
        # settings for releases whose argparse surface predates them.
        if product_version >= (0, 12, 0):
            argv += ["--provider", "custom"]
        if product_version >= (0, 8, 0):
            argv += ["--max-turns", str(self.config.max_turns)]
        if self.config.thinking is not None and product_version >= (0, 20, 0):
            argv += ["--reasoning", self.config.thinking]
        argv += ["--query", prompt]

        env = {
            **self.config.resolved_env,
            INTERCEPT_KEY_VAR: secret,
            # All historical releases recognize the standard OpenAI-compatible
            # variables, including v0.2.0 before config interpolation existed.
            "OPENAI_API_KEY": secret,
            "OPENAI_BASE_URL": endpoint,
            "HERMES_INFERENCE_PROVIDER": "custom",
            "HERMES_EPHEMERAL_SYSTEM_PROMPT": system_prompt or "",
            "HERMES_HOME": state_dir,
            "HERMES_SKIP_NODE_BOOTSTRAP": "1",
            "HOME": f"{state_dir}/home",
            "NO_COLOR": "1",
        }
        return await runtime.run_program(argv, env)


__all__ = ["HermesAgentHarness", "HermesAgentHarnessConfig"]
