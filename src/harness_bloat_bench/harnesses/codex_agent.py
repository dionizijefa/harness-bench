"""Version-aware compatibility wrapper for Verifiers' Codex CLI harness."""

import asyncio
import shlex
from typing import Any, cast

from verifiers.v1.clients import ModelContext
from verifiers.v1.harnesses.codex import CodexHarness, CodexHarnessConfig
from verifiers.v1.harnesses.codex.harness import CODEX_BIN, CODEX_DIR
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

from harness_bloat_bench.harness_cache import ensure_codex_cached, stage_cached_file

from ._common import release_version, run_install, shell_assignment


def _version_tuple(version: str) -> tuple[int, ...]:
    normalized = release_version(version)
    try:
        return tuple(int(part) for part in normalized.split("."))
    except ValueError as error:
        raise ValueError(f"Codex version must be numeric: {version}") from error


def _install_script(version: str) -> str:
    """Verify the pinned Codex binary staged from the persistent cache."""
    version = release_version(version)
    _version_tuple(version)
    return f"""\
set -eu
{shell_assignment("VERSION", version)}
BIN={shlex.quote(CODEX_BIN)}

current="$($BIN --version 2>/dev/null || true)"
case "$current" in
    "$VERSION"|"v$VERSION"|"codex $VERSION"|"codex-cli $VERSION") exit 0 ;;
    *) echo "cached Codex binary version mismatch: expected $VERSION, got $current" >&2; exit 1 ;;
esac
"""


def _versioned_argv(argv: list[str], version: str) -> list[str]:
    """Remove Verifiers feature overrides and translate explicit v2 opt-in."""
    release = _version_tuple(version)
    filtered: list[str] = []
    index = 0
    before_model = True
    while index < len(argv):
        arg = argv[index]
        if arg == "-m":
            before_model = False
        if (
            before_model
            and arg == "--disable"
            and index + 1 < len(argv)
            and argv[index + 1] in {"apps", "plugins", "multi_agent"}
        ):
            index += 2
            continue
        if (
            before_model
            and arg == "-c"
            and index + 1 < len(argv)
            and argv[index + 1].startswith("features.multi_agent_v2.enabled=")
        ):
            setting = argv[index + 1]
            if setting.endswith("=false") or release < (0, 118, 0):
                index += 2
                continue
            if release < (0, 124, 0):
                filtered += [
                    "-c",
                    setting.replace(
                        "features.multi_agent_v2.enabled=",
                        "features.multi_agent_v2=",
                        1,
                    ),
                ]
                index += 2
                continue
        filtered.append(arg)
        index += 1
    return filtered


class _VersionedRuntime:
    """Delegate runtime operations while adapting Codex's final command."""

    def __init__(self, runtime: Runtime, version: str) -> None:
        self._runtime = runtime
        self._version = version

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self._runtime.run_program(
            _versioned_argv(argv, self._version), env
        )


class CodexAgentHarness(CodexHarness):
    """Expose the stock Codex harness under the explicit ``codex_agent`` ID."""

    async def setup(self, runtime: Runtime) -> None:
        cached_binary = await asyncio.to_thread(
            ensure_codex_cached, self.config.version
        )
        await stage_cached_file(runtime, cached_binary, CODEX_BIN)
        script = _install_script(self.config.version)
        guarded = (
            f"mkdir -p {shlex.quote(CODEX_DIR)} && "
            f"flock {shlex.quote(f'{CODEX_DIR}/install.lock')} "
            f"sh -c {shlex.quote(script)}"
        )
        await run_install(runtime, "Codex", self.config.version, guarded)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        versioned_runtime = cast(
            Runtime, _VersionedRuntime(runtime, self.config.version)
        )
        return await super().launch(
            ctx, trace, versioned_runtime, endpoint, secret, mcp_urls
        )


__all__ = ["CodexAgentHarness", "CodexHarnessConfig"]
