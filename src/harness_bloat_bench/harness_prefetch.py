"""Resolve complete, host-side harness installations before timed rollouts."""

from __future__ import annotations

from pathlib import Path

from harness_bloat_bench.harness_cache import (
    ensure_claude_cached,
    ensure_codex_cached,
    ensure_docker_built_tree,
    ensure_omp_cached,
    ensure_opencode_cached,
    ensure_pi_cached,
)
from harness_bloat_bench.harnesses.hermes_agent import (
    HERMES_DIR,
    _install_script as hermes_install_script,
)
from harness_bloat_bench.harnesses.opencode import (
    OPENCODE_ARTIFACT_ALIASES,
    _version_tuple as opencode_version_tuple,
)
from harness_bloat_bench.harnesses.prime_agent import (
    PRIME_AGENT_DIR,
    _install_script as prime_install_script,
)


def ensure_harness_cached(
    harness: str, version: str, arch: str | None = None
) -> tuple[Path, ...]:
    """Cache every artifact variant a rollout may need for one harness release."""
    if harness == "claude_code_agent":
        return (ensure_claude_cached(version, arch),)
    if harness in {"codex", "codex_agent"}:
        return (ensure_codex_cached(version, arch),)
    if harness == "hermes_agent":
        return (
            ensure_docker_built_tree(
                harness="hermes-agent",
                version=version,
                source_dir=HERMES_DIR,
                install_script=hermes_install_script(version),
                arch=arch,
            ),
        )
    if harness == "omp_agent":
        return (ensure_omp_cached(version, arch),)
    if harness == "opencode":
        artifact_version = OPENCODE_ARTIFACT_ALIASES.get(version, version)
        paths = [
            ensure_opencode_cached(version, artifact_version, arch, musl=False)
        ]
        if opencode_version_tuple(artifact_version) >= (1, 0, 0):
            paths.append(
                ensure_opencode_cached(version, artifact_version, arch, musl=True)
            )
        return tuple(paths)
    if harness == "pi":
        return (ensure_pi_cached(version, arch),)
    if harness == "prime_agent":
        return (
            ensure_docker_built_tree(
                harness="prime-agent",
                version=version,
                source_dir=PRIME_AGENT_DIR,
                install_script=prime_install_script(version),
                arch=arch,
                bundle_python_runtime=False,
            ),
        )
    raise ValueError(f"unknown harness: {harness}")
