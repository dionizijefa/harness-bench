#!/usr/bin/env python3
"""Preload every configured harness artifact before benchmark rollouts."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from harness_bloat_bench.definitions import DEFAULT_HARNESS_VERSIONS
from harness_bloat_bench.harness_cache import linux_arch
from harness_bloat_bench.harness_prefetch import ensure_harness_cached


HARNESSES = (
    "claude_code_agent",
    "codex_agent",
    "deepseek_harness",
    "hermes_agent",
    "omp_agent",
    "opencode",
    "pi_agent",
    "pi_rlm_runtime",
    "prime_agent",
)


def configured_versions(config_dir: Path) -> dict[str, list[str]]:
    versions: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(r"id:\s*([a-z_]+),\s*version:\s*([^}\s]+)")
    for config in sorted(config_dir.glob("*-versions.yaml")):
        for harness, version in pattern.findall(config.read_text(encoding="utf-8")):
            if version not in versions[harness]:
                versions[harness].append(version)
    for harness in HARNESSES:
        default = DEFAULT_HARNESS_VERSIONS[harness]
        if not versions[harness]:
            versions[harness].append(default)
        elif default not in versions[harness]:
            versions[harness].append(default)
    return dict(versions)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harness",
        action="append",
        choices=HARNESSES,
        help="cache only this harness (repeatable; default: all harnesses)",
    )
    parser.add_argument(
        "--arch",
        help="Linux release architecture: x64 or arm64 (default: current machine)",
    )
    args = parser.parse_args()
    arch = linux_arch(args.arch)
    matrix = configured_versions(repo_root / "configs")
    selected = args.harness or list(HARNESSES)
    total = sum(len(matrix[harness]) for harness in selected)
    index = 0
    for harness in selected:
        for version in matrix[harness]:
            index += 1
            print(
                f"[{index}/{total}] Caching {harness} {version} ({arch})...",
                flush=True,
            )
            for path in ensure_harness_cached(harness, version, arch):
                print(f"    {path}", flush=True)


if __name__ == "__main__":
    main()
