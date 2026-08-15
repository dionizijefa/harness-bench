#!/usr/bin/env python3
"""Download configured OMP releases into the persistent project cache."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from harness_bloat_bench.harness_cache import ensure_omp_cached, linux_arch


def configured_versions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    versions = re.findall(r"id:\s*omp_agent,\s*version:\s*([^}\s]+)", text)
    if not versions:
        raise SystemExit(f"no OMP versions found in {path}")
    return versions


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "versions",
        nargs="*",
        help="versions to cache (default: every version in configs/omp-agent-versions.yaml)",
    )
    parser.add_argument(
        "--arch",
        help="Linux release architecture: x64 or arm64 (default: current machine)",
    )
    args = parser.parse_args()
    arch = linux_arch(args.arch)
    versions = args.versions or configured_versions(
        repo_root / "configs" / "omp-agent-versions.yaml"
    )

    for index, version in enumerate(versions, 1):
        print(f"[{index}/{len(versions)}] Caching OMP {version} ({arch})...", flush=True)
        path = ensure_omp_cached(version, arch)
        print(f"    {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
