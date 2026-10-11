"""Read private worker settings, with exported variables taking precedence."""

from __future__ import annotations

import os
from pathlib import Path


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def local_settings(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in (repo_root / ".env", repo_root / ".dagster" / "worker-pool.env"):
        if path.is_file():
            values.update(parse_env(path.read_text(encoding="utf-8")))
    values.update(os.environ)
    return values
