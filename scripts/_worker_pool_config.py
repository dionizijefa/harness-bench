"""Read private worker settings, with exported variables taking precedence."""

from __future__ import annotations

import os
import re
import shlex
import tempfile
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
    path = repo_root / ".env"
    if path.is_file():
        values.update(parse_env(path.read_text(encoding="utf-8")))
    values.update(os.environ)
    return values


def update_env(path: Path, settings: dict[str, str]) -> None:
    """Update selected settings while preserving other entries and comments."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.is_file() else []
    remaining = dict(settings)
    for index, line in enumerate(lines):
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if match and (key := match.group(1)) in settings:
            lines[index] = f"{key}={shlex.quote(settings[key])}\n"
            remaining.pop(key, None)
    if remaining and lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.extend(f"{key}={shlex.quote(value)}\n" for key, value in remaining.items())
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as file:
        temporary_path = Path(file.name)
        file.write("".join(lines))
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
