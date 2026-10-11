#!/usr/bin/env python3
"""Render the distributed Dagster config for this machine's paths and user."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=str(repo_root))
    parser.add_argument("--user", default=f"{os.getuid()}:{os.getgid()}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_dir = str(Path(args.project_dir).resolve())
    if ":" in project_dir or "\n" in project_dir:
        parser.error("project directory cannot contain a colon or newline")
    config = (repo_root / "deploy/distributed/dagster.yaml").read_text(encoding="utf-8")
    for placeholder, value in {
        "__RUN_USER__": args.user,
        "__PROJECT_VOLUME__": f"{project_dir}:{project_dir}",
        "__PROJECT_DIR__": project_dir,
    }.items():
        config = config.replace(json.dumps(placeholder), json.dumps(value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(config, encoding="utf-8")


if __name__ == "__main__":
    main()
