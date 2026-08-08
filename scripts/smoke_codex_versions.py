#!/usr/bin/env python3
"""Run untracked Codex version checks against Harbor's hello-world task."""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment


DEFAULT_MODEL = "~deepseek/deepseek-v4-flash-latest"


def eval_config(version: str, model: str, output_dir: Path) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "model": model,
            "client": {
                "type": "eval",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_var": "OPENROUTER_API_KEY",
            },
            "taskset": {
                "id": "harbor",
                "dataset": "harbor/hello-world",
                "tasks": ["hello-world"],
                # The official hello-world task declares a trivial Dockerfile.
                # Codex's Ubuntu runtime already satisfies it, so no build is needed.
                "ignore_dockerfile": True,
            },
            "harness": {
                "id": "codex_agent",
                "version": version,
                "runtime": {"type": "docker", "cpu": 2, "memory": 4},
            },
            "retries": {"rollout": {"max_retries": 0}},
            "timeout": {"rollout": 300, "scoring": 300},
            "num_tasks": 1,
            "num_rollouts": 1,
            "max_concurrent": 1,
            "rich": False,
            "push": False,
            "output_dir": output_dir,
        }
    )


async def smoke_version(
    version: str, model: str, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        try:
            with tempfile.TemporaryDirectory(prefix=f"codex-{version}-smoke-") as temp:
                config = eval_config(version, model, Path(temp))
                traces = await run_eval(Environment(config), config)
                trace = traces[0]
                error = trace.error.model_dump() if trace.error else None
                passed = error is None and trace.reward >= 1
                return {
                    "version": version,
                    "passed": passed,
                    "reward": trace.reward,
                    "error": error,
                }
        except Exception as error:
            return {
                "version": version,
                "passed": False,
                "reward": None,
                "error": {"type": type(error).__name__, "message": str(error)},
            }


async def run(args: argparse.Namespace) -> list[dict]:
    semaphore = asyncio.Semaphore(args.max_concurrent)
    return await asyncio.gather(
        *(smoke_version(version, args.model, semaphore) for version in args.versions)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("versions", nargs="+", help="Codex versions to exercise")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="read OPENROUTER_API_KEY from one stdin line (useful over SSH)",
    )
    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.api_key_stdin:
        os.environ["OPENROUTER_API_KEY"] = sys.stdin.readline().strip()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is not configured")

    results = asyncio.run(run(args))
    print(json.dumps(results, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
