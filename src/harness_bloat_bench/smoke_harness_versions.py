"""Shared Harbor hello-world smoke runner for versioned harness adapters."""

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


DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def eval_config(
    harness_id: str,
    version: str,
    model: str,
    output_dir: Path,
) -> EvalConfig:
    """Build the identical OpenRouter-backed hello-world check for a harness."""
    return EvalConfig.model_validate(
        {
            "model": model,
            "client": {
                "type": "eval",
                "base_url": OPENROUTER_BASE_URL,
                "api_key_var": "OPENROUTER_API_KEY",
            },
            "taskset": {
                "id": "harbor",
                "dataset": "harbor/hello-world",
                "tasks": ["hello-world"],
                # The official hello-world task declares a trivial Dockerfile.
                # The harness Ubuntu runtimes already satisfy it, so no build is needed.
                "ignore_dockerfile": True,
            },
            "harness": {
                "id": harness_id,
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
    harness_id: str,
    temp_prefix: str,
    version: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"{temp_prefix}-{version}-smoke-"
            ) as temp:
                config = eval_config(harness_id, version, model, Path(temp))
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


async def run(
    args: argparse.Namespace,
    harness_id: str,
    temp_prefix: str,
) -> list[dict]:
    semaphore = asyncio.Semaphore(args.max_concurrent)
    return await asyncio.gather(
        *(
            smoke_version(
                harness_id,
                temp_prefix,
                version,
                args.model,
                semaphore,
            )
            for version in args.versions
        )
    )


def main(*, description: str, harness_id: str, temp_prefix: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("versions", nargs="+", help="Harness versions to exercise")
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

    results = asyncio.run(run(args, harness_id, temp_prefix))
    print(json.dumps(results, indent=2))
    return 0 if all(result["passed"] for result in results) else 1
