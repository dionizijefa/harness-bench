#!/usr/bin/env python3
"""Run one isolated Codex hello-world diagnostic, without importing benchmark stats."""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from uuid import uuid4

from verifiers.v1.cli.eval.runner import run_eval
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.env import Environment

from harness_bloat_bench.harnesses import codex_agent
from harness_bloat_bench.smoke_harness_versions import eval_config


MODELS = {
    "muse": "meta/muse-spark-1.3-contributor",
    "glm": "z-ai/glm-5.3-flash",
}


def scrub(text):
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        text = text.replace(key, "[REDACTED]")
    return re.sub(r"sk-or-[A-Za-z0-9_-]+", "[REDACTED]", text)


async def diagnose(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = eval_config("codex_agent", "0.147.0", MODELS[args.model], output / "eval")
    if args.effort:
        config.sampling.reasoning_effort = args.effort
    config.timeout.rollout = 180
    records = []
    original_request = EvalClient._request
    original_program = codex_agent._VersionedRuntime.run_program

    def save(name, payload):
        (output / name).write_text(scrub(json.dumps(payload, indent=2, default=str)) + "\n")

    async def request(client, url, body, headers, *, stream=False):
        if len(records) >= 8:
            raise RuntimeError("Diagnostic stopped at its eight-request limit")
        record = {"url": url, "request": body}
        records.append(record)
        save("requests.json", records)
        print(json.dumps({"request": len(records), "model": body.get("model"),
                          "reasoning": body.get("reasoning"),
                          "tools": [{"type": t.get("type"), "name": t.get("name")}
                                    for t in body.get("tools", [])]}), flush=True)
        try:
            result = await original_request(client, url, body, headers, stream=stream)
            record["http_status"] = result.status_code
            return result
        except Exception as error:
            record["error"] = scrub(str(error))
            raise
        finally:
            save("requests.json", records)

    async def program(runtime, argv, env):
        effective = codex_agent._versioned_argv(argv, runtime._version)
        save("codex-command.json", effective)
        result = await original_program(runtime, argv, env)
        save("codex-program.json", {"exit_code": result.exit_code,
                                    "stdout": result.stdout, "stderr": result.stderr})
        return result

    EvalClient._request = request
    codex_agent._VersionedRuntime.run_program = program
    result = {"model": MODELS[args.model], "codex_version": "0.147.0",
              "task": "harbor/hello-world", "effort_override": args.effort,
              "output": str(output), "statistics_imported": False}
    try:
        traces = await run_eval(Environment(config), config)
        trace = traces[0]
        result.update(reward=trace.reward, passed=trace.error is None and trace.reward >= 1,
                      error=trace.error.model_dump() if trace.error else None)
    except Exception as error:
        result.update(passed=False, error={"type": type(error).__name__, "message": scrub(str(error))})
    finally:
        EvalClient._request = original_request
        codex_agent._VersionedRuntime.run_program = original_program
    result["requests"] = len(records)
    save("result.json", result)
    print(scrub(json.dumps(result, indent=2)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--effort", choices=["medium", "high", "max"])
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/diagnostics") / ("codex-openrouter-" + uuid4().hex[:10]))
    options = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is not configured")
    asyncio.run(diagnose(options))
