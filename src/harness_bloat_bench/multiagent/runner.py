"""Run native VBA harness CLIs and retain legacy Terminal-Bench 2.1 entry points."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import html
import json
import logging
import os
import platform
import signal
import socket
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request
from verifiers.v1.runtimes.docker import docker
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset

from .accounting import Ledger, scrub, utc_now
from .config import TeamConfig, load_config
from .gateway import Gateway
from .harnesses import configure_artifact_cache, prepare_artifact
from .vba import load_vba_task, prepare_vba_images
from .vba_run import VbaRun




def write_report(output, results, elapsed, catalog, usage):
    manifest = json.loads((output / "manifest.json").read_text())
    overall = {"benchmark_type": "multiagent", "schema_version": 1, "runtime_seconds": elapsed, "teams": results, "model_catalog": catalog, "usage": usage}
    (output / "comparison.json").write_text(scrub(json.dumps(overall, indent=2)) + "\n")
    rows = []
    for result in results:
        name = html.escape(result["name"])
        usage = result["usage"]
        cost = f'${usage["total_cost_usd"]:.6f}' if usage["total_cost_usd"] is not None else f'≥ ${usage["provider_cost_usd"]:.6f} (incomplete)'
        partial = f'{result["partial_reward"]:.1%}' + ('' if result['score_complete'] else ' (incomplete)') if "partial_reward" in result else '—'
        components = f'{result["passed_subtasks"]}/{result["subtask_count"]}' if "subtask_count" in result else '—'
        parallel = result.get("max_observed_parallel_workers") if result.get("max_observed_parallel_workers") is not None else "unmeasured"
        overlap = f'{result["worker_overlap_seconds"]:.1f}s' if result.get("worker_overlap_seconds") is not None else "unmeasured"
        rows.append(f'<tr><td>{name}</td><td>{partial}</td><td>{components}</td><td>{result["passed_tasks"]}/{result["task_count"]}</td><td>{result["scored_tasks"]}</td><td>{cost}</td><td>{usage["total_tokens"]:,}</td><td>{parallel}</td><td>{overlap}</td><td>{html.escape(result["orchestrator"]["state"])}</td></tr>')
    agents = []
    for team in results:
        executors = [team["executor"]] if team.get("protocol") == "native_harness" else team.get("workers", team["tasks"])
        for agent in [team["orchestrator"], *executors]:
            usage = agent["usage"]
            agents.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in [team["name"], agent.get("agent_id", agent.get("task_id", "coordinator")), agent.get("model", "—"), agent["state"], agent.get("verifier_state", "—"), usage["model_call_count"], usage["input_tokens"], usage["output_tokens"], usage["reasoning_tokens"], f'${usage["provider_cost_usd"]:.6f}', usage["calls_missing_cost"]]) + '</tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multiagent harness pilot</title><style>body{font:15px/1.6 system-ui;margin:40px auto;padding:0 24px;max-width:1300px;color:#18394c;background:#fbfaf6}h1{font-size:36px}table{width:100%;border-collapse:collapse;background:white;margin:24px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #dde3e3}th{background:#edf2ef}p{max-width:1000px}.scroll{overflow:auto}a{color:#176582}</style><h1>Multiagent harness pilot</h1>'''
    page += f'<p>Comparison wall time: {elapsed:.1f}s. Hidden verifier evidence determines scores. The complete comparison has a {manifest["overall_timeout_seconds"]:.0f}-second execution cap; models, scoring protocols and limits are recorded in the frozen configs. Cleanup follows.</p>'
    if any(team.get("scoring") == "vba_subtasks_v1" for team in results):
        page += '<p>Terminal-Bench 4.0 vba-userform-port: one shared application, ten equally weighted subtasks and 28 behavioral checks. Each subtask earns its passed-check fraction times 10%. Partial reward uses a fixed denominator; missing or timed-out checks earn no credit and are marked unscored. API checks launch the declared backend independently; UI checks require the full React app. After agents stop, app sources transfer to a clean verifier container, dependencies are reinstalled, and database state is restored between checks. This custom partial-credit protocol is not the official all-or-nothing Terminal-Bench score.</p>'
    else:
        page += '<p>Legacy Terminal-Bench 2.1 protocol: shared teams submit a combined four-task environment for sequential original graders; isolated teams use separate original task images. Combined OS dependencies differ from standalone task images.</p>'
    if any(team.get("protocol") == "native_harness" for team in results):
        page += '<p>VBA teams use the pinned harness\'s native delegation and default tool selection, with no benchmark MCP tools. The orchestrator\'s final exit submits the work. Model usage is recorded separately for the orchestrator and the native executor pool; individual native helper identities, counts and overlap are unmeasured.</p>'
    if any(team.get("protocol") != "native_harness" for team in results):
        page += '<p>Legacy teams use benchmark MCP delegation through pinned harness CLIs with native subagents disabled and per-agent accounting.</p>'
    environment = manifest.get("environment", {})
    if environment:
        page += '<p>Local execution environment: ' + html.escape(json.dumps(environment)) + '. Container architecture differing from host/Docker architecture uses emulation; this can affect runtime and completion.</p>'
    page += '<div class="scroll"><table><tr><th>Team</th><th>Partial reward</th><th>Subtasks passed</th><th>Tasks passed</th><th>Tasks scored</th><th>Total cost</th><th>Tokens</th><th>Parallel workers</th><th>Overlap</th><th>Coordinator</th></tr>' + ''.join(rows) + '</table></div>'
    page += '<h2>Model accounting</h2><div class="scroll"><table><tr><th>Team</th><th>Task / role</th><th>Model</th><th>Execution</th><th>Grader</th><th>Calls</th><th>Input</th><th>Output</th><th>Reasoning</th><th>Provider cost</th><th>Calls missing cost</th></tr>' + ''.join(agents) + '</table></div>'
    grades = []
    for team in results:
        for task in team["tasks"]:
            grades.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in [team["name"], task["task_id"], task.get("verifier_state", "unscored"), task.get("reward"), task.get("error", "")]) + '</tr>')
    page += '<h2>Task grades</h2><div class="scroll"><table><tr><th>Team</th><th>Task</th><th>Grader</th><th>Reward</th><th>Error</th></tr>' + ''.join(grades) + '</table></div>'
    subtasks = []
    for team in results:
        for task in team["tasks"]:
            for subtask in task.get("subtasks", []):
                values = [team["name"], subtask["title"], subtask["criteria"], subtask["verifier_state"],
                          f'{subtask["passed_checks"]}/{subtask["check_count"]}', f'{subtask["scored_checks"]}/{subtask["check_count"]}',
                          f'{subtask["weight"]:.0%}', f'{subtask["earned_reward"]:.2%}']
                subtasks.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in values) + '</tr>')
    if subtasks:
        page += '<h2>Subtask verification</h2><div class="scroll"><table><tr><th>Team</th><th>Subtask</th><th>Criteria</th><th>Outcome</th><th>Checks passed</th><th>Checks scored</th><th>Weight</th><th>Credit earned</th></tr>' + ''.join(subtasks) + '</table></div>'
    page += '<p>Input counts include cached tokens; output counts include reasoning tokens. Missing usage or billing is explicitly reported; known provider spend is a lower bound when calls are unaccounted. Price estimates are stored separately. Setup and model availability preflight happen before the paid-run clock; sandbox setup, model work, and grading share its deadline.</p><p><a href="comparison.json">Comparison JSON</a> · <a href="manifest.json">Frozen configs and limits</a> · <a href="events.jsonl">Lifecycle events</a> · <a href="model-catalog.json">Model price snapshot</a> · <a href="multiagent.sqlite">Per-call accounting database</a></p></html>'
    (output / "report.html").write_text(page)


async def run_configs(configs: list[TeamConfig], output: Path, key: str, overall_timeout: float, prepare_only=False):
    output.mkdir(parents=True, exist_ok=False)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        available = {model["id"]: model for model in response.json()["data"]}
    names = {spec.model for config in configs for spec in (config.orchestrator, config.executor)}
    missing = names - available.keys()
    if missing:
        raise ValueError("Requested models unavailable: " + ", ".join(sorted(missing)))
    catalog = {model: {key: available[model].get(key) for key in ("id", "name", "pricing", "context_length", "supported_parameters")} for model in sorted(names)}
    (output / "model-catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    manifest = {"benchmark_type": "multiagent", "created_at": utc_now(), "overall_timeout_seconds": overall_timeout, "configs": [config.model_dump() for config in configs], "protocols": {config.name: "native_harness" if config.scoring == "vba_subtasks_v1" else config.execution_mode for config in configs}, "statistics_database": "multiagent.sqlite", "single_agent_statistics_imported": False}
    manifest["config_sha256"] = hashlib.sha256(json.dumps(manifest["configs"], sort_keys=True).encode()).hexdigest()
    manifest["implementation_sha256"] = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        relative = path.relative_to(Path(__file__).resolve().parents[3])
        source = path.read_bytes()
        manifest["implementation_sha256"][str(relative)] = hashlib.sha256(source).hexdigest()
        snapshot = output / "runner-source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "preparing", "output": str(output), "teams": [config.name for config in configs]}), flush=True)
    configure_artifact_cache()
    artifacts, tasksets, shared_images, verifier_images = {}, {}, {}, {}
    manifest["images"] = {}
    manifest["task_environments"] = {}
    for config in configs:
        artifacts[config.name] = await asyncio.to_thread(prepare_artifact, config)
        # VBA's Dockerfiles are built explicitly below, never replaced with the
        # sandbox fallback. Other Dockerfile-only tasks remain rejected.
        is_vba = config.scoring == "vba_subtasks_v1"
        if is_vba:
            task, source_reference = await load_vba_task(config)
            loaded = [task]
        else:
            loaded = await asyncio.to_thread(lambda: list(HarborTaskset(HarborConfig(dataset=config.dataset, tasks=config.task_ids, ignore_dockerfile=config.dataset == "harbor/hello-world")).load()))
        tasksets[config.name] = {task.data.name.rsplit("/", 1)[-1]: task for task in loaded}
        if set(tasksets[config.name]) != set(config.task_ids):
            raise ValueError("Dataset did not resolve exactly the requested task IDs")
        if is_vba:
            environment = await prepare_vba_images(config, loaded[0], output)
            environment["source_reference"] = source_reference
            manifest["task_environments"][config.name] = environment
            shared_images[config.name] = environment["images"]["agent"]
            verifier_images[config.name] = environment["images"]["verifier"]
            images = set(environment["images"].values())
        else:
            images = {task.data.image or config.sandbox.image for task in loaded}
        if config.execution_mode == "isolated_tasks":
            images.add(config.sandbox.image)
        for image in images:
            check = await docker("image", "inspect", image)
            if check.exit_code:
                check = await docker("pull", "--platform", config.sandbox.platform, image)
                if check.exit_code:
                    raise RuntimeError("Image preflight failed: " + check.stderr[-1000:])
                check = await docker("image", "inspect", image)
            image_info = json.loads(check.stdout)[0]
            manifest["images"][image] = {field: image_info.get(field) for field in ("Id", "RepoDigests", "Architecture", "Os")}
        if config.execution_mode == "shared_container" and not is_vba:
            from .shared_environment import prepare_shared_image
            shared_images[config.name] = await prepare_shared_image(config, tasksets[config.name], manifest["images"], output)
            check = await docker("image", "inspect", shared_images[config.name])
            image_info = json.loads(check.stdout)[0]
            manifest["images"][shared_images[config.name]] = {field: image_info.get(field) for field in ("Id", "RepoDigests", "Architecture", "Os")}
    manifest["shared_images"] = shared_images
    manifest["verifier_images"] = verifier_images
    info = await docker("info", "--format", "{{json .}}")
    docker_info = json.loads(info.stdout) if info.exit_code == 0 else {}
    manifest["environment"] = {"host_system": platform.system(), "host_architecture": platform.machine(), "docker_architecture": docker_info.get("Architecture"), "docker_cpus": docker_info.get("NCPU"), "docker_memory_gb": round(docker_info.get("MemTotal", 0) / 2**30, 2), "docker_version": docker_info.get("ServerVersion"), "container_platforms": sorted({config.sandbox.platform for config in configs})}
    interpreters = {}
    for config in configs:
        if config.sandbox.platform in interpreters:
            continue
        probe = "mab-preflight-" + uuid4().hex[:8]
        try:
            check = await docker("run", "--detach", "--platform", config.sandbox.platform, "--name", probe, "--entrypoint", "sleep", shared_images.get(config.name, config.sandbox.image), "30")
            if check.exit_code:
                raise RuntimeError("Platform preflight failed: " + check.stderr[-1000:])
            processes = await docker("top", probe, "-eo", "pid,args")
            interpreters[config.sandbox.platform] = [line.split(maxsplit=1)[1] for line in processes.stdout.splitlines()[1:]] if processes.exit_code == 0 else ["unknown"]
        finally:
            await docker("rm", "--force", probe)
    manifest["environment"]["container_entry_processes"] = interpreters
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if prepare_only:
        print(json.dumps({"event": "prepared", "output": str(output)}), flush=True)
        return
    ledger = Ledger(output)
    gateway = Gateway(key, ledger, available)
    started = time.monotonic()
    deadline = started + overall_timeout
    teams = []
    for config in configs:
        args = (config, output, ledger, gateway, artifacts[config.name], tasksets[config.name], min(deadline, started + config.batch_timeout_seconds))
        if config.scoring == "vba_subtasks_v1":
            teams.append(VbaRun(config, output, ledger, gateway, artifacts[config.name], tasksets[config.name][config.task_ids[0]],
                                args[-1], agent_image=shared_images[config.name], verifier_image=verifier_images[config.name]))
        elif config.execution_mode == "shared_container":
            from .legacy_shared import LegacySharedTeam
            teams.append(LegacySharedTeam(*args, shared_image=shared_images[config.name]))
        else:
            from .legacy_team import Team
            teams.append(Team(*args))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            for team in teams:
                if getattr(team, "mcp", None) is not None:
                    await stack.enter_async_context(team.mcp.session_manager.run())
            yield
    app = FastAPI(lifespan=lifespan)
    @app.post("/agents/{agent_id}/v1/{path:path}")
    async def relay(agent_id: str, path: str, request: Request):
        return await gateway.relay(agent_id, path, request)
    for team in teams:
        if getattr(team, "mcp", None) is not None:
            app.mount(team.mcp_path, team.mcp.streamable_http_app())
    server_socket = socket.socket()
    server_socket.bind(("0.0.0.0", 0))
    server_socket.listen(128)
    port = server_socket.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    # Preserve the CLI's SIGINT/SIGTERM cancellation handler.
    server.capture_signals = contextlib.nullcontext
    serving = asyncio.create_task(server.serve(sockets=[server_socket]))
    team_tasks = []
    async def progress():
        while True:
            await asyncio.sleep(15)
            for team in teams:
                team.write_result()
            statuses = []
            for team in teams:
                usage = ledger.summarize(team=team.config.name)
                statuses.append({"name": team.config.name, "coordinator": team.coordinator["state"], "phase": getattr(team, "phase", None),
                                 "tasks": {row["task_id"]: row["state"] for row in team.summary()["tasks"]},
                                 "partial_reward": team.summary().get("partial_reward"),
                                 "calls": usage["model_call_count"], "tokens": usage["total_tokens"], "provider_cost_usd": usage["provider_cost_usd"]})
            print(json.dumps({"event": "progress", "elapsed_seconds": round(time.monotonic() - started, 1), "teams": statuses}), flush=True)
    monitoring = asyncio.create_task(progress())
    try:
        while not server.started:
            if serving.done():
                await serving
                raise RuntimeError("Local model relay did not start")
            await asyncio.sleep(0.02)
        ledger.event("comparison_started", teams=[config.name for config in configs], timeout_seconds=overall_timeout)
        print(json.dumps({"event": "comparison_started", "output": str(output), "timeout_seconds": overall_timeout}), flush=True)
        team_tasks = [asyncio.create_task(team.run(f"http://127.0.0.1:{port}")) for team in teams]
        async with asyncio.timeout_at(deadline):
            await asyncio.gather(*team_tasks)
    except TimeoutError:
        ledger.event("comparison_deadline_reached")
    finally:
        monitoring.cancel()
        await asyncio.gather(monitoring, return_exceptions=True)
        for task in team_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*team_tasks, return_exceptions=True)
        for team in teams:
            for task in getattr(team, "active", {}).values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*getattr(team, "active", {}).values(), return_exceptions=True)
            await asyncio.gather(*(runtime.stop() for runtime in team.runtimes), return_exceptions=True)
            team.write_result()
        await gateway.close()
        for team in teams:
            if team.finished_at is None:
                team.finished_at = time.monotonic()
            team.write_result()
        results = [team.summary() for team in teams]
        for name in ledger.summarize()["served_models"]:
            if name in available and name not in catalog:
                catalog[name] = {field: available[name].get(field) for field in ("id", "name", "pricing", "context_length", "supported_parameters")}
        (output / "model-catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        usage = ledger.summarize()
        elapsed = time.monotonic() - started
        ledger.event("comparison_finished", runtime_seconds=elapsed)
        write_report(output, results, elapsed, catalog, usage)
        server.should_exit = True
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(serving, timeout=5)
        ledger.db.close()
        print(scrub(json.dumps({"event": "comparison_finished", "output": str(output), "runtime_seconds": elapsed, "teams": [{"name": result["name"], "passed_tasks": result["passed_tasks"], "task_count": result["task_count"], "partial_reward": result.get("partial_reward"), "passed_subtasks": result.get("passed_subtasks"), "score_complete": result.get("score_complete"), "coordinator_state": result["orchestrator"]["state"], "usage": result["usage"]} for result in results]})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True, help="Fresh standalone multiagent JSON; repeat for concurrent teams")
    parser.add_argument("--output", type=Path, default=Path("outputs/multiagent") / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]))
    parser.add_argument("--overall-timeout-seconds", type=float, help="Defaults to the largest configured team deadline (up to 29700s)")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    configs = [load_config(path) for path in args.config]
    if args.overall_timeout_seconds is None:
        args.overall_timeout_seconds = max(config.batch_timeout_seconds for config in configs)
    if not 0 < args.overall_timeout_seconds <= 29700:
        parser.error("Overall timeout must be positive and at most 29700 seconds")
    if len({config.name for config in configs}) != len(configs):
        parser.error("Team names must be unique")
    if len({json.dumps({"tasks": config.task_ids, "dataset": config.dataset, "scoring": config.scoring, "execution_mode": config.execution_mode, "orchestrator": config.orchestrator.model_dump(), "executor": config.executor.model_dump()}, sort_keys=True) for config in configs}) > 1:
        parser.error("Comparison teams must use identical tasks and model settings")
    key = os.environ.get("MULTIAGENT_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not key and not args.prepare_only:
        parser.error("Set MULTIAGENT_OPENROUTER_API_KEY or OPENROUTER_API_KEY")
    async def execute():
        current = asyncio.current_task()
        for signum in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(signum, current.cancel)
        await run_configs(configs, args.output.resolve(), key or "", args.overall_timeout_seconds, args.prepare_only)
    try:
        asyncio.run(execute())
    except KeyboardInterrupt:
        pass
    except asyncio.CancelledError:
        print("Comparison cancelled; partial accounting and outcomes saved.", flush=True)


if __name__ == "__main__":
    main()
