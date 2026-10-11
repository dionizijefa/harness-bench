"""Run harness teams with shared-container collaboration and independent graders."""
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
import secrets
import signal
import socket
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, Request
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.runtimes.docker import docker
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTaskset

from .accounting import Ledger, scrub, utc_now
from .config import TeamConfig, load_config
from .gateway import AgentRoute, Gateway
from .harnesses import BenchRuntime, configure_artifact_cache, launch_harness, prepare_artifact, setup_harness
from .shared_environment import prepare_shared_image


class Team:
    def __init__(self, config, output, ledger, gateway, artifact, tasks, deadline):
        self.config, self.output, self.ledger, self.gateway = config, output, ledger, gateway
        self.artifact, self.tasks, self.deadline = artifact, tasks, deadline
        self.started = time.monotonic()
        self.finished_at = None
        self.active: dict[str, asyncio.Task] = {}
        self.rows = {task: {"task_id": task, "state": "not_delegated", "passed": None, "reward": None} for task in config.task_ids}
        self.runtimes: list[BenchRuntime] = []
        self.semaphore = asyncio.Semaphore(config.max_parallel_workers) if config.max_parallel_workers is not None else contextlib.nullcontext()
        self.coordinator = {"agent_id": config.name + "-main", "role": "orchestrator", "state": "planned", "model": config.orchestrator.model}
        self.spawns = 0
        self.worker_intervals = []
        self.mcp = FastMCP("Harness Bench", stateless_http=True, json_response=True, transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
        self.mcp.tool()(self.delegate_tasks)
        self.mcp.tool()(self.wait_for_workers)
        self.mcp.tool()(self.team_status)
        self.mcp_path = "/tools/" + secrets.token_urlsafe(24)

    def event(self, event, **fields):
        self.ledger.event(event, team=self.config.name, **fields)

    def route(self, role, task_id=None):
        model = self.config.orchestrator if role == "orchestrator" else self.config.executor
        agent_id = self.config.name + ("-main" if task_id is None else "-" + task_id)
        route = AgentRoute(self.config.name, agent_id, role, task_id, model, self.deadline - self.config.grading_reserve_seconds, self.config.max_cost_usd)
        self.gateway.register(route)
        return route

    def runtime(self, agent_id, task_id=None):
        sandbox = self.config.sandbox
        config = DockerConfig(
            image=(self.tasks[task_id].data.image or sandbox.image) if task_id else sandbox.image,
            workdir=self.tasks[task_id].data.workdir or "/app" if task_id else "/app",
            cpu=sandbox.worker_cpus if task_id else sandbox.coordinator_cpus,
            memory=sandbox.task_memory_gb.get(task_id, sandbox.worker_memory_gb) if task_id else sandbox.coordinator_memory_gb,
        )
        runtime = BenchRuntime(config, "mab-" + uuid4().hex[:12], sandbox.platform, self.output / self.config.name / agent_id)
        self.runtimes.append(runtime)
        return runtime

    async def delegate_tasks(self, assignments: list[dict[str, str]]) -> dict:
        """Start independent task workers in parallel. Each assignment needs task_id and optional instructions. Returns immediately; use wait_for_workers to collect verified results. Each task can be launched once."""
        if time.monotonic() >= self.deadline - self.config.grading_reserve_seconds:
            return {"error": "Batch execution deadline reached"}
        ids = [assignment.get("task_id") for assignment in assignments]
        if not ids or len(set(ids)) != len(ids) or any(task not in self.rows for task in ids):
            return {"error": "Provide unique task_ids from the declared batch", "task_ids": self.config.task_ids}
        launched = []
        for assignment in assignments:
            task_id = assignment["task_id"]
            if task_id not in self.active:
                self.spawns += 1
                self.rows[task_id].update(state="queued", delegated_at=utc_now())
                self.event("worker_spawn", task_id=task_id, instructions=assignment.get("instructions", ""))
                self.active[task_id] = asyncio.create_task(self.worker(task_id, assignment.get("instructions", "")))
                launched.append(task_id)
        return {"launched": launched, "workers": self.team_status(), "parallel_limit": self.config.max_parallel_workers}

    async def wait_for_workers(self, wait_seconds: float = 60) -> dict:
        """Wait up to 60 seconds for all delegated workers, returning authoritative grader outcomes and per-worker usage. Repeat while any worker is queued/running/scoring."""
        pending = [task for task in self.active.values() if not task.done()]
        if pending:
            await asyncio.wait(pending, timeout=max(0, min(wait_seconds, 60, self.deadline - time.monotonic())))
        return self.team_status()

    def team_status(self) -> dict:
        """Return worker execution states, unchanged benchmark scores, and main/worker costs. A worker saying it succeeded is not a grader pass."""
        return {"workers": [dict(row, usage=self.ledger.summarize(agent_id=self.config.name + "-" + task)) for task, row in self.rows.items()], "remaining_seconds": max(0, self.deadline - time.monotonic()), "usage": self.ledger.summarize(team=self.config.name)}

    async def worker(self, task_id, instructions):
        row = self.rows[task_id]
        route = self.route("executor", task_id)
        runtime = self.runtime(route.agent_id, task_id)
        started = time.monotonic()
        task_deadline = min(self.deadline, started + self.config.worker_timeout_seconds)
        route.deadline = task_deadline - self.config.grading_reserve_seconds
        row.update(agent_id=route.agent_id, model=route.model.model, state="queued")
        try:
            async with asyncio.timeout_at(task_deadline), self.semaphore:
                row.update(state="setting_up", started_at=utc_now())
                async with asyncio.timeout_at(route.deadline):
                    await runtime.start()
                    await setup_harness(self.config, runtime, self.artifact)
                row["state"] = "running"
                execution_started = time.monotonic()
                row["setup_seconds"] = execution_started - started
                self.event("worker_started", task_id=task_id, agent_id=route.agent_id)
                runtime.program_timeout = max(0.1, route.deadline - time.monotonic())
                self.worker_intervals.append([execution_started, None])
                interval = self.worker_intervals[-1]
                try:
                    async with asyncio.timeout_at(min(task_deadline, route.deadline + 3)):
                        result = await launch_harness(self.config, runtime, route, self.base_url + "/agents/" + route.agent_id + "/v1", str(self.tasks[task_id].data.prompt) + ("\n\nCoordinator guidance:\n" + instructions if instructions else ""))
                    row.update(agent_exit_code=result.exit_code, agent_state="timed_out" if result.exit_code in (124, 137) else "completed" if result.exit_code == 0 else "error")
                    if result.exit_code not in (0, 124, 137):
                        row["agent_error"] = (result.stderr or result.stdout)[-2000:]
                except TimeoutError:
                    row["agent_state"] = "timed_out"
                finally:
                    interval[1] = time.monotonic()
                    row["agent_runtime_seconds"] = interval[1] - execution_started
                row["state"] = "scoring"
                grading_started = time.monotonic()
                self.event("worker_scoring", task_id=task_id, agent_state=row["agent_state"])
                async with asyncio.timeout_at(task_deadline):
                    reward = await self.tasks[task_id].solved(runtime)
                    # Harbor's convenience scorer returns zero for a missing
                    # reward file. Treat missing/malformed evidence as unscored.
                    reward_file = float((await runtime.read("/logs/verifier/reward.txt")).decode().strip())
                    if reward_file != reward:
                        raise RuntimeError("Verifier reward file disagrees with its returned score")
                row.update(state="completed", reward=reward, passed=reward >= 1, verifier_state="passed" if reward >= 1 else "failed")
                row["grading_seconds"] = time.monotonic() - grading_started
                await self.capture_verifier(runtime, route.agent_id)
        except asyncio.CancelledError:
            if row["reward"] is None:
                row.update(state="timed_out", verifier_state="unscored", error="Batch execution deadline or cancellation")
            else:
                row["log_collection_interrupted"] = True
            raise
        except TimeoutError:
            if row["reward"] is None:
                row.update(state="timed_out", verifier_state="unscored", error="Worker lifecycle deadline reached")
            else:
                row["log_collection_interrupted"] = True
        except Exception as error:
            row.update(state="error", verifier_state="error", error=scrub(str(error)))
        finally:
            row.update(ended_at=utc_now(), runtime_seconds=time.monotonic() - started)
            with contextlib.suppress(Exception):
                async with asyncio.timeout(3):
                    row.update(await runtime.inspect_resources())
            self.event("worker_finished", **row)
            await runtime.stop()
            self.write_result()

    async def capture_verifier(self, runtime, agent_id):
        directory = self.output / self.config.name / agent_id / "verifier"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("reward.txt", "ctrf.json"):
            with contextlib.suppress(Exception):
                async with asyncio.timeout(2):
                    (directory / name).write_bytes(await runtime.read("/logs/verifier/" + name))

    async def run(self, base_url):
        coordinator_started = time.monotonic()
        self.base_url = base_url
        route = self.route("orchestrator")
        runtime = self.runtime(route.agent_id)
        tasks = [{"task_id": task, "prompt": self.tasks[task].data.prompt} for task in self.config.task_ids]
        prompt = (
            "You coordinate an independent-task benchmark batch. Use the bench MCP tools. "
            "Immediately call delegate_tasks ONCE with assignments for ALL task_ids below, so their cheap workers start in parallel. "
            "Each worker uses the same harness as you with the configured cheap executor model, in its own isolated sandbox. "
            "You cannot solve tasks directly. Provide concise useful instructions if needed, without altering requirements. "
            "Then call wait_for_workers with wait_seconds=60 until every worker has a terminal state. "
            "Report only authoritative grader pass/fail or unscored outcomes; never claim success based on a worker's answer. "
            f"Do not retry or relaunch tasks. You have at most {max(0, self.deadline - time.monotonic()):.0f} seconds left in the absolute batch cap, including setup and scoring.\n"
            + json.dumps(tasks, indent=2)
        )
        self.event("coordinator_started", agent_id=route.agent_id, model=route.model.model)
        try:
            self.coordinator.update(state="setting_up", started_at=utc_now())
            async with asyncio.timeout_at(route.deadline):
                await runtime.start()
                await setup_harness(self.config, runtime, self.artifact)
            self.coordinator["state"] = "running"
            runtime.program_timeout = max(0.1, route.deadline - time.monotonic())
            async with asyncio.timeout_at(min(self.deadline, route.deadline + 3)):
                result = await launch_harness(self.config, runtime, route, base_url + "/agents/" + route.agent_id + "/v1", prompt, base_url + self.mcp_path + "/mcp")
            self.coordinator.update(state="completed" if result.exit_code == 0 else "timed_out" if result.exit_code in (124, 137) else "error", exit_code=result.exit_code)
            if result.exit_code:
                self.coordinator["error"] = (result.stderr or result.stdout)[-2000:]
        except asyncio.CancelledError:
            self.coordinator.update(state="timed_out", error="Overall comparison deadline or cancellation")
            for task in self.active.values():
                task.cancel()
            raise
        except TimeoutError:
            self.coordinator.update(state="timed_out", error="Coordinator execution deadline reached")
        except Exception as error:
            self.coordinator.update(state="error", error=scrub(str(error)))
        finally:
            self.coordinator["ended_at"] = utc_now()
            self.coordinator["runtime_seconds"] = time.monotonic() - coordinator_started
            with contextlib.suppress(Exception):
                async with asyncio.timeout(3):
                    self.coordinator.update(await runtime.inspect_resources())
            await runtime.stop()
            if self.active:
                remaining = self.deadline - time.monotonic()
                if remaining > 0:
                    done, pending = await asyncio.wait(self.active.values(), timeout=remaining)
                else:
                    pending = list(self.active.values())
                for task in pending:
                    task.cancel()
                await asyncio.gather(*self.active.values(), return_exceptions=True)
            self.event("coordinator_finished", **self.coordinator)
            self.finished_at = time.monotonic()
            self.write_result()

    def summary(self):
        intervals = [(start, end or time.monotonic()) for start, end in self.worker_intervals]
        edges = sorted([(start, 1) for start, end in intervals] + [(end, -1) for start, end in intervals])
        active = maximum = 0
        overlap = 0.0
        previous = None
        for timestamp, delta in edges:
            if previous is not None and active >= 2:
                overlap += timestamp - previous
            active += delta
            maximum = max(maximum, active)
            previous = timestamp
        main_usage = self.ledger.summarize(agent_id=self.coordinator["agent_id"])
        total_usage = self.ledger.summarize(team=self.config.name)
        passed = sum(row["passed"] is True for row in self.rows.values())
        return {
            "benchmark_type": "multiagent", "name": self.config.name,
            "execution_mode": self.config.execution_mode,
            "protocol": "MCP batch delegation; same pinned harness for coordinator and isolated workers; native harness subagents disabled",
            "harness": self.config.harness.model_dump(),
            "orchestrator": {**self.coordinator, "usage": main_usage},
            "tasks": [dict(row, usage=self.ledger.summarize(agent_id=self.config.name + "-" + task)) for task, row in self.rows.items()],
            "passed_tasks": passed, "task_count": len(self.rows), "pass_rate": passed / len(self.rows),
            "scored_tasks": sum(row.get("reward") is not None for row in self.rows.values()),
            "worker_spawn_count": self.spawns, "max_observed_parallel_workers": maximum, "worker_overlap_seconds": overlap,
            "coordinator_delegated_all_tasks": set(self.active) == set(self.rows),
            "runtime_seconds": (self.finished_at or time.monotonic()) - self.started,
            "limits": {"batch_seconds": self.config.batch_timeout_seconds, "worker_seconds": self.config.worker_timeout_seconds, "parallel_workers": self.config.max_parallel_workers, "max_cost_usd": self.config.max_cost_usd, "grading_reserve_seconds": self.config.grading_reserve_seconds},
            "usage": total_usage,
            "coordinator_cost_fraction": main_usage["provider_cost_usd"] / total_usage["provider_cost_usd"] if total_usage["provider_cost_usd"] else None,
        }

    def write_result(self):
        directory = self.output / self.config.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "result.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(scrub(json.dumps(self.summary(), indent=2)) + "\n")
        temporary.replace(path)


def write_report(output, results, elapsed, catalog, usage):
    manifest = json.loads((output / "manifest.json").read_text())
    overall = {"benchmark_type": "multiagent", "schema_version": 1, "runtime_seconds": elapsed, "teams": results, "model_catalog": catalog, "usage": usage}
    (output / "comparison.json").write_text(scrub(json.dumps(overall, indent=2)) + "\n")
    rows = []
    for result in results:
        name = html.escape(result["name"])
        usage = result["usage"]
        cost = f'${usage["total_cost_usd"]:.6f}' if usage["total_cost_usd"] is not None else f'≥ ${usage["provider_cost_usd"]:.6f} (incomplete)'
        rows.append(f'<tr><td>{name}</td><td>{result["passed_tasks"]}/{result["task_count"]}</td><td>{result["scored_tasks"]}</td><td>{cost}</td><td>{usage["total_tokens"]:,}</td><td>{result["max_observed_parallel_workers"]}</td><td>{result["worker_overlap_seconds"]:.1f}s</td><td>{html.escape(result["orchestrator"]["state"])}</td></tr>')
    agents = []
    for team in results:
        for agent in [team["orchestrator"], *team.get("workers", team["tasks"])]:
            usage = agent["usage"]
            agents.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in [team["name"], agent.get("agent_id", agent.get("task_id", "coordinator")), agent.get("model", "—"), agent["state"], agent.get("verifier_state", "—"), usage["model_call_count"], usage["input_tokens"], usage["output_tokens"], usage["reasoning_tokens"], f'${usage["provider_cost_usd"]:.6f}', usage["calls_missing_cost"]]) + '</tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multiagent harness pilot</title><style>body{font:15px/1.6 system-ui;margin:40px auto;padding:0 24px;max-width:1300px;color:#18394c;background:#fbfaf6}h1{font-size:36px}table{width:100%;border-collapse:collapse;background:white;margin:24px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #dde3e3}th{background:#edf2ef}p{max-width:1000px}.scroll{overflow:auto}a{color:#176582}</style><h1>Multiagent harness pilot</h1>'''
    page += f'<p>Comparison wall time: {elapsed:.1f}s. Independent hidden Harbor graders determine each task outcome. The complete comparison has a {manifest["overall_timeout_seconds"]:.0f}-second execution cap; models, execution modes and limits are recorded in the frozen configs. Cleanup follows.</p>'
    page += '<p>Shared-container protocol: one orchestrator and its helpers have terminal/file access to the same container and all task state. Helpers report work to the orchestrator; its final exit submits the whole batch. Agents stop before the original task graders run sequentially. The combined OS/dependency environment differs from standalone Terminal-Bench images. Legacy isolated-task runs are explicitly identified by their execution mode. Both modes use benchmark MCP delegation with real pinned harness CLIs; native harness subagents are disabled. All model calls use cancellable streams and per-agent accounting. This pilot does not establish a general best harness.</p>'
    environment = manifest.get("environment", {})
    if environment:
        page += '<p>Local execution environment: ' + html.escape(json.dumps(environment)) + '. Container architecture differing from host/Docker architecture uses emulation; this can affect runtime and completion.</p>'
    page += '<div class="scroll"><table><tr><th>Team</th><th>Passes</th><th>Scored</th><th>Total cost</th><th>Tokens</th><th>Parallel workers</th><th>Overlap</th><th>Coordinator</th></tr>' + ''.join(rows) + '</table></div>'
    page += '<h2>Per-agent accounting</h2><div class="scroll"><table><tr><th>Team</th><th>Task / role</th><th>Model</th><th>Execution</th><th>Grader</th><th>Calls</th><th>Input</th><th>Output</th><th>Reasoning</th><th>Provider cost</th><th>Calls missing cost</th></tr>' + ''.join(agents) + '</table></div>'
    grades = []
    for team in results:
        for task in team["tasks"]:
            grades.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in [team["name"], task["task_id"], task.get("verifier_state", "unscored"), task.get("reward"), task.get("error", "")]) + '</tr>')
    page += '<h2>Task grades</h2><div class="scroll"><table><tr><th>Team</th><th>Task</th><th>Grader</th><th>Reward</th><th>Error</th></tr>' + ''.join(grades) + '</table></div>'
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
    manifest = {"benchmark_type": "multiagent", "created_at": utc_now(), "overall_timeout_seconds": overall_timeout, "configs": [config.model_dump() for config in configs], "protocols": {config.name: config.execution_mode for config in configs}, "statistics_database": "multiagent.sqlite", "single_agent_statistics_imported": False}
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
    artifacts, tasksets, shared_images = {}, {}, {}
    manifest["images"] = {}
    for config in configs:
        artifacts[config.name] = await asyncio.to_thread(prepare_artifact, config)
        # The official hello-world smoke fixture's Dockerfile adds nothing our
        # pinned Python base image needs. Real benchmark Dockerfiles stay rejected.
        loaded = await asyncio.to_thread(lambda: list(HarborTaskset(HarborConfig(dataset=config.dataset, tasks=config.task_ids, ignore_dockerfile=config.dataset == "harbor/hello-world")).load()))
        tasksets[config.name] = {task.data.name.rsplit("/", 1)[-1]: task for task in loaded}
        if set(tasksets[config.name]) != set(config.task_ids):
            raise ValueError("Dataset did not resolve exactly the requested task IDs")
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
        if config.execution_mode == "shared_container":
            shared_images[config.name] = await prepare_shared_image(config, tasksets[config.name], manifest["images"], output)
            check = await docker("image", "inspect", shared_images[config.name])
            image_info = json.loads(check.stdout)[0]
            manifest["images"][shared_images[config.name]] = {field: image_info.get(field) for field in ("Id", "RepoDigests", "Architecture", "Os")}
    manifest["shared_images"] = shared_images
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
    from .shared import SharedTeam
    teams = []
    for config in configs:
        args = (config, output, ledger, gateway, artifacts[config.name], tasksets[config.name], min(deadline, started + config.batch_timeout_seconds))
        teams.append(SharedTeam(*args, shared_image=shared_images[config.name]) if config.execution_mode == "shared_container" else Team(*args))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            for team in teams:
                await stack.enter_async_context(team.mcp.session_manager.run())
            yield
    app = FastAPI(lifespan=lifespan)
    @app.post("/agents/{agent_id}/v1/{path:path}")
    async def relay(agent_id: str, path: str, request: Request):
        return await gateway.relay(agent_id, path, request)
    for team in teams:
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
                                 "tasks": {task: row["state"] for task, row in team.rows.items()},
                                 "workers": {agent: row["state"] for agent, row in getattr(team, "workers", team.rows).items()},
                                 "calls": usage["model_call_count"], "tokens": usage["total_tokens"], "provider_cost_usd": usage["provider_cost_usd"]})
            print(json.dumps({"event": "progress", "elapsed_seconds": round(time.monotonic() - started, 1), "teams": statuses}), flush=True)
    monitoring = asyncio.create_task(progress())
    try:
        while not server.started:
            if serving.done():
                await serving
                raise RuntimeError("Local delegation server did not start")
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
            for task in team.active.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*team.active.values(), return_exceptions=True)
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
        print(scrub(json.dumps({"event": "comparison_finished", "output": str(output), "runtime_seconds": elapsed, "teams": [{"name": result["name"], "passed_tasks": result["passed_tasks"], "task_count": result["task_count"], "max_parallel_workers": result["max_observed_parallel_workers"], "coordinator_state": result["orchestrator"]["state"], "usage": result["usage"]} for result in results]})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True, help="Fresh standalone multiagent JSON; repeat for concurrent teams")
    parser.add_argument("--output", type=Path, default=Path("outputs/multiagent") / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]))
    parser.add_argument("--overall-timeout-seconds", type=float, default=900)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.overall_timeout_seconds <= 1200:
        parser.error("Overall timeout must be positive and at most 1200 seconds")
    configs = [load_config(path) for path in args.config]
    if len({config.name for config in configs}) != len(configs):
        parser.error("Team names must be unique")
    if len({json.dumps({"tasks": config.task_ids, "dataset": config.dataset, "execution_mode": config.execution_mode, "orchestrator": config.orchestrator.model_dump(), "executor": config.executor.model_dump()}, sort_keys=True) for config in configs}) > 1:
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
