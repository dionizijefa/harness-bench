"""Legacy Terminal-Bench 2.1 orchestration through benchmark MCP tools."""
from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from verifiers.v1.runtimes import DockerConfig

from .accounting import scrub, utc_now
from .gateway import AgentRoute
from .harnesses import BenchRuntime, launch_harness, setup_harness


class Team:
    def __init__(self, config, output, ledger, gateway, artifact, tasks, deadline):
        if config.scoring == "vba_subtasks_v1":
            raise ValueError("Native VBA runs do not use legacy Team orchestration")
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
            "dataset": self.config.dataset, "scoring": self.config.scoring,
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
