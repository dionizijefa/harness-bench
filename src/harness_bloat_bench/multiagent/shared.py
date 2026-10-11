"""One orchestrator and its helpers work on all tasks in one live container."""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.errors import SandboxError

from .accounting import scrub, utc_now
from .gateway import AgentRoute
from .harnesses import BenchRuntime, ContainerSession, launch_harness, setup_harness
from .runner import Team
from .shared_environment import TASK_WORKDIRS
from .vba_rubric import TASK_ID, rubric, score_results


class VerifierSession(ContainerSession):
    async def run(self, argv, env):
        if argv != ["sh", "-c", "cd /tests && bash test.sh"]:
            return await super().run(argv, env)
        # Preserve each task's working directory. In particular, Bottle's first
        # pytest invocation must discover its original repo tests, not /tests.
        if self.config.workdir == "/app" and self.task_id == "fix-code-vulnerability":
            env = {**env, "PYTHONPATH": "/app", "PYTEST_ADDOPTS": "--ignore=/app/dclm --ignore=/app/pypi"}
        result = await self._stream_program(["bash", "/tests/test.sh"], env, self.output / "verifier", self.program_timeout)
        (self.output / "verifier/process.json").write_text(json.dumps({"exit_code": result.exit_code}))
        return result


class SharedTeam(Team):
    def __init__(self, *args, shared_image: str, verifier_image: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_image = shared_image
        self.verifier_image = verifier_image
        self.is_vba = self.config.scoring == "vba_subtasks_v1"
        self.workdir = "/workspace" if self.is_vba else "/app"
        self.task_workdirs = {TASK_ID: self.workdir} if self.is_vba else TASK_WORKDIRS
        self.container = None
        self.sessions = []
        self.workers = {}
        self.services = {}
        self.service_lock = asyncio.Lock()
        self.accepting_workers = False
        self.phase = "planned"
        for row in self.rows.values():
            row.update(state="pending", verifier_state="unscored")
            if self.is_vba:
                row.update(score_results([]))
        self.mcp = FastMCP("Harness Bench", stateless_http=True, json_response=True, transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
        for tool in (self.spawn_subagent, self.wait_for_subagents, self.stop_subagent, self.start_service, self.stop_service, self.team_status):
            self.mcp.tool()(tool)

    @property
    def execution_deadline(self):
        return self.deadline - self.config.grading_reserve_seconds

    def create_container(self):
        sandbox = self.config.sandbox
        runtime = BenchRuntime(DockerConfig(image=self.shared_image, workdir=self.workdir, cpu=sandbox.shared_cpus, memory=sandbox.shared_memory_gb),
                               "mab-" + uuid4().hex[:12], sandbox.platform, self.output / self.config.name / "sandbox")
        self.runtimes.append(runtime)
        return runtime

    def session(self, agent_id, task_id=None):
        session = ContainerSession(self.container, self.output / self.config.name / agent_id, self.task_workdirs.get(task_id, self.workdir))
        self.sessions.append(session)
        return session

    async def spawn_subagent(self, instructions: str, task_id: str | None = None, subtask_id: str | None = None) -> dict:
        """Start a helper in this SAME container. Give concrete instructions, optionally associated with a task_id and rubric subtask_id. Files, services and changes are shared immediately. You may launch multiple helpers for a component or follow-up work. Returns an agent_id; collect its output with wait_for_subagents. Finishing a helper does NOT submit or grade the work."""
        if not self.accepting_workers or time.monotonic() >= self.execution_deadline:
            return {"error": "Agent execution is closed"}
        if self.is_vba and task_id is None:
            task_id = TASK_ID
        subtask_ids = [row["subtask_id"] for row in rubric()["subtasks"]] if self.is_vba else []
        if subtask_id is not None and subtask_id not in subtask_ids:
            return {"error": "Unknown subtask_id", "subtask_ids": subtask_ids}
        if task_id is not None and task_id not in self.tasks:
            return {"error": "Unknown task_id", "task_ids": self.config.task_ids}
        if not instructions.strip():
            return {"error": "Provide instructions for the helper"}
        self.spawns += 1
        agent_id = f"{self.config.name}-helper-{self.spawns}"
        self.workers[agent_id] = {"agent_id": agent_id, "task_id": task_id, "subtask_id": subtask_id, "role": "executor", "model": self.config.executor.model, "state": "queued", "instructions": instructions}
        self.event("worker_spawn", agent_id=agent_id, task_id=task_id, subtask_id=subtask_id, instructions=instructions)
        self.active[agent_id] = asyncio.create_task(self.worker(agent_id, instructions))
        return {"agent_id": agent_id, "task_id": task_id, "subtask_id": subtask_id, "state": "queued", "shared_container": True}

    async def wait_for_subagents(self, agent_ids: list[str] | None = None, wait_seconds: float = 60) -> dict:
        """Wait up to 60 seconds for any selected helper to finish and return its output/status. You may inspect or edit the shared workspace, then spawn follow-up helpers. Helpers do not run the hidden benchmark grader."""
        ids = list(self.active) if agent_ids is None else agent_ids
        if any(agent_id not in self.active for agent_id in ids):
            return {"error": "Unknown agent_id"}
        pending = [self.active[agent_id] for agent_id in ids if not self.active[agent_id].done()]
        if pending:
            await asyncio.wait(pending, timeout=max(0, min(wait_seconds, 60, self.execution_deadline - time.monotonic())), return_when=asyncio.FIRST_COMPLETED)
        return {"workers": [self.worker_status(agent_id) for agent_id in ids], "remaining_execution_seconds": max(0, self.execution_deadline - time.monotonic())}

    async def stop_subagent(self, agent_id: str) -> dict:
        """Stop a helper's agent process and foreground commands. Its shared file changes remain. Services started through start_service remain running. You can inspect the work or spawn a replacement."""
        task = self.active.get(agent_id)
        if task is None:
            return {"error": "Unknown agent_id"}
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return self.worker_status(agent_id)

    def worker_status(self, agent_id):
        return {**self.workers[agent_id], "usage": self.ledger.summarize(agent_id=agent_id)}

    async def start_service(self, command: str, workdir: str | None = None) -> dict:
        """Start a persistent task service outside agent process lifetimes. Supply its FOREGROUND shell command (no daemonize/setsid/&) and absolute workdir. VBA grading launches the packaged app separately; legacy grading uses live services. Returns a service_id and log_path; use terminal checks to verify readiness."""
        async with self.service_lock, asyncio.timeout(5):
            return await self._start_service(command, workdir or self.workdir)

    async def _start_service(self, command, workdir):
        if not self.accepting_workers or time.monotonic() >= self.execution_deadline:
            return {"error": "Agent execution is closed"}
        if not command.strip() or not workdir.startswith("/"):
            return {"error": "Provide a command and an absolute working directory"}
        service_id = "service-" + uuid4().hex[:12]
        marker = "/tmp/bench-" + service_id + ".pid"
        log = "/tmp/bench-" + service_id + ".log"
        session = ContainerSession(self.container, self.output / self.config.name / "services" / service_id, workdir)
        # run_background is a separate docker exec, not a child of any agent.
        await session.run_background(["setsid", "sh", "-c", 'echo $$ > "$1"; exec bash -c "$2"', "bench-service", marker, command], {}, log)
        while True:
            try:
                await session.read(marker)
                break
            except (SandboxError, OSError):
                await asyncio.sleep(.05)
        record = {"service_id": service_id, "command": command, "workdir": workdir, "log_path": log, "pid_path": marker, "state": "started"}
        self.services[service_id] = record
        self.event("service_started", **record)
        return dict(record)

    async def stop_service(self, service_id: str) -> dict:
        """Stop a task service started by start_service, for example before restarting it with a corrected command."""
        async with self.service_lock:
            return await self._stop_service(service_id)

    async def _stop_service(self, service_id):
        if not self.accepting_workers:
            return {"error": "Agent execution is closed"}
        record = self.services.get(service_id)
        if record is None:
            return {"error": "Unknown service_id"}
        await self.container.stop_process_group(record["pid_path"])
        record["state"] = "stopped"
        self.event("service_stopped", service_id=service_id)
        return dict(record)

    def team_status(self) -> dict:
        """Return helper outputs, batch phase and usage. Hidden grading happens only after the orchestrator finishes the entire batch."""
        return {"phase": self.phase, "tasks": list(self.rows.values()), "workers": [self.worker_status(agent_id) for agent_id in self.workers], "services": list(self.services.values()),
                "remaining_execution_seconds": max(0, self.execution_deadline - time.monotonic()), "usage": self.ledger.summarize(team=self.config.name)}

    async def worker(self, agent_id, instructions):
        row = self.workers[agent_id]
        runtime = self.session(agent_id, row["task_id"])
        started = time.monotonic()
        deadline = min(self.execution_deadline, started + self.config.worker_timeout_seconds)
        route = AgentRoute(self.config.name, agent_id, "executor", row["task_id"], self.config.executor, deadline, self.config.max_cost_usd)
        self.gateway.register(route)
        interval = None
        try:
            async with asyncio.timeout_at(deadline), self.semaphore:
                row.update(state="running", started_at=utc_now())
                interval = [time.monotonic(), None]
                self.worker_intervals.append(interval)
                runtime.program_timeout = max(.1, deadline - time.monotonic())
                prompt = ("You are a helper in the orchestrator's shared Docker container. All agents share files and processes. "
                          "Complete the assigned subtask and report your changes and checks. Coordinate file ownership through the orchestrator. "
                          + self.environment_prompt() +
                          "For persistent task services, return the foreground startup command and working directory to the orchestrator, "
                          "which can launch them with start_service. Harness-managed background commands may be stopped when you finish. "
                          "Do not inspect or modify hidden tests or verifier outputs.\n\n" + instructions)
                if row["task_id"]:
                    prompt += "\n\nFull task requirements:\n" + str(self.tasks[row["task_id"]].data.prompt)
                if self.is_vba:
                    prompt += "\n\nPartial-credit verification criteria:\n" + json.dumps(self.public_subtasks(), indent=2)
                    if row["subtask_id"]:
                        prompt += "\n\nAssigned rubric subtask: " + row["subtask_id"]
                result = await launch_harness(self.config, runtime, route, self.base_url + "/agents/" + agent_id + "/v1", prompt)
                row.update(state="completed" if result.exit_code == 0 else "timed_out" if result.exit_code in (124, 137) else "error",
                           exit_code=result.exit_code, output=scrub(result.stdout[-16000:]), stderr=scrub(result.stderr[-4000:]))
        except asyncio.CancelledError:
            row.update(state="stopped")
            raise
        except TimeoutError:
            row.update(state="timed_out")
        except Exception as error:
            row.update(state="error", error=scrub(str(error)))
        finally:
            route.deadline = time.monotonic()
            if interval:
                interval[1] = time.monotonic()
            row.update(ended_at=utc_now(), runtime_seconds=time.monotonic() - started)
            for name in ("stdout", "stderr"):
                path = runtime.output / (name + ".log")
                if path.exists():
                    row.setdefault("output" if name == "stdout" else name, scrub(path.read_text(errors="replace")[-16000:]))
            self.event("worker_finished", **row)
            self.write_result()

    async def close_agents(self):
        self.accepting_workers = False
        async with self.service_lock:
            pass  # Finish an already-admitted service operation before grading.
        for route in self.gateway.routes.values():
            if route.team == self.config.name:
                route.deadline = time.monotonic()
        for task in self.active.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.active.values(), return_exceptions=True)
        # Also catches a process group whose CLI failed before normal teardown.
        for session in self.sessions:
            await session.stop()
        pending = {task for call_id, task in self.gateway.inflight.items() if self.ledger.records[call_id]["team"] == self.config.name}
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for row in self.workers.values():
            if row["state"] == "queued":
                row.update(state="stopped", ended_at=utc_now())
        self.event("agents_stopped")

    async def grade_tasks(self):
        self.phase = "grading"
        if self.is_vba:
            from .vba import grade_vba
            await grade_vba(self)
            return
        for index, task_id in enumerate(self.config.task_ids):
            row = self.rows[task_id]
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                row.update(state="timed_out", verifier_state="unscored", error="Batch deadline reached before grading")
                continue
            # Share remaining grading time so an infrastructure hang cannot take
            # the entire reserve away from every later task.
            grade_deadline = time.monotonic() + remaining / (len(self.config.task_ids) - index)
            runtime = VerifierSession(self.container, self.output / self.config.name / "tasks" / task_id, self.task_workdirs[task_id])
            runtime.task_id = task_id
            runtime.program_timeout = max(.1, grade_deadline - time.monotonic())
            row.update(state="scoring", grading_started_at=utc_now())
            started = time.monotonic()
            try:
                async with asyncio.timeout_at(grade_deadline):
                    # No previous task's test files or reward may leak into this grade.
                    clean = await runtime.run(["sh", "-c", "rm -rf /tests /logs/verifier /tmp/tests.tgz && mkdir -p /tests /logs/verifier"], {})
                    if clean.exit_code:
                        raise RuntimeError("Cannot clear previous verifier state")
                    reward = await self.tasks[task_id].solved(runtime)
                    reward_file = float((await runtime.read("/logs/verifier/reward.txt")).decode().strip())
                    if not math.isfinite(reward_file) or reward_file != reward:
                        raise RuntimeError("Missing, non-finite or inconsistent verifier reward")
                    row.update(state="completed", reward=reward, passed=reward >= 1, verifier_state="passed" if reward >= 1 else "failed")
            except asyncio.CancelledError:
                if row["reward"] is None:
                    row.update(state="timed_out", verifier_state="unscored", error="Grading cancelled")
                raise
            except TimeoutError:
                row.update(state="timed_out", verifier_state="unscored", error="Task grading deadline reached")
            except Exception as error:
                row.update(state="error", verifier_state="error", error=scrub(str(error)))
            finally:
                # Ensure the grader has stopped before another task takes /tests.
                await runtime.stop()
                directory = runtime.output / "verifier"
                directory.mkdir(parents=True, exist_ok=True)
                for name in ("reward.txt", "ctrf.json", "original-repo-ctrf.json"):
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(1):
                            (directory / name).write_bytes(await runtime.read("/logs/verifier/" + name))
                row.update(grading_seconds=time.monotonic() - started, ended_at=utc_now())
                self.event("task_graded", **row)
                self.write_result()

    def public_subtasks(self):
        return [{key: row[key] for key in ("subtask_id", "title", "criteria", "weight")} for row in rubric()["subtasks"]]

    def environment_prompt(self):
        if self.is_vba:
            return ("Port /shared/legacy_app into /workspace/generated_app using React, FastAPI and SQLite. "
                    "Read the legacy README and exported VBA to establish shared API and data contracts before splitting work. "
                    "Each subtask earns independent partial credit; finishing every subtask earns 1.0. "
                    "Use migration_summary.json commands.backend for a standalone backend startup command: API checks run it "
                    "independently of the frontend. UI checks launch run.sh and require the full React app. "
                    "Both kinds of checks use freshly installed dependencies and the original legacy inputs in a separate verifier container. "
                    "Only /workspace/generated_app sources are transferred; installed caches and services are not transferred. "
                    "You may associate helpers with a rubric subtask using spawn_subagent(subtask_id=...). "
                    "The runner's remaining time overrides the task prompt's published time allowance. ")
        return ("Bottle lives at /app and uses the default Python. Use /app/pypi and /opt/bench-pypi/bin/python for PyPI. "
                "Exclude /app/dclm and /app/pypi when discovering Bottle repo tests with pytest. "
                "The Git task lives at /app/dclm; QEMU inputs are /app/alpine.iso and /app/alpine-disk.qcow2. ")

    def coordinator_prompt(self):
        tasks = [{"task_id": task, "working_directory": self.task_workdirs[task], "prompt": self.tasks[task].data.prompt} for task in self.config.task_ids]
        return ("You are the lead agent responsible for completing ALL tasks below in ONE shared Docker container. "
                "You have terminal and file tools and can inspect, edit and test every task yourself. "
                "Use spawn_subagent for useful subtasks, wait_for_subagents to read their outputs, and stop_subagent when needed. "
                "Helpers use the configured executor model and your same harness and container. You choose the division of work; "
                "you can review changes, correct them directly, or spawn more helpers with follow-up instructions. "
                "Coordinate ownership of overlapping files. A helper finishing does not finish the batch. "
                + self.environment_prompt() +
                "All files and running services are visible to all agents. Use start_service with a foreground command to launch "
                "persistent services for development. Legacy graders use those live services; VBA grading restarts the submitted app. "
                "Use stop_service to replace a service. Helpers can return startup commands for you to launch; "
                "do not rely on a helper's harness-managed background processes surviving its exit. Check service readiness using terminal tools. "
                "Do not inspect or modify hidden tests or verifier outputs. Run your own checks and review helper work. "
                "Your FINAL ANSWER submits all work: remaining helpers are stopped before hidden grading begins. "
                "Finish only when you are ready to submit the complete application or task batch. "
                "Do not claim a hidden verifier pass; the runner records those results after you finish. "
                f"You have {max(0, self.execution_deadline - time.monotonic()):.0f} seconds of agent time left; grading has a separate reserve inside the batch deadline.\n\n"
                + json.dumps(tasks, indent=2)
                + ("\n\nPartial-credit verification criteria:\n" + json.dumps(self.public_subtasks(), indent=2) if self.is_vba else ""))

    async def run(self, base_url):
        self.base_url = base_url
        self.container = self.create_container()
        route = self.route("orchestrator")
        ready = False
        self.phase = "setting_up"
        self.coordinator.update(state="setting_up", started_at=utc_now())
        self.event("coordinator_started", agent_id=route.agent_id, model=route.model.model)
        try:
            async with asyncio.timeout_at(self.execution_deadline):
                await self.container.start()
                await setup_harness(self.config, self.container, self.artifact)
                ready = True
                runtime = self.session(route.agent_id)
                runtime.program_timeout = max(.1, self.execution_deadline - time.monotonic())
                self.phase = "working"
                self.accepting_workers = True
                self.coordinator["state"] = "running"
                result = await launch_harness(self.config, runtime, route, base_url + "/agents/" + route.agent_id + "/v1",
                                              self.coordinator_prompt(), base_url + self.mcp_path + "/mcp")
                self.coordinator.update(state="completed" if result.exit_code == 0 else "timed_out" if result.exit_code in (124, 137) else "error", exit_code=result.exit_code)
                if result.exit_code:
                    self.coordinator["error"] = scrub((result.stderr or result.stdout)[-2000:])
        except asyncio.CancelledError:
            self.coordinator.update(state="timed_out", error="Batch cancelled")
            raise
        except TimeoutError:
            self.coordinator.update(state="timed_out", error="Agent execution deadline reached")
        except Exception as error:
            self.coordinator.update(state="error", error=scrub(str(error)))
        finally:
            self.coordinator.update(ended_at=utc_now(), runtime_seconds=time.monotonic() - self.started)
            try:
                await self.close_agents()
                # Do not start grading on external cancellation or after the cap.
                if ready and not asyncio.current_task().cancelling() and time.monotonic() < self.deadline:
                    await self.grade_tasks()
            finally:
                for row in self.rows.values():
                    if row["state"] in ("pending", "scoring"):
                        row.update(state="unscored", verifier_state="unscored", error="Batch ended before grading completed")
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(2):
                        self.coordinator.update(await self.container.inspect_resources())
                for service_id, record in self.services.items():
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(1):
                            result = await self.container.run(["tail", "-c", "64000", record["log_path"]], {})
                            directory = self.output / self.config.name / "services" / service_id
                            directory.mkdir(parents=True, exist_ok=True)
                            (directory / "stdout.log").write_text(scrub(result.stdout))
                await self.container.stop()
                self.phase = "finished"
                self.finished_at = time.monotonic()
                self.event("coordinator_finished", **self.coordinator)
                self.write_result()

    def summary(self):
        result = super().summary()
        result.update(protocol="shared_container", phase=self.phase, shared_image=self.shared_image, services=list(self.services.values()),
                      workers=[self.worker_status(agent_id) for agent_id in self.workers], tasks=list(self.rows.values()))
        # A helper may cover multiple tasks; costs belong to agents/team, not grades.
        result.pop("coordinator_delegated_all_tasks", None)
        if self.is_vba:
            row = self.rows[TASK_ID]
            result.update(scoring=self.config.scoring, dataset=self.config.dataset,
                          partial_reward=row["reward"], score_complete=row["score_complete"],
                          passed_subtasks=row["passed_subtasks"], scored_subtasks=row["scored_subtasks"], subtask_count=row["subtask_count"],
                          scored_checks=row["scored_checks"], check_count=row["check_count"],
                          scored_tasks=int(row["scored_checks"] > 0), verifier_image=self.verifier_image)
        return result
