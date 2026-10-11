"""One native harness invocation, followed by grading its VBA app submission."""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from uuid import uuid4

from verifiers.v1.runtimes import DockerConfig

from .accounting import scrub, utc_now
from .gateway import AgentRoute
from .harnesses import BenchRuntime, ContainerSession, launch_harness, setup_harness
from .vba import grade_vba
from .vba_rubric import TASK_ID, score_results


class VbaRun:
    def __init__(self, config, output, ledger, gateway, artifact, task, deadline, *, agent_image, verifier_image):
        self.config, self.output, self.ledger, self.gateway = config, output, ledger, gateway
        self.artifact, self.task, self.deadline = artifact, task, deadline
        self.agent_image, self.verifier_image = agent_image, verifier_image
        self.started = time.monotonic()
        self.finished_at = None
        self.phase = "planned"
        self.container = None
        self.runtimes = []
        self.task_result = {"task_id": TASK_ID, "state": "pending", "passed": None,
                            "verifier_state": "unscored", **score_results([])}
        self.coordinator = {"agent_id": config.name + "-main", "role": "orchestrator",
                            "state": "planned", "model": config.orchestrator.model}

    @property
    def execution_deadline(self):
        return self.deadline - self.config.grading_reserve_seconds

    def create_container(self):
        sandbox = self.config.sandbox
        runtime = BenchRuntime(DockerConfig(image=self.agent_image, workdir="/workspace",
                                            cpu=sandbox.shared_cpus, memory=sandbox.shared_memory_gb),
                               "mab-" + uuid4().hex[:12], sandbox.platform,
                               self.output / self.config.name / "sandbox")
        self.runtimes.append(runtime)
        return runtime

    def coordinator_prompt(self):
        return (f"You have {max(0, self.execution_deadline - time.monotonic()):.0f} seconds to complete the task below. "
                "This time limit overrides any time allowance in the task. "
                "You may use the harness's native delegation to spawn as many subagents as needed within the shared resources "
                f"({self.config.sandbox.shared_cpus} CPUs, {self.config.sandbox.shared_memory_gb} GiB RAM) to finish in time. "
                "All agents work in ONE shared Docker container. "
                "Your FINAL ANSWER submits all work: remaining helpers are stopped before hidden grading begins. "
                "Do not claim a hidden verifier pass; the runner records those results after you finish.\n\n"
                + str(self.task.data.prompt))

    def event(self, event, **fields):
        self.ledger.event(event, team=self.config.name, **fields)

    async def run(self, base_url):
        self.container = self.create_container()
        executor = AgentRoute(self.config.name, self.config.name + "-native-executors", "executor", TASK_ID,
                              self.config.executor, self.execution_deadline, self.config.max_cost_usd,
                              aggregate_native_workers=True)
        main = AgentRoute(self.config.name, self.coordinator["agent_id"], "orchestrator", TASK_ID,
                          self.config.orchestrator, self.execution_deadline, self.config.max_cost_usd,
                          native_executor=executor)
        self.gateway.register(main)
        self.gateway.register(executor)
        session = None
        ready = cancelled = False
        self.phase = "setting_up"
        self.coordinator.update(state="setting_up", started_at=utc_now())
        self.event("coordinator_started", agent_id=main.agent_id, model=main.model.model)
        try:
            async with asyncio.timeout_at(self.execution_deadline):
                await self.container.start()
                await setup_harness(self.config, self.container, self.artifact)
                ready = True
                session = ContainerSession(self.container, self.output / self.config.name / main.agent_id, "/workspace")
                session.program_timeout = max(.1, self.execution_deadline - time.monotonic())
                self.phase = "working"
                self.coordinator["state"] = "running"
                result = await launch_harness(self.config, session, main, base_url + "/agents/" + main.agent_id + "/v1",
                                              self.coordinator_prompt())
                self.coordinator.update(state="completed" if result.exit_code == 0 else "timed_out" if result.exit_code in (124, 137) else "error",
                                        exit_code=result.exit_code)
                if result.exit_code:
                    self.coordinator["error"] = scrub((result.stderr or result.stdout)[-2000:])
        except asyncio.CancelledError:
            cancelled = True
            self.coordinator.update(state="timed_out", error="Run cancelled")
            raise
        except TimeoutError:
            self.coordinator.update(state="timed_out", error="Agent execution deadline reached")
        except Exception as error:
            self.coordinator.update(state="error", error=scrub(str(error)))
        finally:
            self.coordinator.update(ended_at=utc_now(), runtime_seconds=time.monotonic() - self.started)
            try:
                main.deadline = executor.deadline = time.monotonic()
                pending = {task for call_id, task in self.gateway.inflight.items()
                           if self.ledger.records[call_id]["team"] == self.config.name}
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if session is not None:
                    await session.stop()
                if ready and not cancelled and not asyncio.current_task().cancelling() and time.monotonic() < self.deadline:
                    self.phase = "grading"
                    await grade_vba(self)
            finally:
                if self.task_result["state"] in ("pending", "scoring"):
                    self.task_result.update(state="unscored", verifier_state="unscored", error="Run ended before grading completed")
                with contextlib.suppress(Exception):
                    await self.container.stop()
                self.phase = "finished"
                self.finished_at = time.monotonic()
                self.event("coordinator_finished", **self.coordinator)
                self.write_result()

    def summary(self):
        row = self.task_result
        main_usage = self.ledger.summarize(agent_id=self.coordinator["agent_id"])
        total_usage = self.ledger.summarize(team=self.config.name)
        pool_id = self.config.name + "-native-executors"
        return {
            "benchmark_type": "multiagent", "name": self.config.name, "protocol": "native_harness",
            "execution_mode": self.config.execution_mode, "phase": self.phase,
            "dataset": self.config.dataset, "scoring": self.config.scoring,
            "harness": self.config.harness.model_dump(), "agent_image": self.agent_image, "verifier_image": self.verifier_image,
            "orchestrator": {**self.coordinator, "usage": main_usage},
            "executor": {"agent_id": pool_id, "role": "executor", "model": self.config.executor.model,
                         "accounting": "aggregate_native_executor_pool", "state": "unmeasured",
                         "usage": self.ledger.summarize(agent_id=pool_id)},
            "tasks": [dict(row)], "task_count": 1, "passed_tasks": int(row["passed"] is True),
            "pass_rate": int(row["passed"] is True), "scored_tasks": int(row["scored_checks"] > 0),
            "partial_reward": row["reward"], "score_complete": row["score_complete"],
            **{key: row[key] for key in ("passed_subtasks", "scored_subtasks", "subtask_count", "scored_checks", "check_count")},
            "runtime_seconds": (self.finished_at or time.monotonic()) - self.started,
            "limits": {"batch_seconds": self.config.batch_timeout_seconds, "max_cost_usd": self.config.max_cost_usd,
                       "grading_reserve_seconds": self.config.grading_reserve_seconds},
            "usage": total_usage,
            "coordinator_cost_fraction": main_usage["provider_cost_usd"] / total_usage["provider_cost_usd"] if total_usage["provider_cost_usd"] else None,
        }

    def write_result(self):
        path = self.output / self.config.name / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(scrub(json.dumps(self.summary(), indent=2)) + "\n")
        temporary.replace(path)
