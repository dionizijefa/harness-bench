"""Shared-container lifecycle, interactive delegation and grading isolation."""
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from verifiers.v1.runtimes import DockerConfig, ProgramResult

from harness_bloat_bench.multiagent import legacy_shared as shared
from harness_bloat_bench.multiagent.accounting import Ledger
from harness_bloat_bench.multiagent.config import load_config
from harness_bloat_bench.multiagent.gateway import AgentRoute
from harness_bloat_bench.multiagent.harnesses import BenchRuntime, ContainerSession, launch_harness
from harness_bloat_bench.multiagent.shared_environment import TASK_WORKDIRS, combined_dockerfile, prepare_shared_image


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multiagent/legacy/hermes-sol-deepseek-four-task.json"


class FakeContainer:
    def __init__(self):
        self.starts = self.stops = 0
        self.files = {}
        self.changes = {}
        self.services = set()
        self.trace = []

    async def start(self): self.starts += 1
    async def stop(self): self.stops += 1
    async def inspect_resources(self): return {"container_id": "one-shared-container"}


class FakeSession:
    def __init__(self, owner, output, workdir="/app"):
        self.owner, self.output = owner, output
        self.config = SimpleNamespace(workdir=workdir)
        self.stopped = False

    async def stop(self): self.stopped = True
    async def read(self, path): return self.owner.files[path]
    async def run(self, argv, env):
        assert "rm -rf /tests /logs/verifier" in argv[-1]
        self.owner.files.clear()
        self.owner.trace.append("reset")
        return ProgramResult(0, "", "")


def setup_team(tmp_path, monkeypatch, *, grade=None):
    config = load_config(CONFIG)
    ledger = Ledger(tmp_path)
    gateway = SimpleNamespace(routes={}, inflight={})
    gateway.register = lambda route: gateway.routes.update({route.agent_id: route})
    owner = FakeContainer()
    tasks = {}
    for task_id in config.task_ids:
        async def solved(runtime, task_id=task_id):
            assert runtime.owner is owner
            assert "/logs/verifier/reward.txt" not in owner.files
            assert all(task.done() for task in team.active.values())
            assert all(session.stopped for session in team.sessions)
            assert not team.accepting_workers
            owner.trace.append(task_id)
            reward = await grade(runtime, task_id) if grade else 1.0
            if reward is not None:
                owner.files["/logs/verifier/reward.txt"] = str(reward).encode()
                owner.files["/logs/verifier/ctrf.json"] = json.dumps({"task": task_id}).encode()
            return reward or 0.0
        tasks[task_id] = SimpleNamespace(data=SimpleNamespace(prompt="Requirements for " + task_id), solved=solved)
    team = shared.LegacySharedTeam(config, tmp_path, ledger, gateway, None, tasks, time.monotonic() + 900, shared_image="combined:test")
    def create():
        team.runtimes.append(owner)
        return owner
    team.create_container = create
    monkeypatch.setattr(shared, "ContainerSession", FakeSession)
    monkeypatch.setattr(shared, "VerifierSession", FakeSession)
    async def setup(*args): pass
    monkeypatch.setattr(shared, "setup_harness", setup)
    return team, owner, ledger


def test_one_orchestrator_can_inspect_helpers_and_delegate_followups_before_grading(tmp_path, monkeypatch):
    async def scenario():
        team, owner, ledger = setup_team(tmp_path, monkeypatch)
        running, release = set(), asyncio.Event()
        async def launch(config, runtime, route, endpoint, prompt, mcp_url=None):
            assert runtime.owner is owner
            assert not owner.trace, "No hidden grading while any agent is still working"
            if route.role == "orchestrator":
                assert mcp_url and "ONE shared Docker container" in prompt
                workers = [await team.spawn_subagent("first", task_id) for task_id in config.task_ids]
                while len(running) < 4:
                    await asyncio.sleep(0)
                assert len(team.runtimes) == 1
                release.set()
                await asyncio.gather(*team.active.values())
                status = await team.wait_for_subagents(wait_seconds=0)
                assert len(status["workers"]) == 4
                assert all(worker["output"] == "initial change" for worker in status["workers"])
                assert owner.changes[config.task_ids[0]] == "first"
                # The orchestrator edits a worker's files, then delegates more work.
                owner.changes[config.task_ids[0]] = "orchestrator correction"
                followup = await team.spawn_subagent("followup", config.task_ids[0])
                assert followup["agent_id"] != workers[0]["agent_id"]
                await team.active[followup["agent_id"]]
                return ProgramResult(0, "All tasks ready", "")
            if "followup" in prompt:
                assert owner.changes[route.task_id] == "orchestrator correction"
                owner.changes[route.task_id] = "reviewed"
            else:
                running.add(route.agent_id)
                await release.wait()
                owner.changes[route.task_id] = "first"
            owner.services.add("pypi-server")
            return ProgramResult(0, "initial change", "")
        monkeypatch.setattr(shared, "launch_harness", launch)
        await asyncio.wait_for(team.run("http://localhost"), 3)
        assert owner.starts == owner.stops == 1
        assert "pypi-server" in owner.services
        assert owner.changes[team.config.task_ids[0]] == "reviewed"
        assert owner.trace == [item for task in team.config.task_ids for item in ("reset", task)]
        result = team.summary()
        assert result["passed_tasks"] == 4 and result["pass_rate"] == 1
        assert result["worker_spawn_count"] == 5
        assert result["max_observed_parallel_workers"] == 4
        assert len(result["workers"]) == 5
        for task_id in team.config.task_ids:
            directory = tmp_path / team.config.name / "tasks" / task_id / "verifier"
            assert json.loads((directory / "ctrf.json").read_text()) == {"task": task_id}
        assert "error" in await team.spawn_subagent("too late")
        ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("exit_mode", ["final", "timeout"])
def test_submission_stops_helpers_and_grades_even_after_execution_timeout(tmp_path, monkeypatch, exit_mode):
    async def scenario():
        team, owner, ledger = setup_team(tmp_path, monkeypatch)
        helper_started = asyncio.Event()
        if exit_mode == "timeout":
            team.deadline = time.monotonic() + team.config.grading_reserve_seconds + .05
        async def launch(config, runtime, route, endpoint, prompt, mcp_url=None):
            if route.role == "orchestrator":
                await team.spawn_subagent("unfinished", config.task_ids[0])
                await helper_started.wait()
                if exit_mode == "timeout":
                    await asyncio.Event().wait()
                return ProgramResult(0, "Submit", "")
            helper_started.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(shared, "launch_harness", launch)
        await asyncio.wait_for(team.run("http://localhost"), 1)
        assert all(row["state"] == "stopped" for row in team.workers.values())
        assert team.summary()["passed_tasks"] == 4
        assert team.coordinator["state"] == ("completed" if exit_mode == "final" else "timed_out")
        assert owner.stops == 1
        ledger.db.close()
    asyncio.run(scenario())


def test_missing_reward_cannot_reuse_prior_grade_and_later_tasks_still_run(tmp_path, monkeypatch):
    async def scenario():
        async def grade(runtime, task_id):
            if task_id == "fix-code-vulnerability":
                return None
            return 0.0 if task_id == "qemu-startup" else 1.0
        team, owner, ledger = setup_team(tmp_path, monkeypatch, grade=grade)
        async def launch(*args): return ProgramResult(0, "Submit", "")
        monkeypatch.setattr(shared, "launch_harness", launch)
        await team.run("http://localhost")
        assert team.rows["fix-code-vulnerability"]["verifier_state"] == "error"
        assert team.rows["fix-code-vulnerability"]["reward"] is None
        assert team.rows["qemu-startup"]["passed"] is False
        assert team.summary()["pass_rate"] == .5
        assert team.summary()["scored_tasks"] == 3
        ledger.db.close()
    asyncio.run(scenario())


def test_cancellation_stops_one_container_without_starting_graders(tmp_path, monkeypatch):
    async def scenario():
        team, owner, ledger = setup_team(tmp_path, monkeypatch)
        started = asyncio.Event()
        async def launch(*args):
            started.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(shared, "launch_harness", launch)
        run = asyncio.create_task(team.run("http://localhost"))
        await started.wait()
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        assert owner.starts == owner.stops == 1
        assert not owner.trace
        assert all(row["verifier_state"] == "unscored" for row in team.rows.values())
        assert team.phase == "finished"
        ledger.db.close()
    asyncio.run(scenario())


def test_stalled_grader_leaves_time_for_later_tasks(tmp_path, monkeypatch):
    async def scenario():
        async def grade(runtime, task_id):
            if task_id == "pypi-server":
                await asyncio.Event().wait()
            return 1.0
        team, owner, ledger = setup_team(tmp_path, monkeypatch, grade=grade)
        team.config.grading_reserve_seconds = .4
        team.deadline = time.monotonic() + .6
        async def launch(*args): return ProgramResult(0, "Submit", "")
        monkeypatch.setattr(shared, "launch_harness", launch)
        await asyncio.wait_for(team.run("http://localhost"), 1)
        assert team.rows["pypi-server"]["verifier_state"] == "unscored"
        assert team.rows["pypi-server"]["state"] == "timed_out"
        assert team.summary()["passed_tasks"] == 3
        ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("harness", ["hermes", "opencode"])
def test_harness_enables_coordinator_terminal_and_uses_distinct_agent_state(tmp_path, harness):
    async def scenario():
        config = load_config(ROOT / f"configs/multiagent/legacy/{harness}-sol-deepseek-four-task.json")
        class Runtime:
            config = SimpleNamespace(workdir="/app/pypi")
            writes, calls = {}, []
            def host_url(self, url): return url
            async def write(self, path, data): self.writes[path] = json.loads(data)
            async def run_program(self, argv, env):
                self.calls.append((argv, env))
                return ProgramResult(0, "", "")
        runtime = Runtime()
        for agent_id in ("main", "helper-1"):
            route = AgentRoute("team", agent_id, "orchestrator", None, config.orchestrator, time.monotonic() + 10, None)
            await launch_harness(config, runtime, route, "http://relay", "work", "http://tools/mcp")
        assert len(runtime.writes) == 2
        assert all("/main/" in path or "/helper-1/" in path for path in runtime.writes)
        if harness == "opencode":
            assert all(value["permission"]["*"] == "allow" for value in runtime.writes.values())
            assert all(value["tools"]["task"] is False for value in runtime.writes.values())
        else:
            assert all("terminal,file,mcp-bench" in argv for argv, env in runtime.calls)
            assert all(value["terminal"]["cwd"] == "/app/pypi" for value in runtime.writes.values())
    asyncio.run(scenario())


def test_session_cannot_start_or_remove_container(tmp_path):
    async def scenario():
        owner = BenchRuntime(DockerConfig(image="test", workdir="/app"), "shared", "linux/amd64", tmp_path)
        session = ContainerSession(owner, tmp_path / "helper")
        with pytest.raises(RuntimeError, match="shared container"):
            await session.start()
        session.cleanup()
        await session.stop()
        assert session._container == "shared" and not session._stopped
    asyncio.run(scenario())


def test_original_grader_runs_from_task_workdir_and_preserves_script(tmp_path):
    async def scenario():
        owner = BenchRuntime(DockerConfig(image="test", workdir="/app"), "shared", "linux/amd64", tmp_path)
        runtime = shared.VerifierSession(owner, tmp_path)
        runtime.task_id = "fix-code-vulnerability"
        async def stream(argv, env, directory, timeout):
            directory.mkdir(parents=True, exist_ok=True)
            assert argv == ["bash", "/tests/test.sh"]
            assert runtime.config.workdir == "/app"
            assert env["PYTHONPATH"] == "/app"
            assert "--ignore=/app/dclm" in env["PYTEST_ADDOPTS"]
            return ProgramResult(0, "original repo and hidden tests ran", "")
        runtime._stream_program = stream
        await runtime.run(["sh", "-c", "cd /tests && bash test.sh"], {})
        assert json.loads((tmp_path / "verifier/process.json").read_text())["exit_code"] == 0
    asyncio.run(scenario())


def test_combined_image_keeps_original_paths_and_build_fingerprint_stable(tmp_path, monkeypatch):
    async def scenario():
        config = load_config(CONFIG)
        images = {task_id: "registry/" + task_id + "@sha256:abc" for task_id in TASK_WORKDIRS}
        recipe = combined_dockerfile(images)
        assert "COPY --from=git_input /app/dclm /app/dclm" in recipe
        assert "COPY --from=qemu_input /app/alpine.iso /app/alpine.iso" in recipe
        assert "/tests" not in recipe
        tasks = {task_id: SimpleNamespace(data=SimpleNamespace(image=ref)) for task_id, ref in images.items()}
        manifest = {ref: {"Id": "sha256:abc", "RepoDigests": [ref]} for ref in images.values()}
        async def docker(*args): return ProgramResult(0, "[]", "")
        monkeypatch.setattr("harness_bloat_bench.multiagent.shared_environment.docker", docker)
        first = await prepare_shared_image(config, tasks, manifest, tmp_path)
        manifest[first] = {"Id": "another-image"}
        assert await prepare_shared_image(config, tasks, manifest, tmp_path) == first
    asyncio.run(scenario())
