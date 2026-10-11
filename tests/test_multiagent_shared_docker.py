"""Opt-in local Docker integration checks. No model/API calls are made."""
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask

from harness_bloat_bench.multiagent import shared
from harness_bloat_bench.multiagent.accounting import Ledger
from harness_bloat_bench.multiagent.config import load_config


IMAGE = os.environ.get("MULTIAGENT_DOCKER_TEST_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="Set MULTIAGENT_DOCKER_TEST_IMAGE to the prepared shared image")
ROOT = Path(__file__).resolve().parents[1]


def test_shared_processes_services_and_sequential_harbor_graders(tmp_path, monkeypatch):
    async def scenario():
        config = load_config(ROOT / "configs/multiagent/hermes-sol-deepseek-four-task.json")
        ledger = Ledger(tmp_path)
        gateway = SimpleNamespace(routes={}, inflight={})
        gateway.register = lambda route: gateway.routes.update({route.agent_id: route})
        tasks = {}
        for index, task_id in enumerate(config.task_ids):
            directory = tmp_path / "fixtures" / task_id
            (directory / "tests").mkdir(parents=True)
            # Real Harbor staging/execution; tiny synthetic verifiers isolate
            # lifecycle behavior from the benchmark solutions and network setup.
            script = f'''#!/bin/bash
set -eu
test ! -e /logs/verifier/reward.txt
test "$(cat /tmp/shared-result)" = reviewed
kill -0 "$(cat /tmp/task-service.pid)"
if kill -0 "$(cat /tmp/unfinished-agent.pid)" 2>/dev/null; then exit 1; fi
echo {index % 2} > /logs/verifier/reward.txt
echo '{{"task": "{task_id}"}}' > /logs/verifier/ctrf.json
'''
            (directory / "tests/test.sh").write_text(script)
            tasks[task_id] = HarborTask(HarborData(idx=index, name=task_id, prompt="Synthetic lifecycle check", task_dir=str(directory)))
        team = shared.SharedTeam(config, tmp_path, ledger, gateway, None, tasks, time.monotonic() + 90, shared_image=IMAGE)
        team.config.grading_reserve_seconds = 45
        async def setup(config, runtime, artifact):
            result = await runtime.run(["mkdir", "-p", "/app/pypi", "/app/dclm"], {})
            assert result.exit_code == 0
        monkeypatch.setattr(shared, "setup_harness", setup)
        helper_running = asyncio.Event()
        async def launch(config, runtime, route, endpoint, prompt, mcp_url=None):
            if route.role == "orchestrator":
                delayed = await runtime.run_program(["sh", "-c", "sleep 0.2; echo waited; exit 7"], {})
                assert delayed.exit_code == 7 and delayed.stdout.strip() == "waited"
                probe = await team.start_service("exec sleep 120")
                assert (await team.stop_service(probe["service_id"]))["state"] == "stopped"
                # A real detached task service must survive helper and leader exit.
                service = await team.start_service("echo $$ > /tmp/task-service.pid; exec sleep 120")
                assert service["state"] == "started"
                result = await runtime.run_program(["sh", "-c", "echo initial > /tmp/shared-result"], {})
                assert result.exit_code == 0
                helper = await team.spawn_subagent("edit shared state", config.task_ids[0])
                await team.active[helper["agent_id"]]
                observed = await runtime.run(["cat", "/tmp/shared-result"], {})
                assert observed.stdout.strip() == "helper-change"
                await runtime.run(["sh", "-c", "echo reviewed > /tmp/shared-result"], {})
                await team.spawn_subagent("unfinished", config.task_ids[0])
                await helper_running.wait()
                # Wait for the real helper process to create its PID marker.
                async with asyncio.timeout(5):
                    while (await runtime.run(["test", "-f", "/tmp/unfinished-agent.pid"], {})).exit_code:
                        await asyncio.sleep(.02)
                return ProgramResult(0, "Submit batch", "")
            if "unfinished" in prompt:
                helper_running.set()
                # Simulate a native terminal tool that starts its own session.
                return await runtime.run_program(["sh", "-c", "setsid sh -c 'echo $$ > /tmp/unfinished-agent.pid; exec sleep 120' & wait"], {})
            return await runtime.run_program(["sh", "-c", "echo helper-change > /tmp/shared-result"], {})
        monkeypatch.setattr(shared, "launch_harness", launch)
        await asyncio.wait_for(team.run("http://unused.test"), 60)
        assert len(team.runtimes) == 1
        assert team.summary()["scored_tasks"] == 4
        assert team.summary()["passed_tasks"] == 2
        assert team.coordinator["state"] == "completed"
        for task_id in config.task_ids:
            directory = tmp_path / config.name / "tasks" / task_id / "verifier"
            assert json.loads((directory / "ctrf.json").read_text()) == {"task": task_id}
        ledger.db.close()
    asyncio.run(scenario())
