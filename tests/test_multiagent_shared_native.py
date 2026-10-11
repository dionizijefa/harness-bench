"""Opt-in native CLI/MCP smoke with a local scripted model, never a paid provider."""
import asyncio
import contextlib
import json
import os
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from verifiers.v1.tasksets.harbor.taskset import HarborData, HarborTask

from harness_bloat_bench.multiagent.accounting import Ledger
from harness_bloat_bench.multiagent.config import load_config
from harness_bloat_bench.multiagent.gateway import chat_sse
from harness_bloat_bench.multiagent.harnesses import configure_artifact_cache, prepare_artifact
from harness_bloat_bench.multiagent.shared import SharedTeam


pytestmark = pytest.mark.skipif(os.environ.get("MULTIAGENT_NATIVE_SMOKE") != "1", reason="Set MULTIAGENT_NATIVE_SMOKE=1 for real Docker/CLI checks with a local scripted model")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("harness", ["hermes", "opencode"])
def test_native_orchestrator_delegates_reviews_and_submits_shared_state(tmp_path, harness):
    async def scenario():
        config = load_config(ROOT / f"configs/multiagent/{harness}-sol-deepseek-four-task.json")
        config.grading_reserve_seconds = 30
        configure_artifact_cache()
        artifact = await asyncio.to_thread(prepare_artifact, config)
        tasks = {}
        for index, task_id in enumerate(config.task_ids):
            directory = tmp_path / "fixtures" / task_id / "tests"
            directory.mkdir(parents=True)
            (directory / "test.sh").write_text('''#!/bin/bash
set -eu
test ! -e /logs/verifier/reward.txt
test "$(cat /tmp/native-shared-result)" = helper-reviewed
kill -0 "$(cat /tmp/native-service.pid)"
echo 1 > /logs/verifier/reward.txt
''')
            tasks[task_id] = HarborTask(HarborData(idx=index, name=task_id, prompt="Smoke fixture", task_dir=str(directory.parent)))
        ledger = Ledger(tmp_path)
        gateway = SimpleNamespace(routes={}, inflight={})
        gateway.register = lambda route: gateway.routes.update({route.agent_id: route})
        team = SharedTeam(config, tmp_path, ledger, gateway, artifact, tasks, time.monotonic() + 120, shared_image="ubuntu:24.04")
        counts = {}
        requests = []
        @asynccontextmanager
        async def lifespan(app):
            async with team.mcp.session_manager.run():
                yield
        app = FastAPI(lifespan=lifespan)
        app.mount(team.mcp_path, team.mcp.streamable_http_app())
        @app.post("/agents/{agent_id}/v1/chat/completions")
        async def model(agent_id: str, request: Request):
            body = await request.json()
            requests.append({"agent_id": agent_id, "body": body})
            funcs = [tool.get("function", tool) for tool in body.get("tools", [])]
            stage = counts.get(agent_id, 0)
            selected, args = None, {}
            if funcs:
                names = {fn["name"] for fn in funcs}
                if agent_id.endswith("-main"):
                    if stage == 0:
                        selected, args = "start_service", {"command": "mkdir -p /app/pypi /app/dclm; echo $$ > /tmp/native-service.pid; exec sleep 180"}
                    elif stage == 1:
                        # Use Bottle's existing /app directory in this synthetic image.
                        selected, args = "spawn_subagent", {"instructions": "Write helper to /tmp/native-shared-result", "task_id": "fix-code-vulnerability"}
                    elif stage == 2:
                        selected, args = "wait_for_subagents", {"wait_seconds": 1}
                        if not team.workers or not all(row["state"] == "completed" for row in team.workers.values()):
                            counts[agent_id] = stage - 1
                    elif stage == 3:
                        selected, args = "terminal", {"command": "test \"$(cat /tmp/native-shared-result)\" = helper && printf -- '-reviewed' >> /tmp/native-shared-result"}
                elif stage == 0:
                    selected, args = "terminal", {"command": "printf helper > /tmp/native-shared-result"}
                counts[agent_id] = counts.get(agent_id, 0) + 1
                if selected == "terminal":
                    selected = "terminal" if "terminal" in names else "bash"
                    fn = next(fn for fn in funcs if fn["name"] == selected)
                    if "description" in fn.get("parameters", {}).get("properties", {}):
                        args["description"] = "Check and update shared task state"
                elif selected:
                    if "tool_call" in names:
                        # Hermes exposes MCP tools through its deferred tool catalog.
                        args = {"name": "mcp__bench__" + selected, "arguments": args}
                        selected = "tool_call"
                    else:
                        selected = next(name for name in names if name.endswith("_" + selected))
            if selected:
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-" + str(len(requests)), "type": "function", "function": {"name": selected, "arguments": json.dumps(args)}}]}
                reason = "tool_calls"
            else:
                message, reason = {"role": "assistant", "content": "The shared work is ready."}, "stop"
            raw = {"id": "scripted-" + str(len(requests)), "created": int(time.time()), "model": body["model"], "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": reason}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            return StreamingResponse(chat_sse(raw), media_type="text/event-stream") if body.get("stream") else raw
        sock = socket.socket()
        sock.bind(("0.0.0.0", 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        server.capture_signals = contextlib.nullcontext
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(.02)
            await asyncio.wait_for(team.run(f"http://127.0.0.1:{port}"), 125)
            assert len(team.runtimes) == 1
            assert team.coordinator["state"] == "completed", team.coordinator
            assert team.summary()["passed_tasks"] == 4, team.summary()
            assert team.spawns == 1
            assert len(team.services) == 1
            assert next(iter(team.workers.values()))["state"] == "completed"
            assert len({request["agent_id"] for request in requests}) == 2
        finally:
            (tmp_path / "scripted-model-requests.json").write_text(json.dumps(requests, indent=2))
            server.should_exit = True
            await serving
            ledger.db.close()
    asyncio.run(scenario())
