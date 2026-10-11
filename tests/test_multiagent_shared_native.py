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
from verifiers.v1.runtimes import DockerConfig

from harness_bloat_bench.multiagent.accounting import Ledger
from harness_bloat_bench.multiagent.config import load_config
from harness_bloat_bench.multiagent.gateway import chat_sse
from harness_bloat_bench.multiagent.harnesses import BenchRuntime, configure_artifact_cache, prepare_artifact, setup_harness
from harness_bloat_bench.harnesses.hermes_agent import HERMES_PYTHON_DEPENDENCY_DIR, HermesAgentHarness, HermesAgentHarnessConfig
from harness_bloat_bench.multiagent.legacy_shared import LegacySharedTeam
from harness_bloat_bench.multiagent.vba_run import VbaRun


pytestmark = pytest.mark.skipif(os.environ.get("MULTIAGENT_NATIVE_SMOKE") != "1", reason="Set MULTIAGENT_NATIVE_SMOKE=1 for real Docker/CLI checks with a local scripted model")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not os.environ.get("MULTIAGENT_SHARED_TEST_IMAGE"), reason="Set MULTIAGENT_SHARED_TEST_IMAGE to check task Python isolation")
@pytest.mark.parametrize("setup_mode", ["multiagent", "singleagent"])
def test_hermes_setup_preserves_task_python_and_ssl(tmp_path, monkeypatch, setup_mode):
    async def scenario():
        config = load_config(ROOT / "configs/multiagent/legacy/hermes-sol-deepseek-four-task.json")
        configure_artifact_cache()
        artifact = await asyncio.to_thread(prepare_artifact, config)
        runtime = BenchRuntime(DockerConfig(image=os.environ["MULTIAGENT_SHARED_TEST_IMAGE"], workdir="/app", cpu=4, memory=5),
                               "mab-python-" + str(os.getpid()), "linux/amd64", tmp_path)
        probe = ["python", "-c", "import ssl,sys; print(sys.version, sys.prefix, ssl.OPENSSL_VERSION)"]
        try:
            await runtime.start()
            before = await runtime.run(probe, {})
            assert before.exit_code == 0, before.stderr
            if setup_mode == "multiagent":
                await setup_harness(config, runtime, artifact)
            else:
                monkeypatch.setattr("harness_bloat_bench.harnesses.hermes_agent.ensure_docker_built_tree", lambda **kwargs: artifact)
                await HermesAgentHarness(HermesAgentHarnessConfig(id="hermes_agent", version=config.harness.version)).setup(runtime)
            for env in ({}, {"LD_LIBRARY_PATH": HERMES_PYTHON_DEPENDENCY_DIR}):
                after = await runtime.run(probe, env)
                assert after.exit_code == 0, after.stderr
                assert after.stdout == before.stdout
        finally:
            await runtime.stop()
    asyncio.run(scenario())


@pytest.mark.parametrize("harness", ["hermes", "opencode", "omp_agent", "claude_code_agent", "codex_agent"])
def test_vba_uses_actual_native_subagents_without_mcp(tmp_path, harness, monkeypatch):
    async def scenario():
        config = load_config(ROOT / "configs/multiagent/hermes-sol-deepseek-vba-userform-port.json")
        versions = {"hermes": ("hermes_agent", "0.20.0"), "opencode": ("opencode", "1.18.1"),
                    "omp_agent": ("omp_agent", "17.2.10"), "claude_code_agent": ("claude_code_agent", "2.1.226"),
                    "codex_agent": ("codex_agent", "0.137.0")}
        config.harness.id, config.harness.version = versions[harness]
        config.grading_reserve_seconds = 30
        config.sandbox.shared_cpus = 2
        config.sandbox.shared_memory_gb = 2
        configure_artifact_cache()
        artifact = await asyncio.to_thread(prepare_artifact, config)
        ledger = Ledger(tmp_path)
        gateway = SimpleNamespace(routes={}, inflight={})
        gateway.register = lambda route: gateway.routes.update({route.agent_id: route})
        task_id = config.task_ids[0]
        tasks = {task_id: SimpleNamespace(data=SimpleNamespace(prompt="Write helper to /tmp/native-vba-result"))}
        team = VbaRun(config, tmp_path, ledger, gateway, artifact, tasks[task_id], time.monotonic() + 150,
                          agent_image="ubuntu:24.04", verifier_image="unused:test")
        assert not hasattr(team, "mcp")
        requests, counts, submitted = [], {}, []
        helper_done = asyncio.Event()
        async def grade(run):
            result = await team.container.run(["cat", "/tmp/native-vba-result"], {})
            assert result.exit_code == 0 and result.stdout == "helper"
            submitted.append(True)
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.grade_vba", grade)
        app = FastAPI()
        @app.post("/agents/{agent_id}/v1/{path:path}")
        async def model(agent_id: str, path: str, request: Request):
            body = await request.json()
            route = gateway.routes[agent_id]
            assert (request.headers.get("authorization") == "Bearer " + route.token
                    or request.headers.get("x-api-key") == route.token)
            if route.native_executor and (body.get("model") == route.native_executor_model_alias
                                         or request.headers.get("x-openai-subagent") == "collab_spawn"):
                agent_id = route.native_executor.agent_id
            requests.append({"agent_id": agent_id, "path": path, "headers": dict(request.headers), "body": body})
            if path == "messages/count_tokens":
                return {"input_tokens": 1}
            funcs = {}
            namespaces = {}
            for tool in body.get("tools", []):
                if tool.get("type") == "namespace":
                    for nested in tool.get("tools", []):
                        funcs[nested["name"]] = nested
                        namespaces[nested["name"]] = tool["name"]
                elif tool.get("type") == "function":
                    fn = tool.get("function", tool)
                    funcs[fn["name"]] = fn
                elif "name" in tool and "input_schema" in tool:
                    funcs[tool["name"]] = tool
            assert not any("mcp__bench" in name.lower() or "spawn_subagent" in name for name in funcs)
            stage = counts.get(agent_id, 0)
            if funcs:
                counts[agent_id] = stage + 1
            name, args = None, {}
            if funcs and agent_id.endswith("-main") and stage == 0:
                if harness == "hermes":
                    name, args = "delegate_task", {"goal": "Write helper to /tmp/native-vba-result", "background": False}
                elif harness == "opencode":
                    name, args = "task", {"description": "Write shared result", "prompt": "Write helper to /tmp/native-vba-result", "subagent_type": "general"}
                elif harness == "omp_agent":
                    name, args = "task", {"context": "Write the shared result", "tasks": [{"agent": "task", "task": "Write helper to /tmp/native-vba-result"}]}
                elif harness == "claude_code_agent":
                    name = "Agent" if "Agent" in funcs else "Task"
                    args = {"description": "Write shared result", "prompt": "Write helper to /tmp/native-vba-result", "subagent_type": "general-purpose"}
                else:
                    name, args = "spawn_agent", {"task_name": "write_result", "message": "Write helper to /tmp/native-vba-result"}
                assert name in funcs, sorted(funcs)
            elif funcs and agent_id.endswith("-native-executors") and stage == 0:
                choices = {"hermes": "terminal", "opencode": "bash", "omp_agent": "bash", "claude_code_agent": "Bash", "codex_agent": "exec_command"}
                name = choices[harness]
                assert name in funcs, sorted(funcs)
                args = {"cmd" if harness == "codex_agent" else "command": "printf helper > /tmp/native-vba-result"}
                if "description" in funcs[name].get("parameters", funcs[name].get("input_schema", {})).get("properties", {}):
                    args["description"] = "Write shared result"
            elif funcs and agent_id.endswith("-native-executors"):
                helper_done.set()
            elif funcs and agent_id.endswith("-main"):
                await asyncio.wait_for(helper_done.wait(), 30)
            identifier = "native-response-" + str(len(requests))
            if path == "messages":
                content = [{"type": "tool_use", "id": identifier, "name": name, "input": args}] if name else [{"type": "text", "text": "The work is ready."}]
                raw = {"id": identifier, "type": "message", "role": "assistant", "model": body["model"], "content": content,
                       "stop_reason": "tool_use" if name else "end_turn", "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
                async def events():
                    packets = [{"type": "message_start", "message": {**raw, "content": [], "stop_reason": None}}]
                    for i, block in enumerate(content):
                        packets.append({"type": "content_block_start", "index": i, "content_block": {**block, "input": {}} if name else {**block, "text": ""}})
                        if name:
                            packets.append({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}})
                        else:
                            packets.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}})
                        packets.append({"type": "content_block_stop", "index": i})
                    packets += [{"type": "message_delta", "delta": {"stop_reason": raw["stop_reason"], "stop_sequence": None}, "usage": {"output_tokens": 1}}, {"type": "message_stop"}]
                    for packet in packets:
                        yield "event: " + packet["type"] + "\ndata: " + json.dumps(packet) + "\n\n"
                return StreamingResponse(events(), media_type="text/event-stream") if body.get("stream") else raw
            if path == "responses":
                item = ({"type": "function_call", "id": identifier + "-item", "call_id": identifier, "name": name, "arguments": json.dumps(args), "status": "completed"}
                        if name else {"type": "message", "id": identifier + "-item", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "The work is ready.", "annotations": []}]})
                if name in namespaces:
                    item["namespace"] = namespaces[name]
                raw = {"id": identifier, "object": "response", "created_at": int(time.time()), "status": "completed", "model": body["model"], "output": [item],
                       "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
                async def events():
                    packets = [{"type": "response.created", "response": {**raw, "status": "in_progress", "output": []}},
                               {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": ""} if name else item}]
                    if name:
                        packets.append({"type": "response.function_call_arguments.delta", "item_id": item["id"], "output_index": 0, "delta": json.dumps(args)})
                    packets += [{"type": "response.output_item.done", "output_index": 0, "item": item}, {"type": "response.completed", "response": raw}]
                    for packet in packets:
                        yield "event: " + packet["type"] + "\ndata: " + json.dumps(packet) + "\n\n"
                return StreamingResponse(events(), media_type="text/event-stream") if body.get("stream") else raw
            message = {"role": "assistant", "content": "The work is ready."}
            if name:
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": identifier, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}
            raw = {"id": identifier, "created": int(time.time()), "model": body["model"],
                   "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if name else "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            return StreamingResponse(chat_sse(raw), media_type="text/event-stream") if body.get("stream") else raw
        sock = socket.socket()
        sock.bind(("0.0.0.0", 0))
        sock.listen(128)
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        server.capture_signals = contextlib.nullcontext
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(.02)
            await asyncio.wait_for(team.run(f"http://127.0.0.1:{sock.getsockname()[1]}"), 155)
            assert team.coordinator["state"] == "completed", team.coordinator
            assert submitted == [True]
            assert set(counts) == {config.name + "-main", config.name + "-native-executors"}
            assert not hasattr(team, "workers") and not hasattr(team, "services")
        finally:
            (tmp_path / "native-model-requests.json").write_text(json.dumps(requests, indent=2))
            server.should_exit = True
            await serving
            ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("harness", ["hermes", "opencode"])
def test_native_orchestrator_delegates_reviews_and_submits_shared_state(tmp_path, harness):
    async def scenario():
        config = load_config(ROOT / f"configs/multiagent/legacy/{harness}-sol-deepseek-four-task.json")
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
        team = LegacySharedTeam(config, tmp_path, ledger, gateway, artifact, tasks, time.monotonic() + 120, shared_image="ubuntu:24.04")
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
