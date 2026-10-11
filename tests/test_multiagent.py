import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from harness_bloat_bench.multiagent.accounting import Ledger, normalized_usage
from harness_bloat_bench.multiagent.config import ModelSpec, load_config
from harness_bloat_bench.multiagent.gateway import AgentRoute, Gateway, chat_to_responses, responses_to_chat
from harness_bloat_bench.multiagent.legacy_team import Team
from verifiers.v1.runtimes import ProgramResult


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multiagent/legacy/hermes-sol-deepseek-two-task.json"


def test_provider_cost_is_not_estimated_or_reasoning_double_counted():
    usage = normalized_usage({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.007, "prompt_tokens_details": {"cached_tokens": 80}, "completion_tokens_details": {"reasoning_tokens": 12}}, {})
    assert usage["cost_usd"] == 0.007
    assert usage["estimated_cost_usd"] is None
    assert usage["total_tokens"] == 120
    assert usage["reasoning_tokens"] == 12


def test_missing_usage_survives_partial_failure_and_sqlite(tmp_path):
    ledger = Ledger(tmp_path)
    good = ledger.start_call(team="a", agent_id="main", role="orchestrator")
    ledger.finish_call(good, status="completed", total_tokens=20, cost_usd=0.1)
    bad = ledger.start_call(team="a", agent_id="worker", role="executor", task_id="t")
    ledger.finish_call(bad, status="error", **normalized_usage(None, {}))
    summary = ledger.summarize(team="a")
    assert summary["provider_cost_usd"] == 0.1
    assert summary["total_cost_usd"] is None
    assert summary["calls_missing_usage"] == 1
    assert summary["model_call_count"] == 2
    assert json.loads(ledger.db.execute("SELECT record FROM calls WHERE call_id=?", (bad,)).fetchone()[0])["usage_source"] == "missing"
    ledger.db.close()


def test_price_estimate_respects_cache_and_context_tier():
    usage = normalized_usage({"input_tokens": 300, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 100}}, {"prompt": "0.01", "completion": "0.02", "input_cache_read": "0.001", "overrides": [{"min_prompt_tokens": 250, "prompt": "0.02", "completion": "0.03"}]})
    assert usage["cost_usd"] is None
    assert usage["estimated_cost_usd"] == pytest.approx(4.4)


def test_gpt_tool_calls_use_responses_with_prior_reasoning():
    model = ModelSpec(model="openai/gpt-6.1-sol", reasoning_effort="high", api="responses")
    previous = [{"type": "reasoning", "id": "r"}, {"type": "function_call", "call_id": "c", "name": "delegate_tasks", "arguments": "{}"}]
    body = {"messages": [{"role": "user", "content": "run"}, {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "delegate_tasks", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c", "content": "started"}], "tools": [{"type": "function", "function": {"name": "delegate_tasks", "parameters": {"type": "object"}}}]}
    request = chat_to_responses(body, model, {"c": previous})
    assert request["reasoning"] == {"effort": "high"}
    assert request["input"][1:3] == previous
    assert request["input"][-1]["type"] == "function_call_output"
    assert request["tools"][0]["name"] == "delegate_tasks"
    converted = responses_to_chat({"id": "g", "model": model.model, "output": previous, "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8, "cost": 0.00001}})
    assert converted["choices"][0]["finish_reason"] == "tool_calls"
    assert converted["usage"]["cost"] == 0.00001


@pytest.mark.parametrize("detail", [None, "low", "high"])
def test_chat_vision_content_is_converted_for_responses(detail):
    model = ModelSpec(model="openai/gpt-6.1-sol", reasoning_effort="high", api="responses")
    image = {"url": "data:image/png;base64,screenshot"}
    if detail is not None:
        image["detail"] = detail
    content = [{"type": "text", "text": "Describe this form"}, {"type": "image_url", "image_url": image}]
    body = {"messages": [{"role": "user", "content": content},
                         {"role": "tool", "tool_call_id": "screenshot", "content": content}]}
    request = chat_to_responses(body, model)
    expected = [{"type": "input_text", "text": "Describe this form"},
                {"type": "input_image", "image_url": image["url"], "detail": detail or "auto"}]
    assert request["input"][0]["content"] == expected
    assert request["input"][1]["output"] == expected
    assert body["messages"][0]["content"][1] == {"type": "image_url", "image_url": image}


def test_comparison_configs_are_fresh_and_matching():
    hermes = load_config(CONFIG)
    opencode = load_config(ROOT / "configs/multiagent/legacy/opencode-sol-deepseek-two-task.json")
    assert hermes.task_ids == opencode.task_ids
    assert hermes.orchestrator == opencode.orchestrator
    assert hermes.executor == opencode.executor
    assert hermes.harness != opencode.harness
    assert hermes.batch_timeout_seconds == hermes.worker_timeout_seconds == 1200
    assert hermes.benchmark_type == "multiagent"
    with pytest.raises(ValueError):
        hermes.model_validate({**hermes.model_dump(), "campaign_id": "single-agent"})


def test_delegation_is_real_parallel_and_idempotent(tmp_path):
    async def scenario():
        config = load_config(CONFIG)
        ledger = Ledger(tmp_path)
        team = Team(config, tmp_path, ledger, None, None, {}, time.monotonic() + 1200)
        gate = asyncio.Event()
        active, maximum = 0, 0
        async def worker(task_id, instructions):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await gate.wait()
            active -= 1
        team.worker = worker
        assignments = [{"task_id": task} for task in config.task_ids]
        result = await team.delegate_tasks(assignments)
        await asyncio.sleep(0)
        assert len(result["launched"]) == 2
        assert maximum == 2
        assert (await team.delegate_tasks(assignments))["launched"] == []
        assert team.spawns == 2
        assert "error" in await team.delegate_tasks([{"task_id": "unknown"}])
        gate.set()
        await asyncio.gather(*team.active.values())
        team.deadline = time.monotonic() - 1
        assert "error" in await team.delegate_tasks(assignments)
        ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("exit_code,reward", [(0, 0.0), (124, 1.0)])
def test_hidden_grader_controls_outcome_even_after_agent_timeout(tmp_path, monkeypatch, exit_code, reward):
    async def scenario():
        config = load_config(CONFIG)
        ledger = Ledger(tmp_path)
        gateway = SimpleNamespace(register=lambda route: None)
        async def solved(runtime):
            runtime.graded = True
            return reward
        task = SimpleNamespace(data=SimpleNamespace(prompt="do work"), solved=solved)
        team = Team(config, tmp_path, ledger, gateway, None, {config.task_ids[0]: task}, time.monotonic() + 1200)
        team.base_url = "http://localhost"
        class Runtime:
            stopped = graded = False
            async def start(self): pass
            async def stop(self): self.stopped = True
            async def inspect_resources(self): return {}
            async def read(self, path): return str(reward).encode()
        runtime = Runtime()
        team.runtime = lambda *args: runtime
        async def setup(*args): pass
        async def launch(*args): return ProgramResult(exit_code, "I succeeded!", "")
        monkeypatch.setattr("harness_bloat_bench.multiagent.legacy_team.setup_harness", setup)
        monkeypatch.setattr("harness_bloat_bench.multiagent.legacy_team.launch_harness", launch)
        await team.worker(config.task_ids[0], "")
        row = team.rows[config.task_ids[0]]
        assert row["passed"] is (reward >= 1)
        assert row["agent_state"] == ("timed_out" if exit_code == 124 else "completed")
        assert runtime.graded and runtime.stopped
        ledger.db.close()
    asyncio.run(scenario())


def test_lifecycle_deadline_cleans_sandbox_and_marks_unscored(tmp_path, monkeypatch):
    async def scenario():
        config = load_config(CONFIG)
        config.worker_timeout_seconds = 0.05
        config.grading_reserve_seconds = 0.01
        ledger = Ledger(tmp_path)
        team = Team(config, tmp_path, ledger, SimpleNamespace(register=lambda route: None), None, {}, time.monotonic() + 1200)
        team.base_url = "http://localhost"
        class Runtime:
            stopped = False
            async def start(self): await asyncio.sleep(10)
            async def stop(self): self.stopped = True
            async def inspect_resources(self): return {}
        runtime = Runtime()
        team.runtime = lambda *args: runtime
        started = time.monotonic()
        await team.worker(config.task_ids[0], "")
        assert time.monotonic() - started < 1
        assert runtime.stopped
        assert team.rows[config.task_ids[0]]["state"] == "timed_out"
        assert team.rows[config.task_ids[0]]["passed"] is None
        assert team.rows[config.task_ids[0]]["verifier_state"] == "unscored"
        ledger.db.close()
    asyncio.run(scenario())


def test_queued_worker_has_the_same_absolute_lifecycle_deadline(tmp_path):
    async def scenario():
        config = load_config(CONFIG)
        config.worker_timeout_seconds = 0.05
        config.grading_reserve_seconds = 0.01
        ledger = Ledger(tmp_path)
        team = Team(config, tmp_path, ledger, SimpleNamespace(register=lambda route: None), None, {}, time.monotonic() + 1200)
        team.base_url = "http://localhost"
        class Runtime:
            started = stopped = False
            async def start(self): self.started = True
            async def stop(self): self.stopped = True
            async def inspect_resources(self): return {}
        runtime = Runtime()
        team.runtime = lambda *args: runtime
        await team.semaphore.acquire()
        await team.semaphore.acquire()
        started = time.monotonic()
        await asyncio.wait_for(team.worker(config.task_ids[0], ""), 0.5)
        assert time.monotonic() - started < 0.5
        assert not runtime.started and runtime.stopped
        assert team.rows[config.task_ids[0]]["state"] == "timed_out"
        assert team.rows[config.task_ids[0]]["passed"] is None
        ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_verified_outcome_survives_cancellation_during_log_collection(tmp_path, monkeypatch, reward):
    async def scenario():
        config = load_config(CONFIG)
        ledger = Ledger(tmp_path)
        async def solved(runtime): return reward
        task = SimpleNamespace(data=SimpleNamespace(prompt="work"), solved=solved)
        team = Team(config, tmp_path, ledger, SimpleNamespace(register=lambda route: None), None, {config.task_ids[0]: task}, time.monotonic() + 1200)
        team.base_url = "http://localhost"
        class Runtime:
            stopped = False
            async def start(self): pass
            async def stop(self): self.stopped = True
            async def inspect_resources(self): return {}
            async def read(self, path): return str(reward).encode()
        runtime = Runtime()
        team.runtime = lambda *args: runtime
        async def setup(*args): pass
        async def launch(*args): return ProgramResult(0, "done", "")
        collecting = asyncio.Event()
        async def capture(*args):
            collecting.set()
            await asyncio.Event().wait()
        team.capture_verifier = capture
        monkeypatch.setattr("harness_bloat_bench.multiagent.legacy_team.setup_harness", setup)
        monkeypatch.setattr("harness_bloat_bench.multiagent.legacy_team.launch_harness", launch)
        worker = asyncio.create_task(team.worker(config.task_ids[0], ""))
        await collecting.wait()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        row = team.rows[config.task_ids[0]]
        assert row["state"] == "completed"
        assert row["reward"] == reward
        assert row["passed"] is (reward >= 1)
        assert row["verifier_state"] == ("passed" if reward >= 1 else "failed")
        assert row["log_collection_interrupted"] is True
        assert runtime.stopped
        ledger.db.close()
    asyncio.run(scenario())


def test_concurrent_calls_reserve_cost_before_provider_request(tmp_path):
    async def scenario():
        ledger = Ledger(tmp_path)
        gateway = Gateway("unused-test-key", ledger, {"test": {"pricing": {"prompt": "0", "completion": "0.1"}}}, base_url="https://provider.test")
        await gateway.http.aclose()
        sent, release = asyncio.Event(), asyncio.Event()
        async def provider(request):
            sent.set()
            await release.wait()
            return httpx.Response(200, json={"id": "g", "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3, "cost": 0.1}})
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        spec = ModelSpec(model="test", reasoning_effort="medium", max_output_tokens=64)
        route = AgentRoute("team", "agent", "executor", "task", spec, time.monotonic() + 30, 10)
        gateway.register(route)
        app = FastAPI()
        @app.post("/")
        async def relay(request: Request):
            return await gateway.relay("agent", "chat/completions", request)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local.test", headers={"Authorization": "Bearer " + route.token}) as client:
            body = {"messages": [{"role": "user", "content": "test"}]}
            first = asyncio.create_task(client.post("/", json=body))
            await sent.wait()
            blocked = await client.post("/", json=body)
            assert blocked.status_code == 429
            assert len(ledger.records) == 1
            release.set()
            assert (await first).status_code == 200
            assert ledger.summarize()["total_cost_usd"] == 0.1
        await gateway.close()
        ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("native_pool", [False, True])
def test_native_executor_pool_does_not_share_one_helpers_call_limit(tmp_path, native_pool):
    async def scenario():
        ledger = Ledger(tmp_path)
        gateway = Gateway("unused-test-key", ledger, {"test": {"pricing": {"prompt": "0", "completion": "0"}}})
        await gateway.http.aclose()
        async def provider(request):
            return httpx.Response(200, json={"id": "g", "model": "test", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0}})
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        spec = ModelSpec(model="test", reasoning_effort="medium", max_calls=1)
        route = AgentRoute("team", "executor", "executor", None, spec, time.monotonic() + 30, None,
                           aggregate_native_workers=native_pool)
        gateway.register(route)
        app = FastAPI()
        @app.post("/")
        async def relay(request: Request):
            return await gateway.relay("executor", "chat/completions", request)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local.test",
                                    headers={"Authorization": "Bearer " + route.token}) as client:
            body = {"messages": [{"role": "user", "content": "test"}]}
            assert (await client.post("/", json=body)).status_code == 200
            assert (await client.post("/", json=body)).status_code == (200 if native_pool else 429)
        assert ledger.summarize()["model_call_count"] == (2 if native_pool else 1)
        await gateway.close()
        ledger.db.close()
    asyncio.run(scenario())


def test_missing_reward_is_an_infrastructure_error(tmp_path, monkeypatch):
    async def scenario():
        config = load_config(CONFIG)
        ledger = Ledger(tmp_path)
        async def solved(runtime): return 0.0
        task = SimpleNamespace(data=SimpleNamespace(prompt="work"), solved=solved)
        team = Team(config, tmp_path, ledger, SimpleNamespace(register=lambda route: None), None, {config.task_ids[0]: task}, time.monotonic() + 1200)
        team.base_url = "http://localhost"
        class Runtime:
            async def start(self): pass
            async def stop(self): pass
            async def inspect_resources(self): return {}
            async def read(self, path): raise OSError("reward not written")
        team.runtime = lambda *args: Runtime()
        async def setup(*args): pass
        async def launch(*args): return ProgramResult(0, "done", "")
        monkeypatch.setattr("harness_bloat_bench.multiagent.legacy_team.setup_harness", setup)
        monkeypatch.setattr("harness_bloat_bench.multiagent.legacy_team.launch_harness", launch)
        await team.worker(config.task_ids[0], "")
        row = team.rows[config.task_ids[0]]
        assert row["state"] == "error"
        assert row["reward"] is None
        assert row["passed"] is None
        assert row["verifier_state"] == "error"
        ledger.db.close()
    asyncio.run(scenario())


def test_native_stream_keeps_provider_deltas_and_accounts_final_usage(tmp_path):
    async def scenario():
        ledger = Ledger(tmp_path)
        gateway = Gateway("unused-test-key", ledger, {"test": {"pricing": {"prompt": "0", "completion": "0"}}}, base_url="https://provider.test")
        await gateway.http.aclose()
        base = {"id": "gen-body", "object": "chat.completion.chunk", "created": 1, "model": "test"}
        packets = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Think", "tool_calls": [{"index": 0, "id": "tool-1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":"}}]}, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"pwd\"}"}}]}, "finish_reason": "tool_calls"}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.003, "completion_tokens_details": {"reasoning_tokens": 12}}},
        ]
        wire = ": OPENROUTER PROCESSING\n\n" + "".join("data: " + json.dumps(packet) + "\n\n" for packet in packets) + "data: [DONE]\n\n"
        async def provider(request):
            body = json.loads(request.content)
            assert body["stream"] is True
            assert body["stream_options"] == {"include_usage": True}
            assert body["model"] == "test"
            return httpx.Response(200, headers={"Content-Type": "text/event-stream", "X-Generation-Id": "gen-header"}, content=wire)
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        route = AgentRoute("team", "worker", "executor", "task", ModelSpec(model="test", reasoning_effort="medium"), time.monotonic() + 30, 10)
        gateway.register(route)
        app = FastAPI()
        @app.post("/")
        async def relay(request: Request):
            return await gateway.relay("worker", "chat/completions", request)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local.test") as client:
            result = await client.post("/", headers={"Authorization": "Bearer " + route.token}, json={"stream": True, "model": "ignored", "messages": [{"role": "user", "content": "work"}]})
        assert result.text == wire
        record = next(iter(ledger.records.values()))
        assert record["generation_id"] == "gen-header"
        assert record["total_tokens"] == 120
        assert record["reasoning_tokens"] == 12
        raw = json.loads(next((tmp_path / "team/worker/calls").glob("*-response.json")).read_text())
        assert raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"command":"pwd"}'
        assert not gateway.inflight
        await gateway.close()
        ledger.db.close()
    asyncio.run(scenario())


def test_responses_stream_preserves_reasoning_for_next_coordinator_tool_call(tmp_path):
    async def scenario():
        ledger = Ledger(tmp_path)
        gateway = Gateway("unused-test-key", ledger, {"test": {"pricing": {"prompt": "0", "completion": "0"}}}, base_url="https://provider.test")
        await gateway.http.aclose()
        output = [{"type": "reasoning", "id": "reason-1", "summary": []}, {"type": "function_call", "id": "fn-1", "call_id": "call-1", "name": "delegate_tasks", "arguments": "{}"}]
        completed = {"id": "gen-responses", "model": "test", "status": "completed", "output": output, "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "cost": 0.004}}
        async def provider(request):
            assert request.url.path.endswith("/responses")
            assert json.loads(request.content)["reasoning"]["effort"] == "high"
            wire = "data: " + json.dumps({"type": "response.created", "response": {"id": "gen-responses", "status": "in_progress"}}) + "\n\ndata: " + json.dumps({"type": "response.completed", "response": completed}) + "\n\n"
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=wire)
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        route = AgentRoute("team", "main", "orchestrator", None, ModelSpec(model="test", reasoning_effort="high", api="responses"), time.monotonic() + 30, 10)
        gateway.register(route)
        app = FastAPI()
        @app.post("/")
        async def relay(request: Request):
            return await gateway.relay("main", "chat/completions", request)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local.test") as client:
            result = await client.post("/", headers={"Authorization": "Bearer " + route.token}, json={"messages": [{"role": "user", "content": "coordinate"}]})
        assert result.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call-1"
        assert route.prior_outputs["call-1"] == output
        assert ledger.summarize()["total_cost_usd"] == 0.004
        await gateway.close()
        ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("trigger", ["deadline", "shutdown"])
@pytest.mark.parametrize("usage_before_disconnect", [False, True])
def test_stream_cancellation_recovers_provider_bill_and_closes_connection(tmp_path, trigger, usage_before_disconnect):
    async def scenario():
        ledger = Ledger(tmp_path)
        gateway = Gateway("unused-test-key", ledger, {"test": {"pricing": {"prompt": "0", "completion": "0"}}}, base_url="https://provider.test")
        await gateway.http.aclose()
        reading = asyncio.Event()
        class Generation(httpx.AsyncByteStream):
            closed = False
            async def __aiter__(self):
                yield b'data: {"id":"gen-early","model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
                if usage_before_disconnect:
                    yield b'data: {"id":"gen-early","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"cost":0.004}}\n\n'
                reading.set()
                await asyncio.Event().wait()
            async def aclose(self):
                self.closed = True
        generation = Generation()
        async def provider(request):
            if request.method == "GET":
                assert not usage_before_disconnect, "Delivered provider billing needs no recovery request"
                assert request.url.params["id"] == "gen-early"
                assert generation.closed
                return httpx.Response(200, json={"data": {"total_cost": 0.004, "native_tokens_prompt": 100, "native_tokens_completion": 20, "native_tokens_reasoning": 12, "native_tokens_cached": 80, "model": "test", "cancelled": True}})
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=generation)
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        deadline = time.monotonic() + (0.1 if trigger == "deadline" else 30)
        route = AgentRoute("team", "worker", "executor", "task", ModelSpec(model="test", reasoning_effort="medium"), deadline, 10)
        gateway.register(route)
        app = FastAPI()
        @app.post("/")
        async def relay(request: Request):
            return await gateway.relay("worker", "chat/completions", request)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local.test") as client:
            task = asyncio.create_task(client.post("/", headers={"Authorization": "Bearer " + route.token}, json={"stream": True, "messages": [{"role": "user", "content": "work"}]}))
            await asyncio.wait_for(reading.wait(), 1)
            record = next(iter(ledger.records.values()))
            assert record["generation_id"] == "gen-early"
            if trigger == "deadline":
                await asyncio.wait_for(task, 1)
            await gateway.close()
            await asyncio.gather(task, return_exceptions=True)
        assert generation.closed
        assert record["status"] == ("error" if trigger == "deadline" else "cancelled")
        if not usage_before_disconnect:
            assert record["provider_cancelled"] is True
        assert record["served_model"] == "test"
        assert ledger.summarize()["accounting_complete"]
        assert ledger.summarize()["total_cost_usd"] == 0.004
        assert ledger.summarize()["total_tokens"] == 120
        ledger.db.close()
    asyncio.run(scenario())


def test_global_deadline_finalizes_one_consistent_team_record(tmp_path, monkeypatch):
    from harness_bloat_bench.multiagent import runner
    async def scenario():
        config = load_config(CONFIG)
        catalog = [{"id": spec.model, "pricing": {"prompt": "0", "completion": "0"}} for spec in (config.orchestrator, config.executor)]
        async def get_models(client, url, **kwargs):
            assert url.endswith("/models")
            return httpx.Response(200, request=httpx.Request("GET", url), json={"data": catalog})
        async def docker(*args):
            if args[:2] == ("image", "inspect"):
                return ProgramResult(0, json.dumps([{"Id": "sha256:test", "RepoDigests": [], "Architecture": "amd64", "Os": "linux"}]), "")
            if args[0] == "info":
                return ProgramResult(0, json.dumps({"Architecture": "aarch64", "NCPU": 2, "MemTotal": 2**30}), "")
            if args[0] == "top":
                return ProgramResult(0, "PID COMMAND\n1 sleep 30\n", "")
            return ProgramResult(0, "test-container", "")
        async def interrupted_team(team, base_url):
            team.coordinator["state"] = "running"
            try:
                await asyncio.Event().wait()
            finally:
                # Reproduce cancellation before Team.run can freeze finished_at.
                team.coordinator["state"] = "timed_out"
        tasks = [SimpleNamespace(data=SimpleNamespace(name="dataset/" + task, image=None)) for task in config.task_ids]
        monkeypatch.setattr(runner.httpx.AsyncClient, "get", get_models)
        monkeypatch.setattr(runner, "docker", docker)
        monkeypatch.setattr(runner, "configure_artifact_cache", lambda: None)
        monkeypatch.setattr(runner, "prepare_artifact", lambda config: tmp_path)
        monkeypatch.setattr(runner, "HarborTaskset", lambda config: SimpleNamespace(load=lambda: tasks))
        monkeypatch.setattr(Team, "run", interrupted_team)
        output = tmp_path / "run"
        await runner.run_configs([config], output, "unused-test-key", 0.1)
        comparison = json.loads((output / "comparison.json").read_text())
        team_record = json.loads((output / config.name / "result.json").read_text())
        assert comparison["teams"][0] == team_record
        assert team_record["orchestrator"]["state"] == "timed_out"
        assert team_record["runtime_seconds"] < 2
        manifest = json.loads((output / "manifest.json").read_text())
        assert all((output / "runner-source" / name).is_file() for name in manifest["implementation_sha256"])
    asyncio.run(scenario())
