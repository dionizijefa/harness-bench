"""Native child identity must enforce the selected model and reasoning effort."""
import asyncio
import json
import time

import httpx
import pytest
from fastapi import FastAPI, Request

from harness_bloat_bench.multiagent.accounting import Ledger
from harness_bloat_bench.multiagent.config import ModelSpec
from harness_bloat_bench.multiagent.gateway import AgentRoute, Gateway


PAIRS = [
    ("openai/gpt-6.1-sol", "z-ai/glm-5.3-flash"),
    ("anthropic/claude-opus-5.5", "anthropic/claude-haiku-5.5"),
    ("openai/gpt-6-astra", "openai/gpt-6-luna"),
]


@pytest.mark.parametrize("main_model,helper_model", PAIRS)
@pytest.mark.parametrize("identity,path", [("endpoint", "chat/completions"), ("model_alias", "messages"), ("header", "responses")])
def test_native_requests_cannot_override_the_requested_pair(tmp_path, main_model, helper_model, identity, path):
    async def scenario():
        ledger = Ledger(tmp_path)
        catalog = {model: {"pricing": {"prompt": "0", "completion": "0"}} for model in (main_model, helper_model)}
        gateway = Gateway("provider-secret", ledger, catalog, base_url="https://provider.test")
        await gateway.http.aclose()
        received = []

        def provider(request):
            body = json.loads(request.content)
            received.append(body)
            assert request.headers["authorization"] == "Bearer provider-secret"
            if path == "messages":
                assert request.headers["anthropic-version"] == "2023-06-01"
                raw = {"id": "message", "type": "message", "role": "assistant", "model": body["model"],
                       "content": [{"type": "text", "text": "ready"}], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 1, "output_tokens": 1, "cost": .001}}
            elif path == "responses":
                raw = {"id": "response", "model": body["model"], "status": "completed", "output": [],
                       "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "cost": .001}}
            else:
                raw = {"id": "chat", "created": 1, "model": body["model"], "choices": [{"index": 0,
                       "message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": .001}}
            return httpx.Response(200, json=raw)

        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(provider))
        helper = AgentRoute("team", "helpers", "executor", "task", ModelSpec(model=helper_model, reasoning_effort="high"),
                            time.monotonic() + 30, None, aggregate_native_workers=True)
        main = AgentRoute("team", "main", "orchestrator", "task", ModelSpec(model=main_model, reasoning_effort="max"),
                          time.monotonic() + 30, None, native_executor=helper,
                          native_executor_model_alias="bench-native-executor",
                          native_executor_header=("x-openai-subagent", "collab_spawn"))
        gateway.register(main)
        gateway.register(helper)
        app = FastAPI()

        @app.post("/{agent_id}/{endpoint:path}")
        async def relay(agent_id: str, endpoint: str, request: Request):
            return await gateway.relay(agent_id, endpoint, request)

        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local.test") as client:
                for child in (False, True):
                    target = helper if child and identity == "endpoint" else main
                    headers = {"Authorization": "Bearer " + target.token}
                    body = {"model": "unrequested-model", "stream": False, "reasoning": {"effort": "low"},
                            "thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
                    if path == "responses":
                        body["input"] = [{"role": "user", "content": "original task"}]
                    else:
                        body["messages"] = [{"role": "user", "content": "original task"}]
                    if identity == "model_alias":
                        headers = {"x-api-key": main.token}
                        if child:
                            body["model"] = main.native_executor_model_alias
                    if identity == "header" and child:
                        headers["x-openai-subagent"] = "collab_spawn"
                    response = await client.post(f"/{target.agent_id}/{path}", json=body, headers=headers)
                    assert response.status_code == 200, response.text
                    expected = helper if child else main
                    assert received[-1]["model"] == expected.model.model
                    effort = received[-1]["output_config"]["effort"] if path == "messages" else received[-1]["reasoning"]["effort"]
                    assert effort == expected.model.reasoning_effort
                    if path == "messages":
                        assert received[-1]["thinking"] == {"type": "adaptive"}
                    assert received[-1].get("messages", received[-1].get("input")) == body.get("messages", body.get("input"))
            records = list(ledger.records.values())
            assert [(r["role"], r["requested_model"], r["reasoning_effort"]) for r in records] == [
                ("orchestrator", main_model, "max"), ("executor", helper_model, "high")]
        finally:
            await gateway.close()
            ledger.db.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_messages_stream_preserves_native_events_and_complete_cached_usage(tmp_path, stream):
    async def scenario():
        ledger = Ledger(tmp_path)
        model = "anthropic/claude-haiku-5.5"
        gateway = Gateway("provider-secret", ledger, {model: {"pricing": {"prompt": "0", "completion": "0"}}})
        await gateway.http.aclose()
        message = {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
                   "stop_reason": None, "stop_sequence": None,
                   "usage": {"input_tokens": 100, "output_tokens": 0, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 10}}
        packets = [{"type": "message_start", "message": message},
                   {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                   {"type": "content_block_stop", "index": 0},
                   {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 10, "cost": .015}},
                   {"type": "message_stop"}]
        data = "".join("event: " + p["type"] + "\ndata: " + json.dumps(p) + "\n\n" for p in packets)
        gateway.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text=data, headers={"Content-Type": "text/event-stream"})))
        route = AgentRoute("team", "helper", "executor", "task", ModelSpec(model=model, reasoning_effort="high"), time.monotonic() + 30, None)
        gateway.register(route)
        app = FastAPI()

        @app.post("/messages")
        async def relay(request: Request):
            return await gateway.relay(route.agent_id, "messages", request)

        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local.test") as client:
                response = await client.post("/messages", json={"stream": stream, "messages": [{"role": "user", "content": "task"}]},
                                             headers={"x-api-key": route.token})
                assert response.status_code == 200
                if stream:
                    assert response.text == data
                else:
                    assert response.json()["content"] == [{"type": "text", "text": "done"}]
                usage = ledger.summarize()
                assert usage["input_tokens"] == 150 and usage["output_tokens"] == 10
                assert usage["cached_input_tokens"] == 40 and usage["total_cost_usd"] == .015
        finally:
            await gateway.close()
            ledger.db.close()

    asyncio.run(scenario())
