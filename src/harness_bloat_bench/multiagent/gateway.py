"""Agent-specific OpenRouter relay, including Chat -> Responses compatibility."""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from verifiers.v1.dialects.chat import ChatStreamParser

from .accounting import Ledger, normalized_usage, scrub
from .config import ModelSpec


def responses_content(content):
    if not isinstance(content, list):
        return content
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append({**part, "type": "input_text"})
        elif part.get("type") == "image_url":
            image = part["image_url"]
            parts.append({"type": "input_image", "image_url": image["url"],
                          "detail": image.get("detail", "auto")})
        else:
            parts.append(part)
    return parts


def chat_to_responses(body: dict, model: ModelSpec, prior_outputs: dict[str, list] | None = None) -> dict:
    items = []
    for message in body.get("messages", []):
        role = message["role"]
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": responses_content(message.get("content") or "")})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            previous = (prior_outputs or {}).get(calls[0]["id"]) if calls else None
            if previous:
                items.extend(previous)
                continue
            if message.get("content"):
                items.append({"role": role, "content": responses_content(message["content"])})
            items.extend({"type": "function_call", "call_id": call["id"], **call["function"]} for call in calls)
        else:
            items.append({"role": role, "content": responses_content(message.get("content") or "")})
    result = {
        "model": model.model,
        "input": items,
        "reasoning": {"effort": model.reasoning_effort},
        "max_output_tokens": model.max_output_tokens,
        "stream": False,
        "store": False,
    }
    if body.get("tools"):
        result["tools"] = [{"type": "function", **tool["function"], "strict": False} for tool in body["tools"] if tool.get("type") == "function"]
        result["parallel_tool_calls"] = True
        choice = body.get("tool_choice", "auto")
        if isinstance(choice, dict) and "function" in choice:
            choice = {"type": "function", "name": choice["function"]["name"]}
        result["tool_choice"] = choice
    return result


def responses_to_chat(raw: dict) -> dict:
    text = "".join(part.get("text", "") for item in raw.get("output", []) if item.get("type") == "message" for part in item.get("content", []) if part.get("type") == "output_text")
    calls = [{"id": item["call_id"], "type": "function", "function": {"name": item["name"], "arguments": item["arguments"]}} for item in raw.get("output", []) if item.get("type") == "function_call"]
    message = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    usage = raw.get("usage")
    if usage:
        usage = {**usage, "prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens"), "prompt_tokens_details": usage.get("input_tokens_details"), "completion_tokens_details": usage.get("output_tokens_details")}
    return {
        "id": raw.get("id", ""), "object": "chat.completion", "created": raw.get("created_at", int(time.time())),
        "model": raw.get("model", ""), "usage": usage,
        "choices": [{"index": 0, "message": message, "finish_reason": "length" if raw.get("status") == "incomplete" else "tool_calls" if calls else "stop"}],
    }


async def chat_sse(raw: dict):
    """Replay the completed upstream turn in the CLI's streaming wire format."""
    base = {key: raw[key] for key in ("id", "created", "model")}
    base["object"] = "chat.completion.chunk"
    choice = raw["choices"][0]
    message = dict(choice["message"])
    if message.get("tool_calls"):
        message["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
    for payload in [
        {**base, "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}], "usage": raw.get("usage")},
    ]:
        yield "data: " + json.dumps(payload) + "\n\n"
    yield "data: [DONE]\n\n"


@dataclass
class AgentRoute:
    team: str
    agent_id: str
    role: str
    task_id: str | None
    model: ModelSpec
    deadline: float
    max_cost_usd: float | None
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    prior_outputs: dict[str, list] = field(default_factory=dict)
    native_executor: AgentRoute | None = None
    aggregate_native_workers: bool = False


class Gateway:
    def __init__(self, key: str, ledger: Ledger, catalog: dict[str, dict], base_url: str = "https://openrouter.ai/api/v1"):
        self.key, self.ledger, self.catalog, self.base_url = key, ledger, catalog, base_url
        self.routes: dict[str, AgentRoute] = {}
        self.inflight: dict[str, asyncio.Task] = {}
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=30))

    def register(self, route: AgentRoute):
        self.routes[route.agent_id] = route

    async def relay(self, agent_id: str, path: str, request: Request):
        route = self.routes.get(agent_id)
        if route is None or request.headers.get("authorization") != "Bearer " + route.token:
            raise HTTPException(401, "Invalid agent credential")
        remaining = route.deadline - time.monotonic()
        if remaining <= 0:
            raise HTTPException(408, "Agent deadline reached")
        body = await request.json()
        calls = [r for r in self.ledger.records.values() if r["agent_id"] == agent_id]
        # Native children share credentials; their iteration limits belong to
        # the harness, rather than imposing one helper's limit on the whole pool.
        if not route.aggregate_native_workers and len(calls) >= route.model.max_calls:
            raise HTTPException(429, "Agent call limit reached")
        pricing = self.catalog[route.model.model]["pricing"]
        tiers = [pricing, *pricing.get("overrides", [])]
        if route.model.model.startswith("~"):
            family = route.model.model.lstrip("~").removesuffix("-latest")
            for name, entry in self.catalog.items():
                if name.startswith(family):
                    tiers.extend([entry["pricing"], *entry["pricing"].get("overrides", [])])
        reservation = (len(json.dumps(body).encode()) + 4096) * max(float(t.get("prompt", pricing["prompt"])) for t in tiers) + route.model.max_output_tokens * max(float(t.get("completion", pricing["completion"])) for t in tiers)
        committed = sum(r.get("budget_charge_usd", r.get("reserved_cost_usd", 0)) for r in self.ledger.records.values() if r["team"] == route.team)
        if route.max_cost_usd is not None and committed + reservation > route.max_cost_usd:
            raise HTTPException(429, "Team cost admission ceiling reached")
        if path not in ("chat/completions", "responses"):
            raise HTTPException(404, "Unsupported model endpoint")
        stream = body.get("stream", False)
        translate = path == "chat/completions" and route.model.api == "responses"
        if translate:
            upstream = chat_to_responses(body, route.model, route.prior_outputs)
            upstream_path = "responses"
        else:
            upstream = {**body, "model": route.model.model, "stream": False}
            upstream.pop("stream_options", None)
            upstream.pop("reasoning_effort", None)
            upstream["reasoning"] = {"effort": route.model.reasoning_effort}
            upstream.pop("max_tokens", None)
            upstream.pop("max_completion_tokens", None)
            upstream["max_output_tokens" if path == "responses" else "max_tokens"] = route.model.max_output_tokens
            upstream_path = path
        # Always stream upstream: closing the connection cancels generation on
        # the supported OpenAI/DeepSeek providers, even for a non-streaming CLI.
        upstream["stream"] = True
        if upstream_path == "chat/completions":
            upstream["stream_options"] = {"include_usage": True}
        call_id = self.ledger.start_call(team=route.team, agent_id=agent_id, role=route.role, task_id=route.task_id, requested_model=route.model.model, reasoning_effort=route.model.reasoning_effort, endpoint=upstream_path, reserved_cost_usd=reservation)
        self.inflight[call_id] = asyncio.current_task()
        directory = self.ledger.output / route.team / agent_id / "calls"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{call_id}-request.json").write_text(scrub(json.dumps(upstream, indent=2)))
        started = time.monotonic()
        response = None
        handed_off = False
        try:
            async with asyncio.timeout_at(route.deadline):
                request = self.http.build_request("POST", self.base_url + "/" + upstream_path, headers={"Authorization": "Bearer " + self.key, "X-Title": "Harness Bench Multiagent"}, json=upstream)
                response = await self.http.send(request, stream=True)
            generation_id = response.headers.get("x-generation-id")
            if generation_id:
                self.ledger.records[call_id]["generation_id"] = generation_id
                self.ledger.save(call_id)
            if "text/event-stream" in response.headers.get("content-type", ""):
                collected = {}
                reader = self.read_stream(response, upstream_path, route, call_id, directory, started, reservation, collected)
                if stream and not translate:
                    handed_off = True
                    return StreamingResponse(reader, status_code=response.status_code, media_type="text/event-stream")
                async for _ in reader:
                    pass
                raw = collected["raw"]
            else:
                async with asyncio.timeout_at(route.deadline):
                    await response.aread()
                raw = response.json()
                self.finish_response(route, call_id, raw, started, reservation, "error" if response.status_code >= 400 or raw.get("error") else "completed", response.status_code)
            (directory / f"{call_id}-response.json").write_text(scrub(json.dumps(raw, indent=2)))
            status = self.ledger.records[call_id]["status"]
            if status == "error":
                return JSONResponse(raw, status_code=response.status_code if response.status_code >= 400 else 502)
            for item in raw.get("output", []):
                if item.get("type") == "function_call":
                    route.prior_outputs[item["call_id"]] = raw["output"]
            if translate:
                raw = responses_to_chat(raw)
            if stream and path == "chat/completions":
                return StreamingResponse(chat_sse(raw), media_type="text/event-stream")
            if stream:
                async def events():
                    yield "event: response.completed\ndata: " + json.dumps({"type": "response.completed", "response": raw}) + "\n\n"
                return StreamingResponse(events(), media_type="text/event-stream")
            return JSONResponse(raw)
        except asyncio.CancelledError:
            if self.ledger.records[call_id]["status"] == "in_flight":
                self.ledger.finish_call(call_id, status="cancelled", error="Upstream stream cancelled", budget_charge_usd=reservation, **normalized_usage(None, pricing))
            raise
        except Exception as error:
            message = scrub(str(error).replace(self.key, "[REDACTED]"))
            if self.ledger.records[call_id]["status"] == "in_flight":
                self.ledger.finish_call(call_id, status="error", error=message, duration_seconds=time.monotonic() - started, budget_charge_usd=reservation, **normalized_usage(None, pricing))
            return JSONResponse({"error": {"message": message or "Agent request deadline reached"}}, status_code=504 if isinstance(error, TimeoutError) else 502)
        finally:
            if not handed_off:
                if response is not None:
                    await response.aclose()
                self.inflight.pop(call_id, None)

    def finish_response(self, route, call_id, raw, started, reservation, status, http_status=200):
        record = self.ledger.records[call_id]
        pricing = self.catalog.get(raw.get("model"), self.catalog[route.model.model])["pricing"]
        accounting = normalized_usage(raw.get("usage"), pricing)
        charge = accounting.get("cost_usd")
        if charge is None:
            charge = accounting.get("estimated_cost_usd")
        self.ledger.finish_call(call_id, status=status, http_status=http_status, generation_id=record.get("generation_id") or raw.get("id"), served_model=raw.get("model") or record.get("served_model"), duration_seconds=time.monotonic() - started, budget_charge_usd=charge if charge is not None else reservation, **accounting)
        self.ledger.event("model_call_finished", team=route.team, agent_id=route.agent_id, role=route.role, task_id=route.task_id, call_id=call_id, status=status, **accounting)

    async def read_stream(self, response, path, route, call_id, directory, started, reservation, collected):
        self.inflight[call_id] = asyncio.current_task()
        parser = ChatStreamParser() if path == "chat/completions" else None
        raw = {}
        status = "completed"
        terminal_seen = False
        partial_usage = None
        event_lines = []
        try:
            async with asyncio.timeout_at(route.deadline):
                with (directory / f"{call_id}-response.sse").open("w") as transcript:
                    async for line in response.aiter_lines():
                        transcript.write(scrub(line) + "\n")
                        transcript.flush()
                        if line.startswith("data:"):
                            event_lines.append(line[5:].strip())
                        elif not line and event_lines:
                            payload = "\n".join(event_lines)
                            event_lines.clear()
                            if payload == "[DONE]":
                                terminal_seen = True
                            else:
                                packet = json.loads(payload)
                                if parser:
                                    parser.feed(("data: " + payload + "\n\n").encode())
                                    native = packet
                                else:
                                    native = packet.get("response") or packet
                                    if packet.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                                        raw = native
                                        terminal_seen = True
                                record = self.ledger.records[call_id]
                                changed = False
                                if native.get("id") and not record.get("generation_id"):
                                    record["generation_id"] = native["id"]
                                    changed = True
                                if native.get("model") and not record.get("served_model"):
                                    record["served_model"] = native.get("model")
                                    changed = True
                                if changed:
                                    self.ledger.save(call_id)
                                if native.get("usage"):
                                    partial_usage = native["usage"]
                                if packet.get("error") or native.get("status") == "failed":
                                    status = "error"
                                    raw = native
                        # The CLI receives the original provider events, including
                        # reasoning and tool deltas, without synthetic rewrites.
                        yield line + "\n"
            if not terminal_seen:
                status = "error"
                raw = {"error": {"message": "Provider stream ended without a terminal event"}}
        except asyncio.CancelledError:
            status = "cancelled"
            raw = {"error": {"message": "Upstream generation cancelled"}}
            raise
        except Exception as error:
            status = "error"
            raw = {"error": {"message": scrub(str(error)) or "Agent stream deadline reached"}}
            yield "data: " + json.dumps(raw) + "\n\n"
        finally:
            if parser and status == "completed":
                try:
                    parser.finish()  # materialize accumulated parts, retaining native usage
                    head = parser.head or {}
                    raw = {"id": head.get("id"), "created": head.get("created", int(time.time())), "model": head.get("model"), "object": "chat.completion", "choices": [{"index": 0, "message": parser.message, "finish_reason": parser.finish_reason or "stop"}], "usage": parser.usage}
                except Exception as error:
                    status = "error"
                    raw = {"error": {"message": scrub(str(error))}}
            # A client may disconnect after the usage frame but before [DONE].
            # Preserve the provider's bill even when execution was cancelled.
            if partial_usage and not raw.get("usage"):
                raw["usage"] = partial_usage
            collected["raw"] = raw
            (directory / f"{call_id}-response.json").write_text(scrub(json.dumps(raw, indent=2)))
            self.finish_response(route, call_id, raw, started, reservation, status)
            self.inflight.pop(call_id, None)
            await response.aclose()

    async def reconcile(self):
        """Recover authoritative billing for cancelled/incomplete generations."""
        pending = [r for r in self.ledger.records.values() if r.get("cost_usd") is None and r.get("generation_id")]
        async def recover(record):
            for attempt in range(3):
                try:
                    response = await self.http.get(self.base_url + "/generation", params={"id": record["generation_id"]}, headers={"Authorization": "Bearer " + self.key}, timeout=3)
                    if response.is_success:
                        data = response.json()["data"]
                        cost = data.get("total_cost")
                        inputs = data.get("native_tokens_prompt")
                        outputs = data.get("native_tokens_completion")
                        if inputs is None:
                            inputs = data.get("tokens_prompt")
                        if outputs is None:
                            outputs = data.get("tokens_completion")
                        if cost is not None:
                            usage = normalized_usage({"input_tokens": inputs, "output_tokens": outputs, "cost": cost, "input_tokens_details": {"cached_tokens": data.get("native_tokens_cached") or 0}, "output_tokens_details": {"reasoning_tokens": data.get("native_tokens_reasoning")}}, {})
                            self.ledger.finish_call(record["call_id"], **usage, usage_reconciled_at=time.time(), budget_charge_usd=cost, served_model=data.get("model"), provider_name=data.get("provider_name"), provider_cancelled=data.get("cancelled"))
                            self.ledger.event("billing_reconciled", call_id=record["call_id"], agent_id=record["agent_id"], cost_usd=cost)
                            return
                except (httpx.HTTPError, ValueError, KeyError):
                    pass
                await asyncio.sleep(0.3 * (attempt + 1))
        async with asyncio.timeout(10):
            await asyncio.gather(*(recover(record) for record in pending))

    async def close(self):
        pending = list(self.inflight.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        try:
            await self.reconcile()
        except TimeoutError:
            pass
        await self.http.aclose()
