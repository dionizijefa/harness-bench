"""Durable per-call usage; missing provider accounting is never called free."""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from pathlib import Path
from uuid import uuid4


def scrub(value: str) -> str:
    return re.sub(r"sk-or-[A-Za-z0-9_-]+", "[REDACTED]", value)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalized_usage(raw: dict | None, pricing: dict) -> dict:
    if not raw:
        return {"usage_source": "missing", "cost_usd": None, "estimated_cost_usd": None}
    inputs = raw.get("input_tokens", raw.get("prompt_tokens"))
    outputs = raw.get("output_tokens", raw.get("completion_tokens"))
    input_details = raw.get("input_tokens_details", raw.get("prompt_tokens_details")) or {}
    output_details = raw.get("output_tokens_details", raw.get("completion_tokens_details")) or {}
    cached = input_details.get("cached_tokens", raw.get("cache_read_input_tokens", 0)) or 0
    if inputs is not None and ("cache_read_input_tokens" in raw or "cache_creation_input_tokens" in raw):
        inputs += cached + (raw.get("cache_creation_input_tokens", 0) or 0)
    cost = raw.get("cost")
    estimated = None
    if cost is None and inputs is not None and outputs is not None:
        rates = pricing
        for tier in pricing.get("overrides", []):
            if inputs >= tier.get("min_prompt_tokens", float("inf")):
                rates = {**rates, **tier}
        estimated = (
            max(0, inputs - cached) * float(rates["prompt"])
            + cached * float(rates.get("input_cache_read", rates["prompt"]))
            + outputs * float(rates["completion"])
        )
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "total_tokens": raw.get("total_tokens", inputs + outputs if inputs is not None and outputs is not None else None),
        "cached_input_tokens": cached,
        "reasoning_tokens": output_details.get("reasoning_tokens", output_details.get("thinking_tokens")),
        "cost_usd": cost,
        "estimated_cost_usd": estimated,
        "usage_source": "provider" if cost is not None else "price_estimate" if estimated is not None else "provider_tokens_only",
    }


class Ledger:
    def __init__(self, output: Path):
        self.output = output
        self.db = sqlite3.connect(output / "multiagent.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE calls (call_id TEXT PRIMARY KEY, team TEXT, agent_id TEXT, role TEXT, task_id TEXT, record TEXT NOT NULL)")
        self.db.commit()
        self.records: dict[str, dict] = {}
        self.events: list[dict] = []

    def event(self, event: str, **fields):
        record = {"timestamp": utc_now(), "event": event, **fields}
        self.events.append(record)
        with (self.output / "events.jsonl").open("a") as handle:
            handle.write(scrub(json.dumps(record, default=str)) + "\n")

    def start_call(self, **fields) -> str:
        call_id = uuid4().hex
        self.records[call_id] = {"call_id": call_id, "started_at": utc_now(), "status": "in_flight", **fields}
        self.save(call_id)
        return call_id

    def finish_call(self, call_id: str, **fields):
        self.records[call_id].update(ended_at=utc_now(), **fields)
        self.save(call_id)

    def save(self, call_id: str):
        record = self.records[call_id]
        payload = scrub(json.dumps(record, default=str))
        self.db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?)", (
            call_id, record["team"], record["agent_id"], record["role"], record.get("task_id"), payload,
        ))
        self.db.commit()

    def summarize(self, *, team: str | None = None, agent_id: str | None = None) -> dict:
        calls = [r for r in self.records.values() if (team is None or r["team"] == team) and (agent_id is None or r["agent_id"] == agent_id)]
        usage_missing = sum(r.get("total_tokens") is None for r in calls)
        cost_missing = sum(r.get("cost_usd") is None for r in calls)
        known_cost = sum(r.get("cost_usd") or 0 for r in calls)
        estimated = sum(r.get("estimated_cost_usd") or 0 for r in calls)
        return {
            "model_call_count": len(calls),
            "failed_call_count": sum(r["status"] == "error" for r in calls),
            "cancelled_call_count": sum(r["status"] == "cancelled" for r in calls),
            "inflight_call_count": sum(r["status"] == "in_flight" for r in calls),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in calls),
            "cached_input_tokens": sum(r.get("cached_input_tokens") or 0 for r in calls),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in calls),
            "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in calls),
            "total_tokens": sum(r.get("total_tokens") or 0 for r in calls),
            "provider_cost_usd": known_cost,
            "estimated_cost_usd": estimated,
            "total_cost_usd": known_cost if not cost_missing else None,
            "calls_missing_usage": usage_missing,
            "calls_missing_cost": cost_missing,
            "accounting_complete": not usage_missing and not cost_missing,
            "served_models": sorted({r["served_model"] for r in calls if r.get("served_model")}),
        }
