"""Shared object-store helpers for distributed rollout workers.

The object store serves two different durability needs:

* Dagster's IO manager moves small step values between Celery workers.
* A stable result cache keyed by logical task execution prevents a completed
  model rollout from being repeated when a worker result must be redelivered.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import socket
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


OBJECT_STORE_ENDPOINT_ENV = "HARNESS_BLOAT_OBJECT_STORE_ENDPOINT"
OBJECT_STORE_BUCKET_ENV = "HARNESS_BLOAT_OBJECT_STORE_BUCKET"
WORKER_NAME_ENV = "HARNESS_BLOAT_WORKER_NAME"


def object_store_enabled() -> bool:
    return bool(
        os.environ.get(OBJECT_STORE_ENDPOINT_ENV)
        and os.environ.get(OBJECT_STORE_BUCKET_ENV)
    )


def worker_name() -> str:
    return os.environ.get(WORKER_NAME_ENV) or socket.gethostname()


def object_store_client():
    endpoint = os.environ.get(OBJECT_STORE_ENDPOINT_ENV)
    bucket = os.environ.get(OBJECT_STORE_BUCKET_ENV)
    if not endpoint or not bucket:
        raise RuntimeError(
            f"{OBJECT_STORE_ENDPOINT_ENV} and {OBJECT_STORE_BUCKET_ENV} are required "
            "for distributed execution"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        use_ssl=endpoint.startswith("https://"),
    )


def _bucket() -> str:
    bucket = os.environ.get(OBJECT_STORE_BUCKET_ENV)
    if not bucket:
        raise RuntimeError(f"{OBJECT_STORE_BUCKET_ENV} is required")
    return bucket


def _logical_execution_key(spec: dict[str, Any]) -> str:
    return str(
        spec.get("task_execution_id")
        or f"{spec.get('campaign_id') or 'adhoc'}--{spec['key']}"
    )


def _cached_result_key(spec: dict[str, Any]) -> str:
    return f"result-cache/{_logical_execution_key(spec)}.json"


def load_cached_result(spec: dict[str, Any], dagster_run_id: str) -> dict | None:
    """Return a completed logical result, propagating object-store outages.

    A transport failure is intentionally not treated as a cache miss. Running a
    model call while the idempotency store is unavailable would turn a harmless
    storage outage into duplicate benchmark spend.
    """

    if not object_store_enabled():
        return None
    client = object_store_client()
    try:
        payload = client.get_object(Bucket=_bucket(), Key=_cached_result_key(spec))
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    row = json.loads(payload["Body"].read())
    if not isinstance(row, dict):
        raise RuntimeError("cached rollout result is not a JSON object")
    if row.get("task_execution_id") != spec.get("task_execution_id"):
        raise RuntimeError("cached rollout result has a conflicting logical identity")

    reused = dict(row)
    reused["reused_from_run_id"] = row.get("dagster_run_id")
    reused["reused_result"] = True
    reused["cache_hit_worker"] = worker_name()
    reused["dagster_run_id"] = dagster_run_id
    reused["rollout_key"] = spec["key"]
    reused["state_db_path"] = spec.get("state_db_path")
    reused["timestamp"] = dt.datetime.now(dt.UTC).isoformat()
    return reused


def store_completed_result(spec: dict[str, Any], row: dict[str, Any]) -> None:
    """Persist the first completed result for a logical task execution."""

    if not object_store_enabled():
        return
    body = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    try:
        object_store_client().put_object(
            Bucket=_bucket(),
            Key=_cached_result_key(spec),
            Body=body,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"PreconditionFailed", "412"} or status == 412:
            return
        raise


def upload_rollout_artifacts(spec: dict[str, Any], row: dict[str, Any]) -> str | None:
    """Upload a rollout directory and return its stable S3 URI."""

    if not object_store_enabled() or spec.get("dry_run"):
        return None
    output_dir = Path(row["output_dir"])
    if not output_dir.is_dir():
        return None

    prefix = f"artifacts/{_logical_execution_key(spec)}"
    client = object_store_client()
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(output_dir).as_posix()}"
            client.upload_file(str(path), _bucket(), key)
    return f"s3://{_bucket()}/{prefix}/"
