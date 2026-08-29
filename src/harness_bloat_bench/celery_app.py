"""Celery application shared by control-step and rollout workers."""

from __future__ import annotations

import os

from dagster_celery.make_app import make_app
from dagster_celery.tasks import create_task


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required to start a worker-pool process")
    return value


app = make_app(
    app_args={
        "broker": _required("DAGSTER_CELERY_BROKER_URL"),
        "backend": _required("DAGSTER_CELERY_BACKEND_URL"),
        "include": ["harness_bloat_bench.definitions"],
        "config_source": {
            "enable_utc": True,
            "timezone": "UTC",
            "broker_heartbeat": 30,
            "broker_heartbeat_checkrate": 2,
            "worker_prefetch_multiplier": 1,
            "worker_cancel_long_running_tasks_on_connection_loss": True,
            "task_acks_late": True,
            "task_reject_on_worker_lost": True,
            "task_track_started": True,
            "broker_connection_retry_on_startup": True,
            "broker_connection_max_retries": 100_000,
        },
    }
)

execute_plan = create_task(app)
