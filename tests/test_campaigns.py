import copy
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import dagster as dg
import yaml

import harness_bloat_bench.definitions as definitions
from harness_bloat_bench.definitions import (
    CAMPAIGN_ID_TAG,
    CELERY_QUEUE_TAG,
    COMBINATION_ID_TAG,
    HARNESS_TAG,
    REPETITION_TAG,
    TASK_COUNT_TAG,
    TASK_SELECTION_TAG,
    CampaignConfig,
    _campaign_task_ids,
    _persist_rollout_row,
    _task_mapping_key,
    plan_terminal_bench_campaign,
    terminal_bench_campaign_sensor,
    terminal_bench_combination,
    terminal_bench_worker_pool_combination,
    terminal_bench_worker_pool_sensor,
)
from harness_bloat_bench.state import (
    bind_combination_run,
    combination_id,
    mark_task_running,
    reconcile_combination,
    task_execution_id,
)


def _planner_config(tmp_path: Path, **overrides) -> dict:
    config = {
        "campaign_id": "campaign-test",
        "batch_id": "experiment-test",
        "models": ["model-a"],
        "harnesses": [{"id": "codex_agent", "version": "0.137.0"}],
        "repetitions": 2,
        "task_selection": {"mode": "selected", "ids": ["task-a", "task-b"]},
        "output_dir": str(tmp_path / "artifacts"),
        "dry_run": True,
        "remote": None,
    }
    config.update(overrides)
    return {"ops": {"plan_campaign": {"config": config}}}


def test_checked_in_campaign_configs_validate() -> None:
    project_root = Path(__file__).parents[1]
    for filename in (
        "campaign-dry-run.yaml",
        "campaign-remote.example.yaml",
        "campaign-worker-pool-canary.yaml",
    ):
        config = yaml.safe_load((project_root / "configs" / filename).read_text())
        dg.validate_run_config(plan_terminal_bench_campaign, config)


def test_logical_ids_are_deterministic_and_scoped() -> None:
    first = combination_id(
        "campaign-a", "model-a", "codex_agent", "0.137.0", 1
    )
    same = combination_id(
        "campaign-a", "model-a", "codex_agent", "0.137.0", 1
    )
    another_repetition = combination_id(
        "campaign-a", "model-a", "codex_agent", "0.137.0", 2
    )

    assert first == same
    assert first.startswith("combination_")
    assert first != another_repetition
    assert task_execution_id(first, "task-a") == task_execution_id(first, "task-a")
    assert task_execution_id(first, "task-a") != task_execution_id(first, "task-b")
    assert task_execution_id(first, "task-a") != task_execution_id(
        another_repetition, "task-a"
    )
    assert _task_mapping_key(
        "adaptive-rejection-sampler", task_execution_id(first, "task-a")
    ).startswith("adaptive_rejection_sampler__")


def test_campaign_planner_persists_the_complete_plan_idempotently(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "state.sqlite"
    monkeypatch.setenv("HARNESS_BLOAT_STATE_DB", str(database_path))
    instance = dg.DagsterInstance.ephemeral()
    run_config = _planner_config(tmp_path)

    first = plan_terminal_bench_campaign.execute_in_process(
        instance=instance, run_config=run_config
    )
    second = plan_terminal_bench_campaign.execute_in_process(
        instance=instance, run_config=run_config
    )

    assert first.success
    assert second.success
    assert first.output_for_node("plan_campaign") == "campaign-test"
    assert second.output_for_node("plan_campaign") == "campaign-test"
    first_run = instance.get_run_by_id(first.run_id)
    assert first_run is not None
    assert first_run.tags[CAMPAIGN_ID_TAG] == "campaign-test"
    assert first_run.tags[TASK_SELECTION_TAG] == "selected"
    assert first_run.tags[TASK_COUNT_TAG] == "2"

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT state, task_count, combination_count, task_execution_count "
            "FROM benchmark_campaigns"
        ).fetchone() == ("planned", 2, 2, 4)
        assert connection.execute(
            "SELECT COUNT(*) FROM benchmark_combinations"
        ).fetchone() == (2,)
        tasks = connection.execute(
            "SELECT combination_id, task_id, task_execution_id, state "
            "FROM benchmark_task_executions ORDER BY combination_id, task_id"
        ).fetchall()
    assert len(tasks) == 4
    assert all(row[2] == task_execution_id(row[0], row[1]) for row in tasks)
    assert all(row[3] == "planned" for row in tasks)

    conflicting = plan_terminal_bench_campaign.execute_in_process(
        instance=instance,
        run_config=_planner_config(tmp_path, repetitions=3),
        raise_on_error=False,
    )
    assert not conflicting.success
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT combination_count, task_execution_count "
            "FROM benchmark_campaigns"
        ).fetchone() == (2, 4)


def test_campaign_planner_rejects_retries_until_they_are_tracked(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_STATE_DB", str(tmp_path / "state.sqlite"))

    result = plan_terminal_bench_campaign.execute_in_process(
        run_config=_planner_config(
            tmp_path,
            campaign_id="campaign-with-retries",
            rollout_retries=1,
        ),
        raise_on_error=False,
    )

    assert not result.success
    assert not (tmp_path / "state.sqlite").exists()


def test_sensor_launches_one_run_per_combination_and_tasks_finish_independently(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "state.sqlite"
    monkeypatch.setenv("HARNESS_BLOAT_STATE_DB", str(database_path))
    instance = dg.DagsterInstance.ephemeral()
    planner = plan_terminal_bench_campaign.execute_in_process(
        instance=instance,
        run_config=_planner_config(
            tmp_path,
            harnesses=[
                {"id": "codex_agent", "version": "0.137.0"},
                {"id": "omp_agent", "version": "17.2.10"},
            ],
        ),
    )
    assert planner.success

    tick = terminal_bench_campaign_sensor.evaluate_tick(
        dg.build_sensor_context(instance=instance)
    )
    requests = list(tick.run_requests)
    assert len(requests) == 4
    assert len({request.run_key for request in requests}) == 4
    assert {request.tags[HARNESS_TAG] for request in requests} == {
        "codex_agent",
        "omp_agent",
    }
    assert {request.tags[REPETITION_TAG] for request in requests} == {"1", "2"}
    assert all(request.tags[TASK_COUNT_TAG] == "2" for request in requests)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT state, COUNT(*) FROM benchmark_combinations GROUP BY state"
        ).fetchall() == [("queued", 4)]

    tampered_config = copy.deepcopy(requests[0].run_config)
    tampered_config["ops"]["plan_task_executions"]["config"]["task_ids"] = [
        "different-task"
    ]
    tampered = terminal_bench_combination.execute_in_process(
        instance=instance,
        run_config=tampered_config,
        raise_on_error=False,
    )
    assert not tampered.success
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT state, COUNT(*) FROM benchmark_task_executions GROUP BY state"
        ).fetchall() == [("planned", 8)]

    child_runs = []
    for request in requests:
        child = terminal_bench_combination.execute_in_process(
            instance=instance,
            run_config=request.run_config,
            tags=request.tags,
        )
        assert child.success
        child_runs.append(child)
        run = instance.get_run_by_id(child.run_id)
        assert run is not None
        assert run.tags[CAMPAIGN_ID_TAG] == "campaign-test"
        assert run.tags[COMBINATION_ID_TAG] == request.run_key

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT state FROM benchmark_campaigns WHERE campaign_id = ?",
            ("campaign-test",),
        ).fetchone() == ("completed",)
        assert connection.execute(
            "SELECT state, COUNT(*) FROM benchmark_combinations GROUP BY state"
        ).fetchall() == [("completed", 4)]
        assert connection.execute(
            "SELECT state, outcome, COUNT(*) FROM benchmark_task_executions "
            "GROUP BY state, outcome"
        ).fetchall() == [("completed", "dry_run", 8)]
        assert connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT task_execution_id) "
            "FROM rollout_results"
        ).fetchone() == (8, 8)

    duplicate = terminal_bench_combination.execute_in_process(
        instance=instance,
        run_config=requests[0].run_config,
        tags=requests[0].tags,
        raise_on_error=False,
    )
    assert not duplicate.success
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM rollout_results"
        ).fetchone() == (8,)


def test_worker_pool_sensor_routes_compute_and_centralizes_sqlite_writes(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "state.sqlite"
    monkeypatch.setenv("HARNESS_BLOAT_STATE_DB", str(database_path))
    instance = dg.DagsterInstance.ephemeral()
    planner = plan_terminal_bench_campaign.execute_in_process(
        instance=instance,
        run_config=_planner_config(tmp_path, repetitions=1),
    )
    assert planner.success

    tick = terminal_bench_worker_pool_sensor.evaluate_tick(
        dg.build_sensor_context(instance=instance)
    )
    request = tick.run_requests[0]
    assert "execution" not in request.run_config
    assert CELERY_QUEUE_TAG not in request.tags

    child = terminal_bench_worker_pool_combination.execute_in_process(
        instance=instance,
        run_config=request.run_config,
        tags=request.tags,
    )
    assert child.success
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT state, outcome, COUNT(*) FROM benchmark_task_executions "
            "GROUP BY state, outcome"
        ).fetchall() == [("completed", "dry_run", 2)]
        rows = connection.execute(
            "SELECT worker_name, reused_result FROM rollout_results"
        ).fetchall()
    assert len(rows) == 2
    assert all(worker and reused == 0 for worker, reused in rows)


def test_dagster_failure_reconciliation_terminates_unreported_tasks(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "state.sqlite"
    monkeypatch.setenv("HARNESS_BLOAT_STATE_DB", str(database_path))
    instance = dg.DagsterInstance.ephemeral()
    planner = plan_terminal_bench_campaign.execute_in_process(
        instance=instance,
        run_config=_planner_config(tmp_path, repetitions=1),
    )
    assert planner.success
    tick = terminal_bench_campaign_sensor.evaluate_tick(
        dg.build_sensor_context(instance=instance)
    )
    request = tick.run_requests[0]

    reconcile_combination(
        database_path,
        request.run_key,
        dagster_run_id="failed-run",
        dagster_status="FAILURE",
    )

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT state, error_type FROM benchmark_combinations"
        ).fetchone() == ("error", "DagsterRunFailed")
        assert connection.execute(
            "SELECT state, error_type, COUNT(*) FROM benchmark_task_executions "
            "GROUP BY state, error_type"
        ).fetchall() == [("error", "DagsterRunFailed", 2)]
        assert connection.execute(
            "SELECT state FROM benchmark_campaigns"
        ).fetchone() == ("error",)


def test_reconciliation_recovers_a_result_persisted_before_state_update(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "state.sqlite"
    monkeypatch.setenv("HARNESS_BLOAT_STATE_DB", str(database_path))
    instance = dg.DagsterInstance.ephemeral()
    planner = plan_terminal_bench_campaign.execute_in_process(
        instance=instance,
        run_config=_planner_config(tmp_path, repetitions=1),
    )
    assert planner.success
    request = terminal_bench_campaign_sensor.evaluate_tick(
        dg.build_sensor_context(instance=instance)
    ).run_requests[0]
    config = request.run_config["ops"]["plan_task_executions"]["config"]
    execution_key = task_execution_id(request.run_key, "task-a")
    bind_combination_run(
        database_path, request.run_key, "crashed-run", started=True
    )
    mark_task_running(
        database_path,
        execution_key,
        dagster_run_id="crashed-run",
        dagster_step_key="run_rollout[task_a]",
    )
    _persist_rollout_row(
        {
            "dagster_run_id": "crashed-run",
            "campaign_id": "campaign-test",
            "combination_id": request.run_key,
            "task_execution_id": execution_key,
            "batch_id": "experiment-test",
            "rollout_key": execution_key,
            "timestamp": "2026-08-24T00:00:00+00:00",
            "model": config["model"],
            "harness": config["harness"],
            "harness_version": config["harness_version"],
            "dataset": config["dataset"],
            "task_id": "task-a",
            "rollout": 1,
            "status": "dry_run",
            "passed": None,
            "reward": None,
            "output_dir": str(tmp_path / "artifacts" / "crashed-run" / execution_key),
            "state_db_path": str(database_path),
        }
    )

    reconcile_combination(
        database_path,
        request.run_key,
        dagster_run_id="crashed-run",
        dagster_status="FAILURE",
    )

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT state, outcome FROM benchmark_task_executions "
            "WHERE task_execution_id = ?",
            (execution_key,),
        ).fetchone() == ("completed", "dry_run")
        # The second task was never persisted and remains an execution error.
        assert connection.execute(
            "SELECT state, COUNT(*) FROM benchmark_task_executions GROUP BY state"
        ).fetchall() == [("completed", 1), ("error", 1)]


def test_full_task_selection_snapshots_discovery_and_excludes_images(
    monkeypatch,
) -> None:
    class FakeHarborTaskset:
        def __init__(self, _config) -> None:
            pass

        def load(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(data=SimpleNamespace(task_dir="/tasks/task-a")),
                SimpleNamespace(
                    data=SimpleNamespace(task_dir="/tasks/code-from-image")
                ),
                SimpleNamespace(data=SimpleNamespace(task_dir="/tasks/task-b")),
            ]

    monkeypatch.setattr(definitions, "HarborTaskset", FakeHarborTaskset)
    config = CampaignConfig(
        task_selection={"mode": "full"}, dry_run=True, remote=None
    )

    assert _campaign_task_ids(config) == ["task-a", "task-b"]
