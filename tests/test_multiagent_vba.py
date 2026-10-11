"""Partial credit, independent checks and the single-task benchmark protocol."""
import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from verifiers.v1.runtimes import ProgramResult

from harness_bloat_bench.multiagent import vba, vba_verifier
from harness_bloat_bench.multiagent.accounting import Ledger
from harness_bloat_bench.multiagent.config import TeamConfig, load_config
from harness_bloat_bench.multiagent.runner import write_report
from harness_bloat_bench.multiagent.vba_run import VbaRun
from harness_bloat_bench.multiagent.vba_rubric import DATASET, TASK_ID, rubric, score_results, validate_traces

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multiagent/hermes-sol-deepseek-vba-userform-port.json"
TRACE_IDS = [trace for spec in rubric()["subtasks"] for trace in spec["trace_ids"]]


def evidence(trace, passed=True):
    return {"trace_id": trace, "passed": passed, "status": "passed" if passed else "failed"}


def test_primary_and_legacy_configs_are_explicitly_separated():
    configs = [load_config(path) for path in (ROOT / "configs/multiagent").glob("*.json")]
    assert len(configs) == 3
    for config in configs:
        assert config.dataset == DATASET and config.task_ids == [TASK_ID]
        assert config.scoring == "vba_subtasks_v1"
        assert config.batch_timeout_seconds == 4500 and config.grading_reserve_seconds == 3600
        assert config.worker_timeout_seconds == config.batch_timeout_seconds - config.grading_reserve_seconds == 900
    legacy = [load_config(path) for path in (ROOT / "configs/multiagent/legacy").glob("*.json")]
    assert len(legacy) == 5
    assert all(config.scoring == "task" and config.dataset.endswith("2-1") for config in legacy)
    historical = legacy[0].model_dump(exclude={"scoring"})
    assert TeamConfig.model_validate(historical).scoring == "task"


@pytest.mark.parametrize("changes", [{"task_ids": [TASK_ID, "pypi-server"]}, {"dataset": "terminal-bench/terminal-bench@latest"},
                                    {"execution_mode": "isolated_tasks"}, {"scoring": "task"}])
def test_invalid_task_protocol_combinations_are_rejected(changes):
    config = load_config(CONFIG).model_dump()
    with pytest.raises(ValidationError):
        TeamConfig.model_validate({**config, **changes})


def test_partial_credit_does_not_require_sibling_success_or_shrink_denominator():
    scored = score_results([evidence("001_contract_meta"), evidence("008_atomic_child_validation_rollback")])
    assert scored["reward"] == .15
    assert not scored["score_complete"] and scored["scored_checks"] == 2
    assert scored["check_count"] == 28 and scored["passed_subtasks"] == 1
    atomic = next(row for row in scored["subtasks"] if row["subtask_id"] == "atomic-saves")
    assert atomic["reward"] == .5 and atomic["earned_reward"] == .05
    assert atomic["verifier_state"] == "unscored"
    assert sum(row["check_count"] for row in scored["subtasks"]) == 28
    failed_rest = [evidence(trace, trace in {"001_contract_meta", "008_atomic_child_validation_rollback"}) for trace in TRACE_IDS]
    assert score_results(failed_rest)["reward"] == .15
    assert score_results(failed_rest)["score_complete"]


def test_full_success_failure_and_empty_evidence_are_distinct():
    full = score_results([evidence(trace) for trace in TRACE_IDS])
    assert full["reward"] == 1 and full["passed_subtasks"] == 10 and full["score_complete"]
    failed = score_results([evidence(trace, False) for trace in TRACE_IDS])
    assert failed["reward"] == 0 and failed["score_complete"]
    missing = score_results([])
    assert missing["reward"] == 0 and not missing["score_complete"]
    assert missing["scored_checks"] == 0 and missing["subtask_count"] == 10


@pytest.mark.parametrize("results", [[evidence("unknown")], [evidence(TRACE_IDS[0])] * 2,
                                     [{"trace_id": TRACE_IDS[0], "passed": "true", "status": "passed"}],
                                     [{"trace_id": TRACE_IDS[0], "passed": 1, "status": "passed"}],
                                     [{"trace_id": TRACE_IDS[0], "passed": True, "status": "unscored"}]])
def test_malformed_results_cannot_create_credit(results):
    with pytest.raises(ValueError):
        score_results(results)


def fixture_task(path):
    for directory in ("environment", "tests"):
        (path / directory / "legacy_app").mkdir(parents=True)
        (path / directory / "legacy_app/README.md").write_text("Legacy inputs")
        (path / directory / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY legacy_app /shared/legacy_app\n")
    (path / "tests/traces").mkdir()
    for trace in TRACE_IDS:
        (path / "tests/traces" / (trace + ".json")).write_text(json.dumps({"trace_id": trace, "steps": [{"action": "api_reset"}]}))
    (path / "instruction.md").write_text("Port the legacy app")
    (path / "task.toml").write_text('[task]\nname="terminal-bench/vba-userform-port"\n')
    return SimpleNamespace(data=SimpleNamespace(task_dir=str(path), prompt="Port the legacy app"))


def test_upstream_trace_drift_fails_preflight(tmp_path):
    fixture_task(tmp_path)
    validate_traces(tmp_path)
    (tmp_path / "tests/traces/001_contract_meta.json").unlink()
    with pytest.raises(ValueError, match="do not match"):
        validate_traces(tmp_path)


def test_task_loader_resolves_dataset_digest_and_downloads_only_selected_task(tmp_path, monkeypatch):
    from harbor.models.task.id import PackageTaskId
    from harbor.registry.client.package import PackageDatasetClient
    from harbor.tasks.client import TaskClient
    fixture_task(tmp_path)
    selected = PackageTaskId(org="terminal-bench", name=TASK_ID, ref="sha256:abc")
    other = PackageTaskId(org="terminal-bench", name="another-task", ref="sha256:def")
    async def metadata(self, dataset):
        assert dataset == DATASET
        return SimpleNamespace(task_ids=[other, selected], version="sha256:dataset")
    async def download(self, task_ids, **kwargs):
        assert task_ids == [selected] and kwargs == {"export": False}
        return SimpleNamespace(paths=[tmp_path])
    monkeypatch.setattr(PackageDatasetClient, "get_dataset_metadata", metadata)
    monkeypatch.setattr(TaskClient, "download_tasks", download)
    task, source = asyncio.run(vba.load_vba_task(load_config(CONFIG)))
    assert task.data.name == "terminal-bench/vba-userform-port"
    assert source["task_reference"]["ref"] == "sha256:abc"
    assert source["dataset_version"] == "sha256:dataset"


def test_original_dockerfiles_build_separate_images_with_stable_fingerprints(tmp_path, monkeypatch):
    async def scenario():
        task = fixture_task(tmp_path / "task")
        calls = []
        async def docker(*args):
            calls.append(args)
            return ProgramResult(1 if args[0] == "image" else 0, "", "")
        monkeypatch.setattr(vba, "docker", docker)
        first = await vba.prepare_vba_images(load_config(CONFIG), task, tmp_path / "out")
        second = await vba.prepare_vba_images(load_config(CONFIG), task, tmp_path / "out")
        assert first == second
        builds = [call for call in calls if call[0] == "build"]
        assert len(builds) == 4
        assert builds[0][-1].endswith("/environment") and builds[1][-1].endswith("/tests")
        recipe = Path(builds[0][builds[0].index("--file") + 1]).read_text()
        assert "COPY legacy_app" in recipe and "procps" in recipe and "/tests" not in recipe
        assert first["images"]["agent"] != first["images"]["verifier"]
    asyncio.run(scenario())


def test_published_amd64_images_are_preserved(tmp_path, monkeypatch):
    async def scenario():
        task = fixture_task(tmp_path / "task")
        config_path = tmp_path / "task/task.toml"
        config_path.write_text(config_path.read_text() + '\n[environment]\ndocker_image="agent@sha256:abc"\n[verifier.environment]\ndocker_image="grader@sha256:def"\n')
        builds = []
        async def docker(*args):
            if args[0] == "build": builds.append(args)
            return ProgramResult(1 if args[0] == "image" else 0, "", "")
        monkeypatch.setattr(vba, "docker", docker)
        images = await vba.prepare_vba_images(load_config(CONFIG), task, tmp_path / "out")
        assert images["images"]["verifier"] == "grader@sha256:def"
        assert len(builds) == 1
        recipe = Path(builds[0][builds[0].index("--file") + 1]).read_text()
        assert recipe.startswith("FROM agent@sha256:abc\n") and "COPY" not in recipe
    asyncio.run(scenario())


def test_prepare_only_connects_single_task_loader_and_both_images(tmp_path, monkeypatch):
    from harness_bloat_bench.multiagent import runner
    config = load_config(CONFIG)
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url):
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"data": [{"id": model.model} for model in (config.orchestrator, config.executor)]})
    async def load(config):
        return SimpleNamespace(data=SimpleNamespace(name="terminal-bench/" + TASK_ID)), {"dataset_version": "sha256:dataset"}
    async def prepare(config, task, output):
        return {"images": {"agent": "agent:test", "verifier": "grader:test"}, "rubric": rubric()}
    calls = []
    async def docker(*args):
        calls.append(args)
        if args[:2] == ("image", "inspect"):
            return ProgramResult(0, '[{"Id":"sha256:test","Architecture":"amd64"}]', "")
        return ProgramResult(0, "{}", "")
    monkeypatch.setattr(runner.httpx, "AsyncClient", Client)
    monkeypatch.setattr(runner, "load_vba_task", load)
    monkeypatch.setattr(runner, "prepare_vba_images", prepare)
    monkeypatch.setattr(runner, "prepare_artifact", lambda config: tmp_path)
    monkeypatch.setattr(runner, "configure_artifact_cache", lambda: None)
    monkeypatch.setattr(runner, "docker", docker)
    monkeypatch.setattr(runner, "HarborTaskset", lambda *args: pytest.fail("Must not load the entire 66-task dataset"))
    asyncio.run(runner.run_configs([config], tmp_path / "run", "", 4500, prepare_only=True))
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["shared_images"][config.name] == "agent:test"
    assert manifest["verifier_images"][config.name] == "grader:test"
    assert set(manifest["images"]) == {"agent:test", "grader:test"}
    assert manifest["task_environments"][config.name]["source_reference"]["dataset_version"] == "sha256:dataset"
    assert any(call[0] == "run" and "agent:test" in call for call in calls)
    assert not (tmp_path / "run/multiagent.sqlite").exists()


def make_run(path):
    path.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(path)
    config = load_config(CONFIG)
    gateway = SimpleNamespace(routes={}, inflight={})
    gateway.register = lambda route: gateway.routes.update({route.agent_id: route})
    task = SimpleNamespace(data=SimpleNamespace(prompt="Port the original VBA application"))
    benchmark = VbaRun(config, path, ledger, gateway, None, task, time.monotonic() + config.batch_timeout_seconds,
                      agent_image="agent:test", verifier_image="verifier:test")
    return benchmark, ledger


def test_single_task_uses_native_delegation_without_mcp(tmp_path, monkeypatch):
    async def scenario():
        benchmark, ledger = make_run(tmp_path)
        prompt = benchmark.coordinator_prompt()
        assert "Bottle" not in prompt and "PyPI" not in prompt and "four tasks" not in prompt
        assert "Port the original VBA application" in prompt
        assert "atomic-saves" not in prompt and "001_contract_meta" not in prompt
        assert not any(hasattr(benchmark, name) for name in ("mcp", "workers", "services", "active", "spawn_subagent", "wait_for_subagents"))
        assert "spawn_subagent" not in prompt
        assert prompt.endswith(benchmark.task.data.prompt)
        assert benchmark.create_container().config.workdir == "/workspace"
        class Container:
            async def start(self): pass
            async def stop(self): pass
        benchmark.create_container = Container
        stopped = []
        class Session:
            def __init__(self, *args): pass
            async def stop(self): stopped.append(True)
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.ContainerSession", Session)
        async def setup(*args): pass
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.setup_harness", setup)
        submitted = []
        async def grade(run):
            assert run is benchmark and stopped == [True]
            assert all(route.deadline <= time.monotonic() for route in benchmark.gateway.routes.values())
            submitted.append(True)
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.grade_vba", grade)
        async def launch(config, runtime, route, endpoint, prompt, mcp_url=None):
            assert mcp_url is None
            assert "Port the original VBA application" in prompt
            assert route.native_executor.model == config.executor
            assert route.native_executor.aggregate_native_workers
            assert route.native_executor.deadline == benchmark.execution_deadline
            return ProgramResult(0, "Changes ready", "")
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.launch_harness", launch)
        await benchmark.run("http://localhost")
        assert submitted == [True]
        assert "worker_spawn_count" not in benchmark.summary()
        assert "workers" not in benchmark.summary() and "services" not in benchmark.summary()
        assert benchmark.summary()["task_count"] == 1 and benchmark.summary()["scored_tasks"] == 0
        ledger.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_native_run_stops_cli_before_grading_and_skips_grading_on_cancellation(tmp_path, monkeypatch, cancelled):
    async def scenario():
        benchmark, ledger = make_run(tmp_path)
        lifecycle = []
        class Container:
            async def start(self): pass
            async def stop(self): lifecycle.append("container stopped")
        class Session:
            def __init__(self, *args): pass
            async def stop(self): lifecycle.append("CLI stopped")
        benchmark.create_container = Container
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.ContainerSession", Session)
        async def setup(*args): pass
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.setup_harness", setup)
        started = asyncio.Event()
        async def launch(*args):
            started.set()
            await asyncio.Event().wait()
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.launch_harness", launch)
        async def grade(run): lifecycle.append("graded")
        monkeypatch.setattr("harness_bloat_bench.multiagent.vba_run.grade_vba", grade)
        if not cancelled:
            benchmark.deadline = time.monotonic() + benchmark.config.grading_reserve_seconds + .05
        running = asyncio.create_task(benchmark.run("http://localhost"))
        await started.wait()
        if cancelled:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        else:
            await asyncio.wait_for(running, 1)
        assert lifecycle == (["CLI stopped", "container stopped"] if cancelled else ["CLI stopped", "graded", "container stopped"])
        assert benchmark.coordinator["state"] == "timed_out" and benchmark.phase == "finished"
        assert all(route.deadline <= time.monotonic() for route in benchmark.gateway.routes.values())
        assert (tmp_path / benchmark.config.name / "result.json").is_file()
        ledger.db.close()
    asyncio.run(scenario())


def test_report_presents_partial_credit_with_unscored_components(tmp_path):
    benchmark, ledger = make_run(tmp_path)
    benchmark.task_result.update(score_results([evidence("001_contract_meta")]))
    (tmp_path / "manifest.json").write_text('{"overall_timeout_seconds": 4500}')
    write_report(tmp_path, [benchmark.summary()], 10, {}, ledger.summarize())
    report = (tmp_path / "report.html").read_text()
    assert "10.0% (incomplete)" in report and "Subtask verification" in report
    assert "unmeasured" in report and "no benchmark MCP tools" in report
    saved = json.loads((tmp_path / "comparison.json").read_text())["teams"][0]
    assert saved["partial_reward"] == .1 and saved["pass_rate"] == 0
    assert saved["tasks"][0]["subtasks"][0]["verifier_state"] == "passed"
    ledger.db.close()


@pytest.mark.parametrize("first_code", [0, 124])
def test_each_check_restores_database_and_continues_after_failure_or_timeout(tmp_path, monkeypatch, first_code):
    app, tests, logs = (tmp_path / name for name in ("app", "tests", "logs"))
    (app / "backend/data").mkdir(parents=True)
    (app / "backend/data/app.sqlite").write_text("submitted database")
    (app / "run.sh").write_text("original startup")
    (tests / "traces").mkdir(parents=True)
    (tests / "bench-rubric.json").write_text(json.dumps({"subtasks": [{"trace_ids": ["api-a", "api-b", "ui"]}]}))
    for trace in ("api-a", "api-b", "ui"):
        (tests / "traces" / (trace + ".json")).write_text(json.dumps({"trace_id": trace, "steps": [{"action": "goto" if trace == "ui" else "api_reset"}]}))
    monkeypatch.setattr(vba_verifier, "prepare_backend", lambda *args: 0)
    # The frontend fails installation, but both backend checks must still run.
    monkeypatch.setattr(vba_verifier, "prepare_frontend", lambda *args: 1)
    visited = []
    def command(argv, log, timeout, **kwargs):
        trace = Path(argv[argv.index("--trace") + 1]).stem
        assert (app / "backend/data/app.sqlite").read_text() == "submitted database"
        assert (app / "run.sh").read_text() == "original startup"
        (app / "backend/data/app.sqlite").write_text("mutated")
        (app / "run.sh").write_text("backend-only wrapper")
        visited.append(trace)
        if trace == "api-a" and first_code:
            return first_code
        result = Path(argv[argv.index("--result") + 1])
        result.write_text(json.dumps({"trace_id": trace, "passed": trace == "api-b"}))
        assert timeout > 0
        return 0
    monkeypatch.setattr(vba_verifier, "command", command)
    results = vba_verifier.run_checks(app, tests, logs, 30)
    assert visited == ["api-a", "api-b"]
    assert results[0]["status"] == ("unscored" if first_code else "failed")
    assert results[1]["passed"] is True and results[2]["passed"] is False
    assert json.loads((logs / "trace_results.json").read_text()) == results


@pytest.mark.parametrize("interruption", [None, TimeoutError, asyncio.CancelledError])
@pytest.mark.parametrize("remaining_seconds", [4500, 30])
def test_grading_cap_early_completion_and_checkpointed_credit(tmp_path, monkeypatch, interruption, remaining_seconds):
    async def scenario():
        benchmark, ledger = make_run(tmp_path)
        benchmark.deadline = time.monotonic() + remaining_seconds
        benchmark.container = SimpleNamespace(name="agents", run=lambda *args: None)
        async def run(*args): return ProgramResult(0, "", "")
        benchmark.container.run = run
        stopped = []
        class Grader:
            def __init__(self, *args): self.name = "verifier"
            async def start(self): pass
            async def write(self, *args): pass
            async def run(self, *args): return ProgramResult(0, "", "")
            async def read(self, path): return json.dumps([evidence("001_contract_meta")]).encode()
            async def stop(self): stopped.append("container")
        class Session:
            def __init__(self, *args): pass
            async def run_program(self, argv, env):
                budget = float(argv[argv.index("--timeout-seconds") + 1])
                expected_budget = min(3600, remaining_seconds)
                assert expected_budget - 5 <= budget <= expected_budget
                assert self.program_timeout == budget
                if interruption:
                    raise interruption()
                return ProgramResult(0, "Completed immediately", "")
            async def stop(self): stopped.append("session")
        async def docker(*args):
            if args[1] == "agents:/tmp/bench-submission.tgz":
                Path(args[2]).write_bytes(b"frozen submission")
            return ProgramResult(0, "", "")
        monkeypatch.setattr(vba, "BenchRuntime", Grader)
        monkeypatch.setattr(vba, "ContainerSession", Session)
        monkeypatch.setattr(vba, "docker", docker)
        if interruption is asyncio.CancelledError:
            with pytest.raises(asyncio.CancelledError):
                await vba.grade_vba(benchmark)
        else:
            await asyncio.wait_for(vba.grade_vba(benchmark), timeout=1)
        assert stopped == ["session", "container"]
        row = benchmark.task_result
        assert row["state"] == ("timed_out" if interruption else "unscored") and row["reward"] == .1
        assert row["verifier_state"] == "partial" and not row["score_complete"]
        saved = json.loads((tmp_path / benchmark.config.name / "result.json").read_text())
        assert saved["partial_reward"] == .1
        ledger.db.close()
    asyncio.run(scenario())
