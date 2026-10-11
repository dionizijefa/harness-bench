"""Build the official VBA environments and grade a frozen app submission."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import time
import tomllib
from uuid import uuid4

from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.runtimes.docker import docker

from .accounting import scrub, utc_now
from .harnesses import BenchRuntime, ContainerSession
from .vba_rubric import APP_ROOT, EXCLUDES, TASK_ID, rubric, score_results, validate_traces


async def load_vba_task(config):
    """Resolve the pinned dataset but download only its one selected task."""
    from harbor.registry.client.package import PackageDatasetClient
    from harbor.tasks.client import TaskClient
    from verifiers.v1.tasksets.harbor import HarborConfig
    from verifiers.v1.tasksets.harbor.taskset import HarborTask, parse_task

    metadata = await PackageDatasetClient().get_dataset_metadata(config.dataset)
    selected = [task for task in metadata.task_ids if task.get_name().rsplit("/", 1)[-1] == TASK_ID]
    if len(selected) != 1:
        raise ValueError("Pinned dataset must contain exactly one vba-userform-port task")
    downloaded = await TaskClient().download_tasks(selected, export=False)
    if len(downloaded.paths) != 1:
        raise ValueError("Harbor did not download exactly one VBA task")
    # The task Dockerfiles are built explicitly in prepare_vba_images.
    data = parse_task(downloaded.paths[0], 0, HarborConfig(dataset=config.dataset, tasks=[TASK_ID], ignore_dockerfile=True))
    return HarborTask(data), {"dataset_version": metadata.version, "task_reference": selected[0].model_dump(mode="json")}


def tree_hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


async def prepare_vba_images(config, task, output: Path) -> dict:
    source = Path(task.data.task_dir)
    validate_traces(source)
    for directory in ("environment", "tests"):
        if not (source / directory / "Dockerfile").is_file():
            raise ValueError(f"VBA task is missing its {directory}/Dockerfile")
    if tree_hashes(source / "environment/legacy_app") != tree_hashes(source / "tests/legacy_app"):
        raise ValueError("VBA agent and verifier legacy inputs must be identical")
    snapshot = output / "task-sources" / TASK_ID
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        for directory in ("environment", "tests"):
            shutil.copytree(source / directory, snapshot / directory)
        for name in ("instruction.md", "task.toml"):
            shutil.copy2(source / name, snapshot / name)
    hashes = tree_hashes(snapshot)
    hashes.pop("sha256.json", None)
    (snapshot / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    task_config = tomllib.loads((source / "task.toml").read_text())
    published = {
        "agent": task_config.get("environment", {}).get("docker_image"),
        "verifier": task_config.get("verifier", {}).get("environment", {}).get("docker_image"),
    }
    metadata = {"source_sha256": hashes, "rubric": rubric(), "images": {}, "published_images": published}
    for role, directory in (("agent", "environment"), ("verifier", "tests")):
        # The Hub exports digest-pinned amd64 images. Use those when running the
        # published architecture; local/native smoke tests build from source.
        base = published[role] if config.sandbox.platform == "linux/amd64" else None
        if base and role == "verifier":
            metadata["images"][role] = base
            continue
        recipe = f"FROM {base}\nWORKDIR /workspace\n" if base else (source / directory / "Dockerfile").read_text()
        if role == "agent":
            # Process management and harness tooling; preserve the task's
            # original Python, Node, input files and workspace.
            recipe += "\nUSER root\nRUN apt-get update && apt-get install -y --no-install-recommends git procps util-linux && rm -rf /var/lib/apt/lists/*\n"
        digest = hashlib.sha256((recipe + config.sandbox.platform + json.dumps(hashes, sort_keys=True)).encode()).hexdigest()[:20]
        tag = f"harness-bench-vba-{role}:{digest}"
        recipe_path = output / "vba-environments" / digest / "Dockerfile"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(recipe)
        check = await docker("image", "inspect", tag)
        if check.exit_code:
            built = await docker("build", "--platform", config.sandbox.platform, "--tag", tag,
                                 "--file", str(recipe_path), str(snapshot / directory))
            recipe_path.with_suffix(".log").write_text(built.stdout + built.stderr)
            if built.exit_code:
                raise RuntimeError(f"VBA {role} image build failed; see {recipe_path.with_suffix('.log')}")
        metadata["images"][role] = tag
    return metadata


async def grade_vba(run) -> None:
    """After agents stop, copy only app artifacts into the clean verifier image."""
    row = run.task_result
    directory = run.output / run.config.name / "tasks" / TASK_ID / "verifier"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Grading starts on submission and exits as soon as all checks finish.
    # Early submission must not turn unused agent time into a longer grader cap.
    grading_deadline = min(run.deadline, started + run.config.grading_reserve_seconds)
    row.update(state="scoring", grading_started_at=utc_now())
    sandbox = run.config.sandbox
    grader = BenchRuntime(DockerConfig(image=run.verifier_image, workdir="/workspace", cpu=sandbox.shared_cpus, memory=sandbox.shared_memory_gb),
                          "mab-vba-grade-" + uuid4().hex[:12], sandbox.platform, directory)
    run.runtimes.append(grader)
    session = ContainerSession(grader, directory, "/workspace")
    ready = False
    try:
        async with asyncio.timeout_at(grading_deadline):
            archive = directory.parent / "submission.tgz"
            await run.container.run(["mkdir", "-p", APP_ROOT], {})
            packed = await run.container.run(["tar", "-czf", "/tmp/bench-submission.tgz",
                                               *("--exclude=" + item for item in EXCLUDES), "-C", "/workspace", "generated_app"], {})
            if packed.exit_code:
                raise RuntimeError("Cannot snapshot submitted app: " + packed.stderr[-1000:])
            copied = await docker("cp", run.container.name + ":/tmp/bench-submission.tgz", str(archive))
            if copied.exit_code:
                raise RuntimeError("Cannot copy submitted app: " + copied.stderr[-1000:])
            row["submission_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            await grader.start()
            ready = True
            copied = await docker("cp", str(archive), grader.name + ":/tmp/submission.tgz")
            if copied.exit_code:
                raise RuntimeError("Cannot stage submitted app: " + copied.stderr[-1000:])
            unpacked = await grader.run(["tar", "-xzf", "/tmp/submission.tgz", "-C", "/workspace"], {})
            if unpacked.exit_code:
                raise RuntimeError("Cannot unpack submitted app: " + unpacked.stderr[-1000:])
            await grader.write("/tests/bench_partial.py", Path(__file__).with_name("vba_verifier.py").read_bytes())
            await grader.write("/tests/bench-rubric.json", json.dumps(rubric()).encode())
            remaining = max(.1, grading_deadline - time.monotonic() - 3)
            session.program_timeout = remaining
            result = await session.run_program(["/opt/vba-verifier-venv/bin/python", "/tests/bench_partial.py",
                                                "--timeout-seconds", str(remaining)], {})
            (directory / "process.json").write_text(json.dumps({"exit_code": result.exit_code}) + "\n")
            if result.exit_code:
                row["error"] = f"Partial verifier exited {result.exit_code}; retaining completed checks"
    except asyncio.CancelledError:
        row.update(state="timed_out", error="Grading cancelled; retaining completed checks")
        raise
    except TimeoutError:
        row.update(state="timed_out", error="Grading deadline reached; retaining completed checks")
    except Exception as error:
        row.update(state="error", error=scrub(str(error)))
    finally:
        with contextlib.suppress(Exception):
            await session.stop()
        evidence = None
        if ready:
            try:
                async with asyncio.timeout(3):
                    evidence = json.loads(await grader.read("/logs/verifier/trace_results.json"))
                    if not isinstance(evidence, list):
                        raise ValueError("Expected a trace result list")
                    scores = score_results(evidence)
                    row.update(scores)
                    row["passed"] = scores["score_complete"] and scores["passed_checks"] == scores["check_count"]
                    if not scores["score_complete"] and scores["scored_checks"] == 0:
                        row["passed"] = None
                    row["verifier_state"] = "passed" if row["passed"] else "partial" if scores["reward"] > 0 else "failed" if scores["score_complete"] else "unscored"
                    if row["state"] == "scoring":
                        row["state"] = "completed" if scores["score_complete"] else "unscored"
                    (directory / "trace_results.json").write_text(json.dumps(evidence, indent=2) + "\n")
            except Exception as error:
                row.update(state="error", verifier_state="error", error="Invalid or missing partial verifier evidence: " + scrub(str(error)))
            with contextlib.suppress(Exception):
                async with asyncio.timeout(3):
                    await docker("cp", grader.name + ":/logs/verifier/.", str(directory))
        row.setdefault("verifier_state", "unscored")
        if row["state"] == "scoring":
            row.update(state="unscored", error="Grading ended without evidence")
        # The runner computes the reward; no submitted app reward file is trusted.
        (directory / "subtasks.json").write_text(json.dumps(row["subtasks"], indent=2) + "\n")
        (directory / "reward.json").write_text(json.dumps({"reward": row["reward"], "score_complete": row["score_complete"]}) + "\n")
        for subtask in row["subtasks"]:
            run.event("subtask_graded", task_id=TASK_ID, **subtask)
        row.update(grading_seconds=time.monotonic() - started, ended_at=utc_now())
        run.event("task_graded", **row)
        run.write_result()
        await grader.stop()
