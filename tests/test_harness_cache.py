from pathlib import Path

import asyncio
import subprocess

from verifiers.v1.runtimes import DockerRuntime, ProgramResult
from verifiers.v1.runtimes.docker import DockerConfig

from harness_bloat_bench import harness_cache
from harness_bloat_bench import resource_monitor


def test_omp_cache_path_is_versioned_by_linux_architecture(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_CACHE_DIR", str(tmp_path))

    assert harness_cache.omp_cache_path("v17.2.10", "amd64") == (
        tmp_path / "omp" / "linux-x64" / "17.2.10" / "omp"
    )
    assert harness_cache.omp_release_url("17.2.10", "x86_64").endswith(
        "/v17.2.10/omp-linux-x64"
    )


def test_ensure_omp_cached_downloads_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_CACHE_DIR", str(tmp_path))
    downloads = 0

    def fake_download(url: str, destination: Path) -> dict[str, object]:
        nonlocal downloads
        downloads += 1
        assert url.endswith("/v17.2.10/omp-linux-x64")
        destination.write_bytes(b"release binary")
        return {"bytes": 14, "sha256": "test"}

    monkeypatch.setattr(harness_cache, "_download", fake_download)

    first = harness_cache.ensure_omp_cached("17.2.10", "x64")
    second = harness_cache.ensure_omp_cached("17.2.10", "x64")

    assert first == second
    assert first.read_bytes() == b"release binary"
    assert downloads == 1


def test_stage_cached_file_uses_runtime_write_for_non_docker(tmp_path: Path) -> None:
    source = tmp_path / "artifact"
    source.write_bytes(b"cached bytes")

    class FakeRuntime:
        async def write(self, path: str, data: bytes) -> None:
            self.write_call = (path, data)

    runtime = FakeRuntime()
    asyncio.run(harness_cache.stage_cached_file(runtime, source, "/tmp/artifact"))

    assert runtime.write_call == ("/tmp/artifact", b"cached bytes")


def test_composite_cache_reuses_matching_recipe_and_rebuilds_stale_recipe(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_CACHE_DIR", str(tmp_path))
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        commands.append(command)
        if command[1] == "cp":
            copied = Path(command[-1])
            copied.mkdir()
            (copied / "harness").write_bytes(b"installed")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(harness_cache.subprocess, "run", fake_run)

    first = harness_cache.ensure_docker_built_tree(
        harness="composite",
        version="1.0.0",
        source_dir="/tmp/composite",
        install_script="install version one",
        arch="x64",
    )
    first_command_count = len(commands)
    second = harness_cache.ensure_docker_built_tree(
        harness="composite",
        version="1.0.0",
        source_dir="/tmp/composite",
        install_script="install version one",
        arch="x64",
    )

    assert first == second
    assert len(commands) == first_command_count
    assert "--platform" in commands[0]
    assert "linux/amd64" in commands[0]

    harness_cache.ensure_docker_built_tree(
        harness="composite",
        version="1.0.0",
        source_dir="/tmp/composite",
        install_script="install version two",
        arch="x64",
    )

    assert len(commands) > first_command_count


def test_mounted_docker_cache_links_files_and_trees_without_copying(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_CACHE_DIR", str(tmp_path))
    cached_file = tmp_path / "omp" / "linux-x64" / "1.0.0" / "omp"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_bytes(b"binary")
    cached_tree = tmp_path / "prime-agent" / "linux-x64" / "1.0.0" / "tree"
    cached_tree.mkdir(parents=True)
    commands: list[list[str]] = []

    class MountedRuntime(DockerRuntime):
        def __init__(self) -> None:
            self._harness_cache_mounted = True

        async def run(self, argv, env):
            commands.append(argv)
            return ProgramResult(exit_code=0, stdout="", stderr="")

    runtime = MountedRuntime()

    async def stage() -> None:
        await harness_cache.stage_cached_file(runtime, cached_file, "/tmp/omp")
        await harness_cache.stage_cached_tree(
            runtime, cached_tree, "/tmp/vf-prime-agent"
        )

    asyncio.run(stage())

    flattened = "\n".join(" ".join(command) for command in commands)
    assert "/var/cache/harness-bloat/omp/linux-x64/1.0.0/omp" in flattened
    assert "/var/cache/harness-bloat/prime-agent/linux-x64/1.0.0/tree" in flattened
    assert "docker cp" not in flattened


def test_monitored_docker_runtime_mounts_the_host_cache_read_only(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HARNESS_BLOAT_CACHE_DIR", str(tmp_path))
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str) -> ProgramResult:
        calls.append(args)
        stdout = "29.2.1" if args[0] == "version" else "abc1234567890"
        return ProgramResult(exit_code=0, stdout=stdout, stderr="")

    monkeypatch.setattr(resource_monitor, "docker", fake_docker)
    runtime = resource_monitor.MonitoredDockerRuntime(
        DockerConfig(image="task-image"), name="rollout-test"
    )

    asyncio.run(runtime.start())

    run = next(call for call in calls if call[0] == "run")
    volume = run[run.index("--volume") + 1]
    assert volume == f"{tmp_path}:/var/cache/harness-bloat:ro"
    assert runtime._harness_cache_mounted is True
