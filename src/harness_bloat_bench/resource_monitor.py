"""Per-rollout Docker resource accounting using the host's cgroup v2 counters."""

import asyncio
import subprocess
from pathlib import Path

import verifiers.v1.runtimes as vf_runtimes
from verifiers.v1.runtimes.docker import DockerRuntime as BaseDockerRuntime

_last_usage: dict = {}
_monitor_installed = False


def reset_resource_usage() -> None:
    global _last_usage
    _last_usage = {}


def consume_resource_usage() -> dict:
    global _last_usage
    usage = _last_usage
    _last_usage = {}
    return usage


def _read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        fields = line.split()
        for field in fields[1:] if ":" in fields[0] else fields:
            key, separator, value = field.partition("=")
            if separator and value.isdigit():
                values[key] = values.get(key, 0) + int(value)
        if len(fields) == 2 and fields[1].isdigit():
            values[fields[0]] = int(fields[1])
    return values


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def _read_cgroup_usage(cgroup_path: Path) -> dict:
    cpu = _read_key_values(cgroup_path / "cpu.stat")
    memory_events = _read_key_values(cgroup_path / "memory.events")
    io = _read_key_values(cgroup_path / "io.stat")
    usage = {
        "resource_usage_source": "cgroup_v2",
        "cpu_seconds": cpu.get("usage_usec", 0) / 1_000_000,
        "peak_memory_bytes": _read_int(cgroup_path / "memory.peak"),
        "io_read_bytes": io.get("rbytes"),
        "io_write_bytes": io.get("wbytes"),
        "peak_pids": _read_int(cgroup_path / "pids.peak"),
        "oom_kill_count": memory_events.get("oom_kill", 0),
    }
    return {key: value for key, value in usage.items() if value is not None}


def _container_cgroup_path(container: str) -> Path | None:
    try:
        inspected = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if inspected.returncode != 0 or not inspected.stdout.strip().isdigit():
        return None
    pid = inspected.stdout.strip()
    try:
        entries = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    unified = next((line.partition("::")[2] for line in entries if "::" in line), "")
    return Path("/sys/fs/cgroup") / unified.lstrip("/") if unified else None


def _capture_container_usage(container: str | None) -> dict:
    if not container or (path := _container_cgroup_path(container)) is None:
        return {}
    return _read_cgroup_usage(path)


class MonitoredDockerRuntime(BaseDockerRuntime):
    async def teardown(self) -> None:
        global _last_usage
        try:
            try:
                _last_usage = await asyncio.to_thread(
                    _capture_container_usage, self._container
                )
            except Exception:
                # Accounting must never prevent Verifiers from removing a sandbox.
                _last_usage = {}
        finally:
            await super().teardown()


def enable_docker_resource_monitoring() -> None:
    """Make Verifiers use the accounting runtime in this rollout worker process."""
    global _monitor_installed
    if not _monitor_installed:
        vf_runtimes.DockerRuntime = MonitoredDockerRuntime
        _monitor_installed = True
