"""Fresh multiagent CLI configuration using only the shared artifact installers."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from verifiers.v1.runtimes import DockerConfig, DockerRuntime, ProgramResult
from verifiers.v1.runtimes.docker import docker

from harness_bench.harness_cache import (
    CONTAINER_CACHE_ROOT,
    cache_root,
    ensure_docker_built_tree,
    ensure_opencode_cached,
    install_cached_python_runtime_script,
    stage_cached_file,
    stage_cached_tree,
)
from harness_bench.harnesses.hermes_agent import HERMES_BIN, HERMES_DIR, HERMES_PYTHON_DEPENDENCY_DIR, _install_script
from harness_bench.harnesses.opencode import OPENCODE_BIN

from .accounting import scrub
from .config import TeamConfig
from .gateway import AgentRoute


def configure_artifact_cache():
    """Keep recipe invalidation and cold builds outside the single-agent cache."""
    original = cache_root()
    target = Path(os.environ.get("MULTIAGENT_HARNESS_CACHE_DIR", str(original / "multiagent"))).resolve()
    target.mkdir(parents=True, exist_ok=True)
    # Release binaries are immutable; copying them does not invalidate recipes.
    for source in original.glob("opencode/linux-*/*/opencode"):
        destination = target / source.relative_to(original)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    os.environ["HARNESS_BLOAT_CACHE_DIR"] = str(target)
    os.environ["MULTIAGENT_HARNESS_CACHE_DIR"] = str(target)


@lru_cache(maxsize=1)
def bridge_hostname():
    override = os.environ.get("MULTIAGENT_DOCKER_HOSTNAME")
    if override:
        return override
    if sys.platform == "darwin":
        context = os.environ.get("DOCKER_CONTEXT", "")
        if not context:
            context = subprocess.run(["docker", "context", "show"], capture_output=True, text=True, timeout=5).stdout.strip()
        if context.startswith("colima") or ".colima" in os.environ.get("DOCKER_HOST", ""):
            return "host.lima.internal"
    return "host.docker.internal"


def prepare_artifact(config: TeamConfig) -> Path:
    arch = "x64" if config.sandbox.platform == "linux/amd64" else "arm64"
    if config.harness.id == "opencode":
        return ensure_opencode_cached(config.harness.version, config.harness.version, arch)
    # Bullseye security packages have moved to the official Debian archive.
    mirror_setup = "sed -i 's|http://deb.debian.org/debian-security|https://archive.debian.org/debian-security|g;s|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list\n"
    return ensure_docker_built_tree(harness="hermes-agent", version=config.harness.version, source_dir=HERMES_DIR, install_script=mirror_setup + _install_script(config.harness.version), arch=arch)


class BenchRuntime(DockerRuntime):
    def __init__(self, config: DockerConfig, name: str, platform: str, output: Path):
        super().__init__(config, name)
        self.platform, self.output = platform, output
        self.program_timeout = 1200.0
        self._harness_cache_mounted = True

    async def start(self):
        self._container = self.name
        result = await docker(
            "run", "--detach", "--platform", self.platform,
            "--network", "bridge",
            *([] if sys.platform == "darwin" else ["--add-host", "host.docker.internal:host-gateway"]),
            "--cpus", str(self.config.cpu), "--memory", f"{self.config.memory}g",
            "--mount", f"type=bind,source={cache_root()},target={CONTAINER_CACHE_ROOT},readonly",
            "--workdir", self.config.workdir, "--entrypoint", "sleep", "--name", self.name,
            self.config.image, "infinity",
        )
        if result.exit_code:
            raise RuntimeError("Docker sandbox failed: " + result.stderr.strip())
        self.info.id = result.stdout.strip()[:12]

    def host_url(self, url: str) -> str:
        return url.replace("127.0.0.1", bridge_hostname())

    async def run_program(self, argv, env):
        return await self._stream_program(argv, env, self.output, self.program_timeout)

    async def _stream_program(self, argv, env, directory, timeout):
        directory.mkdir(parents=True, exist_ok=True)
        env_args = [arg for key, value in env.items() for arg in ("--env", f"{key}={value}")]
        timeout_args = [] if timeout is None else ["timeout", "--signal=TERM", "--kill-after=2s", f"{max(0.1, timeout):.3f}s"]
        process = await asyncio.create_subprocess_exec(
            "docker", "exec", *env_args, "--workdir", self.config.workdir, self.name,
            *timeout_args, *argv,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        async def pump(stream, path):
            with path.open("wb") as handle:
                while chunk := await stream.read(65536):
                    handle.write(chunk)
                    handle.flush()
        out, err = directory / "stdout.log", directory / "stderr.log"
        pumps = [asyncio.create_task(pump(process.stdout, out)), asyncio.create_task(pump(process.stderr, err))]
        try:
            code = await process.wait()
            await asyncio.gather(*pumps)
            return ProgramResult(code, scrub(out.read_text(errors="replace")), scrub(err.read_text(errors="replace")))
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
            for task in pumps:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)

    async def run(self, argv, env):
        if argv == ["sh", "-c", "cd /tests && bash test.sh"]:
            directory = self.output / "verifier"
            # Preserve grading evidence even if its absolute deadline interrupts
            # package setup or tests. The worker's lifecycle timeout governs it.
            result = await self._stream_program(argv, env, directory, None)
            (directory / "process.json").write_text(json.dumps({"exit_code": result.exit_code}))
            return result
        return await super().run(argv, env)

    async def inspect_resources(self):
        result = await docker("inspect", self.name, "--format", "{{json .State}}")
        if result.exit_code:
            return {}
        state = json.loads(result.stdout)
        return {"container_id": self.info.id, "oom_killed": state.get("OOMKilled"), "container_memory_limit_gb": self.config.memory, "container_cpu_limit": self.config.cpu}


async def setup_harness(config: TeamConfig, runtime: BenchRuntime, artifact: Path):
    if config.harness.id == "opencode":
        await stage_cached_file(runtime, artifact, OPENCODE_BIN)
        check = await runtime.run([OPENCODE_BIN, "--version"], {})
    else:
        await stage_cached_tree(runtime, artifact, HERMES_DIR)
        check = await runtime.run(["sh", "-c", install_cached_python_runtime_script(HERMES_DIR, HERMES_PYTHON_DEPENDENCY_DIR) + f"\nLD_LIBRARY_PATH={shlex.quote(HERMES_PYTHON_DEPENDENCY_DIR)} {shlex.quote(HERMES_BIN)} --version"], {})
    if check.exit_code:
        raise RuntimeError(f"Harness setup exited {check.exit_code}: " + ((check.stderr or check.stdout)[-2000:] or "no output (check container memory and binary platform)"))


async def launch_harness(config: TeamConfig, runtime: BenchRuntime, route: AgentRoute, endpoint: str, prompt: str, mcp_url: str | None = None):
    state = "/tmp/bench-agent-state"
    endpoint = runtime.host_url(endpoint)
    credential = "BENCH_AGENT_KEY"
    env = {credential: route.token}
    if config.harness.id == "opencode":
        provider = {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": endpoint, "apiKey": "{env:" + credential + "}"}, "models": {"agent": {"name": route.model.model, "limit": {"context": 1048576, "output": route.model.max_output_tokens}}}}
        cli_config = {"provider": {"bench": provider}, "permission": {"*": "allow", "task": "deny"}, "tools": {"task": False}, "share": "disabled"}
        if mcp_url:
            cli_config.update(
                permission={"*": "deny", "bench_*": "allow"},
                tools={"*": False, "bench_*": True},
                mcp={"bench": {"type": "remote", "url": runtime.host_url(mcp_url), "timeout": 70000}},
            )
        await runtime.write(state + "/opencode.json", json.dumps(cli_config).encode())
        env.update(OPENCODE_CONFIG=state + "/opencode.json", XDG_CONFIG_HOME=state + "/config", XDG_DATA_HOME=state + "/data", XDG_STATE_HOME=state + "/state")
        argv = [OPENCODE_BIN, "run", "--auto", "--format", "json", "--model", "bench/agent", prompt]
    else:
        cli_config = {
            "model": {"default": route.model.model, "provider": "custom", "base_url": endpoint, "api_key": "${" + credential + "}"},
            "agent": {"max_turns": route.model.max_calls, "system_prompt": "You are a benchmark agent. Follow the user's task precisely. Do not inspect or modify hidden benchmark tests."},
            "terminal": {"backend": "local", "cwd": "/app"},
        }
        if mcp_url:
            cli_config["mcp_servers"] = {"bench": {"url": runtime.host_url(mcp_url)}}
        await runtime.write(state + "/config.yaml", json.dumps(cli_config).encode())
        env.update(HERMES_HOME=state, HERMES_SKIP_NODE_BOOTSTRAP="1", LD_LIBRARY_PATH=HERMES_PYTHON_DEPENDENCY_DIR, HERMES_INTERACTIVE="0")
        argv = [HERMES_BIN, "chat", "--model", route.model.model, "--yolo", "--toolsets", "mcp-bench" if mcp_url else "terminal", "--query", prompt]
    return await runtime.run_program(argv, env)
