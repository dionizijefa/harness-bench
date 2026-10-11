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
from types import SimpleNamespace
from uuid import uuid4

from verifiers.v1.runtimes import DockerConfig, DockerRuntime, ProgramResult
from verifiers.v1.runtimes.docker import docker

from harness_bloat_bench.harness_cache import (
    CONTAINER_CACHE_ROOT,
    cache_root,
    ensure_docker_built_tree,
    ensure_opencode_cached,
    ensure_omp_cached,
    ensure_claude_cached,
    ensure_codex_cached,
    install_cached_python_runtime_script,
    stage_cached_file,
    stage_cached_tree,
)
from harness_bloat_bench.harnesses.hermes_agent import HERMES_BIN, HERMES_DIR, HERMES_PYTHON_DEPENDENCY_DIR, _install_script
from harness_bloat_bench.harnesses.opencode import OPENCODE_BIN
from harness_bloat_bench.harnesses.omp_agent import OmpAgentHarness, OmpAgentHarnessConfig
from harness_bloat_bench.harnesses.claude_code import ClaudeCodeHarness, ClaudeCodeHarnessConfig
from harness_bloat_bench.harnesses.codex_agent import CodexAgentHarness, CodexHarnessConfig
from harness_bloat_bench.harnesses._common import openai_compat_model_config

from .accounting import scrub
from .config import TeamConfig
from .gateway import AgentRoute


EXISTING_ADAPTERS = {
    "omp_agent": (OmpAgentHarness, OmpAgentHarnessConfig),
    "claude_code_agent": (ClaudeCodeHarness, ClaudeCodeHarnessConfig),
    "codex_agent": (CodexAgentHarness, CodexHarnessConfig),
}


def existing_adapter(config, env=None):
    adapter, spec = EXISTING_ADAPTERS[config.harness.id]
    return adapter(spec(id=config.harness.id, version=config.harness.version, env=env or {}))


def configure_artifact_cache():
    """Keep recipe invalidation and cold builds outside the single-agent cache."""
    original = cache_root()
    target = Path(os.environ.get("MULTIAGENT_HARNESS_CACHE_DIR", str(original / "multiagent"))).resolve()
    target.mkdir(parents=True, exist_ok=True)
    # Release binaries are immutable; copying them does not invalidate recipes.
    for harness, filename in (("opencode", "opencode"), ("omp", "omp"), ("claude-code", "claude"), ("codex", "codex")):
        for source in original.glob(f"{harness}/linux-*/*/{filename}"):
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
    installers = {"omp_agent": ensure_omp_cached, "claude_code_agent": ensure_claude_cached, "codex_agent": ensure_codex_cached}
    if config.harness.id in installers:
        return installers[config.harness.id](config.harness.version, arch=arch)
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
            "run", "--detach", "--init", "--platform", self.platform,
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

    async def _stream_program(self, argv, env, directory, timeout, *, cleanup=None):
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
            if cleanup:
                await cleanup()
            await asyncio.gather(*pumps)
            return ProgramResult(code, scrub(out.read_text(errors="replace")), scrub(err.read_text(errors="replace")))
        finally:
            if cleanup:
                await cleanup()
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

    async def stop_process_group(self, marker):
        async with asyncio.timeout(3):
            await docker("exec", self.name, "bash", "-c", '''
if [ -f "$1" ]; then
    read -r pid < "$1"
    case "$pid" in ''|*[!0-9]*) exit 1;; esac
    if [ "$pid" -gt 1 ]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        sleep 0.1
        kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    rm -f -- "$1"
fi
''', "bench-stop-service", marker)

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


class ContainerSession(BenchRuntime):
    """Independent agent state and processes without ownership of the container."""

    def __init__(self, owner: BenchRuntime, output: Path, workdir="/app"):
        super().__init__(owner.config.model_copy(update={"workdir": workdir}), owner.name, owner.platform, output)
        self.owner = owner
        self._container = owner.name
        self.info = owner.info
        self._process_tokens: set[str] = set()

    async def start(self):
        raise RuntimeError("Only the owner can start the shared container")

    def cleanup(self):
        # Runtime finalizers must never remove another agent's shared container.
        pass

    async def stop(self):
        for token in list(self._process_tokens):
            await self._stop_program(token)

    async def _stop_program(self, token):
        if token not in self._process_tokens:
            return
        # Native terminal tools may create their own process groups. An inherited
        # environment marker also finds those descendants, without touching the
        # other agents or services launched through independent docker execs.
        async with asyncio.timeout(3):
            result = await docker("exec", self.name, "bash", "-c", '''
pids=()
for file in /proc/[0-9]*/environ; do
    if grep -zFqx -- "BENCH_PROCESS_TOKEN=$1" "$file" 2>/dev/null; then
        pid=${file#/proc/}; pid=${pid%/environ}
        if [ "$pid" -gt 1 ]; then pids+=("$pid"); fi
    fi
done
if [ "${#pids[@]}" -gt 0 ]; then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 0.1
    kill -KILL "${pids[@]}" 2>/dev/null || true
fi
''', "bench-stop-agent", token)
        if result.exit_code:
            raise RuntimeError("Cannot stop shared session processes: " + result.stderr[-1000:])
        self._process_tokens.discard(token)

    async def _stream_program(self, argv, env, directory, timeout):
        token = uuid4().hex
        self._process_tokens.add(token)
        try:
            return await super()._stream_program(argv, {**env, "BENCH_PROCESS_TOKEN": token}, directory, timeout,
                                                 cleanup=lambda: self._stop_program(token))
        finally:
            await self._stop_program(token)


async def setup_harness(config: TeamConfig, runtime: BenchRuntime, artifact: Path):
    if config.harness.id in EXISTING_ADAPTERS:
        await existing_adapter(config).setup(runtime, artifact=artifact)
        return
    if config.harness.id == "opencode":
        await stage_cached_file(runtime, artifact, OPENCODE_BIN)
        check = await runtime.run([OPENCODE_BIN, "--version"], {})
    else:
        await stage_cached_tree(runtime, artifact, HERMES_DIR, writable=True)
        install = install_cached_python_runtime_script(HERMES_DIR, HERMES_PYTHON_DEPENDENCY_DIR, venv_dir=HERMES_DIR + "/source/venv")
        check = await runtime.run(["sh", "-c", install + f"\nLD_LIBRARY_PATH={shlex.quote(HERMES_PYTHON_DEPENDENCY_DIR)} {shlex.quote(HERMES_BIN)} --version"], {})
    if check.exit_code:
        raise RuntimeError(f"Harness setup exited {check.exit_code}: " + ((check.stderr or check.stdout)[-2000:] or "no output (check container memory and binary platform)"))


async def launch_harness(config: TeamConfig, runtime: BenchRuntime, route: AgentRoute, endpoint: str, prompt: str, mcp_url: str | None = None):
    state = "/tmp/bench-agent-state/" + route.agent_id
    endpoint = runtime.host_url(endpoint)
    credential = "BENCH_AGENT_KEY"
    env = {credential: route.token}
    native = config.scoring == "vba_subtasks_v1"
    executor = route.native_executor
    if native:
        if mcp_url is not None or executor is None:
            raise ValueError("Native VBA runs require an executor route and no benchmark MCP connection")
        executor_endpoint = endpoint.replace(f"/agents/{route.agent_id}/", f"/agents/{executor.agent_id}/")
        env["BENCH_EXECUTOR_KEY"] = executor.token
    if config.harness.id in EXISTING_ADAPTERS:
        if not native:
            raise ValueError("These adapters are wired only for native VBA runs")
        # Reuse each adapter's stock CLI launch with one task prompt. Only model
        # selection and isolated provider credentials differ from a single run.
        trace = SimpleNamespace(id=route.agent_id, task=SimpleNamespace(data=SimpleNamespace(prompt=prompt, system_prompt=None)))
        ctx = SimpleNamespace(model=route.model.model, client=SimpleNamespace(base_url=endpoint, api_key=route.token, headers={}))
        extra = {}
        if config.harness.id == "omp_agent":
            agent_dir = f"/tmp/vf-omp-agent-{trace.id}"
            models = openai_compat_model_config(endpoint=endpoint, api_key="VF_INTERCEPT_KEY")
            models["providers"]["bench-executor"] = {"baseUrl": executor_endpoint, "api": "openai-completions",
                                                      "apiKey": "BENCH_EXECUTOR_KEY", "models": [{"id": "agent"}]}
            await runtime.write(agent_dir + "/config.yml", json.dumps({"task": {"agentModelOverrides": {
                name: "bench-executor/agent" for name in ("task", "sonic", "scout", "designer", "reviewer", "security-reviewer", "librarian")
            }}}).encode())
            extra["model_config"] = models
        elif config.harness.id == "claude_code_agent":
            # The pinned release's SUBAGENT_MODEL overrides invocation and
            # built-in agent model choices without replacing their prompts.
            route.native_executor_model_alias = (executor.model.model if executor.model.model != route.model.model
                                                 else "bench-native-executor")
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = route.native_executor_model_alias
            for family in ("OPUS", "SONNET", "HAIKU"):
                env[f"ANTHROPIC_DEFAULT_{family}_MODEL"] = route.native_executor_model_alias
        else:
            # Codex labels every spawned thread, including full-history forks,
            # on its native provider requests. The relay selects the helper model
            # from that identity while preserving its built-in roles and prompts.
            route.native_executor_header = ("x-openai-subagent", "collab_spawn")
            # Codex refuses to install its native helper binaries when its home
            # is under /tmp. Keep fresh state in the disposable container's home.
            env["CODEX_HOME"] = "/root/.bench-codex/" + route.agent_id
            await runtime.write(env["CODEX_HOME"] + "/config.toml", b"")
        return await existing_adapter(config, env).launch(ctx, trace, runtime, endpoint, route.token, {}, **extra)
    if config.harness.id == "opencode":
        provider = {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": endpoint, "apiKey": "{env:" + credential + "}"}, "models": {"agent": {"name": route.model.model, "limit": {"context": 1048576, "output": route.model.max_output_tokens}}}}
        cli_config = {"provider": {"bench": provider}, "permission": {"*": "allow", "task": "deny"}, "tools": {"task": False}, "share": "disabled"}
        if native:
            executor_provider = {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": executor_endpoint, "apiKey": "{env:BENCH_EXECUTOR_KEY}"}, "models": {"agent": {"name": executor.model.model, "limit": {"context": 1048576, "output": executor.model.max_output_tokens}}}}
            cli_config = {"provider": {"bench": provider, "bench-executor": executor_provider},
                          "share": "disabled", "agent": {name: {"model": "bench-executor/agent"}
                                                          for name in ("general", "explore")}}
        if mcp_url:
            cli_config["mcp"] = {"bench": {"type": "remote", "url": runtime.host_url(mcp_url), "timeout": 70000}}
            if config.execution_mode == "isolated_tasks":
                cli_config.update(permission={"*": "deny", "bench_*": "allow"}, tools={"*": False, "bench_*": True})
        await runtime.write(state + "/opencode.json", json.dumps(cli_config).encode())
        env.update(OPENCODE_CONFIG=state + "/opencode.json", XDG_CONFIG_HOME=state + "/config", XDG_DATA_HOME=state + "/data", XDG_STATE_HOME=state + "/state")
        argv = [OPENCODE_BIN, "run", "--auto", "--format", "json", "--model", "bench/agent", prompt]
    else:
        cli_config = {
            "model": {"default": route.model.model, "provider": "custom", "base_url": endpoint, "api_key": "${" + credential + "}"},
            "agent": {"max_turns": route.model.max_calls},
            "terminal": {"backend": "local", "cwd": runtime.config.workdir},
        }
        if native:
            cli_config["delegation"] = {"model": executor.model.model, "base_url": executor_endpoint,
                                        "api_key": "${BENCH_EXECUTOR_KEY}", "api_mode": "chat_completions",
                                        "reasoning_effort": executor.model.reasoning_effort}
        if mcp_url:
            cli_config["mcp_servers"] = {"bench": {"url": runtime.host_url(mcp_url)}}
        await runtime.write(state + "/config.yaml", json.dumps(cli_config).encode())
        env.update(HERMES_HOME=state, HERMES_SKIP_NODE_BOOTSTRAP="1", LD_LIBRARY_PATH=HERMES_PYTHON_DEPENDENCY_DIR, HERMES_INTERACTIVE="0")
        argv = [HERMES_BIN, "chat", "--model", route.model.model, "--yolo"]
        if not native:
            toolsets = "terminal,file,mcp-bench" if mcp_url else "terminal,file"
            if mcp_url and config.execution_mode == "isolated_tasks":
                toolsets = "mcp-bench"
            argv.extend(("--toolsets", toolsets))
        argv.extend(("--query", prompt))
    return await runtime.run_program(argv, env)
