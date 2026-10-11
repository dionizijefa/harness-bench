"""Standalone partial verifier, staged only into the separate grading container.

Uses the pinned upstream trace engine and assertions. API checks launch the
documented backend command without a frontend prerequisite; UI checks use the
submitted run.sh and retain all upstream React/DOM/runtime checks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

APP_ROOT = Path("/workspace/generated_app")
LOG_DIR = Path("/logs/verifier")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_marked_processes(token: str) -> None:
    # AppHarness uses setsid, so killing only the wrapper's process group does
    # not stop the app after a trace timeout. Match inherited environment tags.
    marker = f"BENCH_CHECK_TOKEN={token}".encode()
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            if marker in path.read_bytes().split(b"\0"):
                os.kill(int(path.parent.name), signal.SIGKILL)
        except (OSError, ValueError):
            pass


def command(argv, log: Path, timeout: float, *, cwd=None, env=None) -> int:
    token = uuid4().hex
    with log.open("wb") as stream:
        process = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {}), "BENCH_CHECK_TOKEN": token},
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=max(0.1, timeout))
        except subprocess.TimeoutExpired:
            return 124
        finally:
            stop_marked_processes(token)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def prepare_backend(app: Path, log: Path, timeout: float) -> int:
    requirements = app / "backend/requirements.txt"
    if not requirements.is_file():
        log.write_text("Missing backend/requirements.txt\n")
        return 1
    # Match upstream fresh dependency installation; never reuse an agent venv.
    return command(["bash", "-c", 'set -e; python3 -m venv "$1/.venv"; "$1/.venv/bin/python" -m pip install --no-cache-dir -r "$1/backend/requirements.txt"',
                    "bench-backend", str(app)], log, timeout)


def prepare_frontend(app: Path, log: Path, timeout: float) -> int:
    if not (app / "frontend/package.json").is_file():
        log.write_text("Missing frontend/package.json\n")
        return 1
    return command(["bash", "-c", '''set -e
if [ -f package-lock.json ]; then
    npm ci --no-audit --no-fund || npm install --no-audit --no-fund
else
    npm install --no-audit --no-fund
fi
if [ ! -f dist/index.html ] && node -e 'const p=require("./package.json"); process.exit(p.scripts && p.scripts.build ? 0 : 1)'; then
    npm run build
fi
'''], log, timeout, cwd=app / "frontend")


def check_trace(trace: dict, app: Path, ui: bool) -> dict:
    # This module is imported only in the grading image. No benchmark test code
    # or trace data is ever installed in the agents' shared environment.
    import test_trace_runner as upstream

    if not ui:
        class BackendHarness(upstream.AppHarness):
            def check_static_stack_contract(self):
                pass

            def wait_frontend(self):
                pass

        upstream.AppHarness = BackendHarness
        try:
            summary = json.loads((app / "migration_summary.json").read_text())
            backend = summary["commands"]["backend"]
            if not isinstance(backend, str) or not backend.strip():
                raise ValueError("commands.backend must be a nonempty startup command")
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {"trace_id": trace["trace_id"], "passed": False, "error": f"Backend launch contract: {error}"}
        (app / "run.sh").write_text("#!/usr/bin/env bash\nset -e\n" + backend + "\n")
        (app / "run.sh").chmod(0o755)
    try:
        return upstream.run_traces(app, [trace])[0]
    except Exception as error:
        return {"trace_id": trace["trace_id"], "passed": False, "error": f"app_startup: {error}"}


def run_checks(app: Path, tests: Path, logs: Path, timeout: float) -> list[dict]:
    deadline = time.monotonic() + timeout
    specs = json.loads((tests / "bench-rubric.json").read_text())["subtasks"]
    # Round-robin components so an early slow component cannot monopolize the
    # grading reserve. Every check gets its own fair share of remaining time.
    ids = [spec["trace_ids"][index] for index in range(max(len(spec["trace_ids"]) for spec in specs))
           for spec in specs if index < len(spec["trace_ids"])]
    results = []
    logs.mkdir(parents=True, exist_ok=True)
    progress = logs / "trace_results.json"
    write_json(progress, results)
    backend = prepare_backend(app, logs / "backend-install.log", min(180, max(.1, (deadline - time.monotonic()) / 3)))
    frontend = prepare_frontend(app, logs / "frontend-install.log", min(180, max(.1, (deadline - time.monotonic()) / 3)))
    write_json(logs / "setup.json", {"backend_exit_code": backend, "frontend_exit_code": frontend})
    base_path = f"{app}/.venv/bin:{app}/frontend/node_modules/.bin:" + os.environ["PATH"]
    env = {"APP_BASE_PATH": base_path, "PYTHON_BIN": f"{app}/.venv/bin/python"}
    run_sh = app / "run.sh"
    original_script = run_sh.read_bytes() if run_sh.is_file() else None
    data = app / "backend/data"
    with tempfile.TemporaryDirectory(prefix="bench-data-") as temporary:
        initial_data = Path(temporary) / "data"
        if data.exists():
            shutil.copytree(data, initial_data)
        for index, trace_id in enumerate(ids):
            trace_path = tests / "traces" / (trace_id + ".json")
            trace = json.loads(trace_path.read_text())
            ui = any(step["action"] == "goto" for step in trace["steps"])
            row = {"trace_id": trace_id, "status": "unscored", "passed": None}
            remaining = deadline - time.monotonic()
            dependency = backend or (frontend if ui else 0)
            if remaining <= 0:
                row["error"] = "Grading deadline reached before this check"
            elif dependency:
                row.update(status="unscored" if dependency == 124 else "failed", passed=None if dependency == 124 else False,
                           error="Dependency installation timed out" if dependency == 124 else "Submitted dependency manifest failed; see installation logs")
            else:
                # Restore submitted database files, then let this trace perform
                # its own reset. Persistence is preserved *within* a trace,
                # including restart_app, but never inherited from another one.
                if data.exists():
                    shutil.rmtree(data)
                if initial_data.exists():
                    shutil.copytree(initial_data, data)
                if original_script is None:
                    run_sh.unlink(missing_ok=True)
                else:
                    run_sh.write_bytes(original_script)
                    run_sh.chmod(0o755)
                result_path = logs / "check-result.json"
                result_path.unlink(missing_ok=True)
                check_timeout = min(180, remaining / (len(ids) - index))
                code = command([sys.executable, str(tests / "bench_partial.py"), "--trace", str(trace_path),
                                "--app", str(app), "--result", str(result_path)],
                               logs / (trace_id + ".log"), check_timeout, env=env)
                if code == 0 and result_path.is_file():
                    evidence = json.loads(result_path.read_text())
                    if evidence.get("trace_id") != trace_id or type(evidence.get("passed")) is not bool:
                        raise ValueError("Malformed verifier child evidence")
                    row.update(evidence, status="passed" if evidence["passed"] else "failed")
                else:
                    row["error"] = "Check timed out" if code == 124 else f"Verifier process exited {code} without evidence"
            results.append(row)
            write_json(progress, results)
            print(json.dumps(row), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=APP_ROOT)
    parser.add_argument("--tests", type=Path, default=Path("/tests"))
    parser.add_argument("--logs", type=Path, default=LOG_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.trace:
        trace = json.loads(args.trace.read_text())
        write_json(args.result, check_trace(trace, args.app, any(step["action"] == "goto" for step in trace["steps"])))
    else:
        run_checks(args.app, args.tests, args.logs, args.timeout_seconds)


if __name__ == "__main__":
    main()
