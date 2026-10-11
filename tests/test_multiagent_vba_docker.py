"""Opt-in grading against the real v4.0.0 task. No model calls or oracle used."""
import asyncio
import json
import os
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace

import pytest
from verifiers.v1.runtimes.docker import docker

from harness_bloat_bench.multiagent.accounting import Ledger
from harness_bloat_bench.multiagent.config import load_config
from harness_bloat_bench.multiagent.vba_run import VbaRun
from harness_bloat_bench.multiagent.vba import grade_vba, prepare_vba_images
from harness_bloat_bench.multiagent.vba_rubric import TASK_ID

TASK_DIR = os.environ.get("MULTIAGENT_VBA_TASK_DIR")
pytestmark = pytest.mark.skipif(not TASK_DIR, reason="Set MULTIAGENT_VBA_TASK_DIR to a v4.0.0 task export")
ROOT = Path(__file__).resolve().parents[1]

# Deliberately incomplete fixture: only the API contract is implemented. No
# React app, no business behavior, no mutations and no persistence workflow.
BACKEND = '''import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException
app = FastAPI()
assert Path("/shared/legacy_app/README.md").read_text().strip() != "modified"
db = Path(__file__).parent / "data/app.sqlite"
db.parent.mkdir(exist_ok=True)
with sqlite3.connect(db) as conn:
    for entity in ("customers", "assets", "technicians", "parts", "work_orders", "work_order_lines"):
        conn.execute(f"CREATE TABLE IF NOT EXISTS {entity} (id TEXT PRIMARY KEY)")
@app.get("/api/health")
def health(): return {"ok": True}
@app.get("/api/meta")
def meta(): return {"source_app_id": "service_desk_alpha", "target_stack": "react_fastapi_sqlite", "database": {"path": "backend/data/app.sqlite"}}
@app.post("/api/reset")
def reset(): return {"ok": True}
@app.get("/api/entities/{entity}")
def records(entity: str): return []
@app.get("/api/entities/{entity}/{record_id}")
def record(entity: str, record_id: str):
    if entity == "work_orders" and record_id == "WO-0001":
        return {"status": "Draft", "customer_id": "CUST-0001"}
    raise HTTPException(404)
'''


def test_blank_and_backend_only_submissions_in_clean_verifier_containers(tmp_path):
    async def scenario():
        config = load_config(ROOT / "configs/multiagent/hermes-sol-deepseek-vba-userform-port.json")
        info = await docker("info", "--format", "{{.Architecture}}")
        config.sandbox.platform = "linux/arm64" if info.stdout.strip() in ("aarch64", "arm64") else "linux/amd64"
        task = SimpleNamespace(data=SimpleNamespace(task_dir=str(Path(TASK_DIR).resolve()), prompt="Port the legacy app"))
        images = await prepare_vba_images(config, task, tmp_path)
        for backend_only in (False, True):
            output = tmp_path / ("backend-only" if backend_only else "blank")
            output.mkdir()
            ledger = Ledger(output)
            gateway = SimpleNamespace(routes={}, inflight={})
            benchmark = VbaRun(config, output, ledger, gateway, None, task, time.monotonic() + 300,
                              agent_image=images["images"]["agent"], verifier_image=images["images"]["verifier"])
            benchmark.container = benchmark.create_container()
            try:
                await benchmark.container.start()
                check = await benchmark.container.run(["bash", "-c", "test ! -e /tests/test_trace_runner.py && test -f /shared/legacy_app/README.md"], {})
                assert check.exit_code == 0
                if backend_only:
                    await benchmark.container.run(["mkdir", "-p", "/workspace/generated_app/backend"], {})
                    await benchmark.container.write("/workspace/generated_app/backend/main.py", BACKEND.encode())
                    await benchmark.container.write("/workspace/generated_app/backend/requirements.txt", b"fastapi==0.139.0\nuvicorn==0.51.0\n")
                    await benchmark.container.write("/workspace/generated_app/migration_summary.json", json.dumps({"commands": {"backend": 'cd backend && python -m uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT"'}}).encode())
                    # Verifier must exclude caches and use its original inputs.
                    await benchmark.container.run(["bash", "-c", "mkdir -p /workspace/generated_app/.venv; touch /workspace/generated_app/.venv/poison; echo modified > /shared/legacy_app/README.md"], {})
                await grade_vba(benchmark)
                row = benchmark.task_result
                assert row["score_complete"], row
                assert row["reward"] == (.1 if backend_only else 0), row
                assert row["passed"] is False
                assert benchmark.summary()["scored_checks"] == 28
                archive = output / config.name / "tasks" / TASK_ID / "submission.tgz"
                with tarfile.open(archive) as submitted:
                    assert all(".venv" not in Path(name).parts for name in submitted.getnames())
                check = await benchmark.container.run(["test", "!", "-e", "/tests/test_trace_runner.py"], {})
                assert check.exit_code == 0
            finally:
                await asyncio.gather(*(runtime.stop() for runtime in benchmark.runtimes), return_exceptions=True)
                ledger.db.close()
    asyncio.run(scenario())
