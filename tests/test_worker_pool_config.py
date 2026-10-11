from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(SCRIPTS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_private_settings_precedence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    config = load_script("_worker_pool_config.py")
    (tmp_path / ".env").write_text("HOST=env-host\nPORT=3210\n")
    (tmp_path / ".dagster").mkdir()
    (tmp_path / ".dagster/worker-pool.env").write_text("HOST='private-host'\n")
    monkeypatch.delenv("HOST", raising=False)
    assert config.local_settings(tmp_path)["HOST"] == "private-host"
    monkeypatch.setenv("HOST", "exported-host")
    assert config.local_settings(tmp_path)["HOST"] == "exported-host"
    assert config.local_settings(tmp_path)["PORT"] == "3210"


def test_rendered_config_uses_actual_path_and_user(tmp_path):
    output = tmp_path / "runtime/dagster.yaml"
    project = tmp_path / 'project with "quotes"'
    subprocess.run([
        sys.executable, str(SCRIPTS / "render-dagster-config.py"),
        "--project-dir", str(project), "--user", "1234:5678", "--output", str(output),
    ], check=True)
    text = output.read_text()
    assert "__PROJECT" not in text
    kwargs = yaml.safe_load(text)["run_launcher"]["config"]["container_kwargs"]
    assert kwargs["user"] == "1234:5678"
    assert kwargs["working_dir"] == str(project)
    assert kwargs["volumes"] == [f"{project}:{project}"]


def test_renderer_defaults_to_machine_user(tmp_path):
    output = tmp_path / "dagster.yaml"
    subprocess.run([
        sys.executable, str(SCRIPTS / "render-dagster-config.py"), "--output", str(output),
    ], check=True)
    kwargs = yaml.safe_load(output.read_text())["run_launcher"]["config"]["container_kwargs"]
    assert kwargs["user"] == f"{os.getuid()}:{os.getgid()}"
    assert kwargs["working_dir"] == str(SCRIPTS.parent)


def test_status_uses_config_and_quotes_remote_directory(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    status = load_script("worker-pool-status")
    monkeypatch.setattr(status, "local_settings", lambda _: {
        "HARNESS_BLOAT_CONTROL_HOST": "worker-alias",
        "HARNESS_BLOAT_REMOTE_PROJECT_DIR": "/srv/project's directory",
        "HARNESS_BLOAT_UI_URL": "https://worker.example/",
    })
    commands = []
    monkeypatch.setattr(status.subprocess, "run", lambda command, **_: commands.append(command))
    status.main()
    assert len(commands) == 3
    assert all(command[:3] == ["ssh", "--", "worker-alias"] for command in commands)
    assert all("'\"'\"'" in command[3] for command in commands)
    assert "https://worker.example/" in capsys.readouterr().out


def test_deployment_discovers_and_saves_private_settings(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    deploy = load_script("deploy-worker-pool")
    monkeypatch.setattr(deploy, "__file__", str(tmp_path / "scripts/deploy-worker-pool"))
    monkeypatch.setattr(deploy, "local_settings", lambda _: {})
    monkeypatch.setattr(sys, "argv", ["deploy-worker-pool", "--host", "worker-alias"])

    def remote_output(host, command):
        assert host == "worker-alias"
        if '"$HOME"' in command:
            return "/home/worker"
        if command == "hostname":
            return "worker-node"
        if "tailscale ip" in command:
            return "100.64.0.10"
        if "tailscale status" in command:
            return "worker.example"
        if "test ! -f" in command:
            return ""
        if command == "id -un":
            return "worker"
        return "1000"

    commands = []
    monkeypatch.setattr(deploy, "remote_output", remote_output)
    monkeypatch.setattr(deploy, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    monkeypatch.setattr(deploy.subprocess, "run", lambda *_, **__: SimpleNamespace(returncode=1))
    deploy.main()
    private_config = tmp_path / ".dagster/worker-pool.env"
    settings = deploy.parse_env(private_config.read_text())
    assert settings["HARNESS_BLOAT_CONTROL_HOST"] == "worker-alias"
    assert settings["HARNESS_BLOAT_REMOTE_PROJECT_DIR"] == "/home/worker/harness-bloat-bench"
    assert settings["HARNESS_BLOAT_WORKER_NAME"] == "worker-node"
    assert settings["HARNESS_BLOAT_UI_URL"] == "http://worker.example:3210/"
    assert private_config.stat().st_mode & 0o777 == 0o600
    assert any("scripts/render-dagster-config.py" in command[-1] for command, _ in commands)
    remote_env = next(kwargs["input"] for _, kwargs in commands if "input" in kwargs)
    assert "HARNESS_BLOAT_WORKER_NAME=worker-node\n" in remote_env
