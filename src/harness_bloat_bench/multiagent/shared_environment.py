"""Combined environment for the four-task pilot, prepared before the run clock."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verifiers.v1.runtimes.docker import docker


TASK_WORKDIRS = {
    "pypi-server": "/app/pypi",
    "fix-code-vulnerability": "/app",
    "qemu-startup": "/app",
    "sanitize-git-repo": "/app/dclm",
}


def combined_dockerfile(images: dict[str, str]) -> str:
    """Copy original task inputs, not solutions or hidden tests, into one image."""
    return f'''FROM {images["qemu-startup"]} AS qemu_input
FROM {images["sanitize-git-repo"]} AS git_input
FROM {images["fix-code-vulnerability"]}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash curl ca-certificates git procps util-linux qemu-system-x86 qemu-utils \\
    wget openssh-client telnet netcat-openbsd expect tmux vim \\
    && rm -rf /var/lib/apt/lists/*
COPY --from=qemu_input /app/alpine.iso /app/alpine.iso
COPY --from=qemu_input /app/alpine-disk.qcow2 /app/alpine-disk.qcow2
COPY --from=git_input /app/dclm /app/dclm
RUN mkdir -p /app/pypi && curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# Match the PyPI task's Python/tooling without replacing Bottle's Python 3.11.
RUN uv venv --python 3.13 --seed /opt/bench-pypi \\
    && uv pip install --python /opt/bench-pypi/bin/python pip==25.2 setuptools==75.6.0 wheel==0.45.1
# Warm verifier dependencies before the timed run. Verifier source stays outside.
RUN pip install pytest==8.4.1 pytest-json-ctrf==0.3.5 \\
    && uvx -p 3.13 -w pytest==8.4.1 -w pip==25.2 -w pytest-json-ctrf==0.3.5 pytest --version \\
    && uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version \\
    && uvx -p 3.13 -w pytest==8.4.1 -w gitpython==3.1.44 -w pytest-json-ctrf==0.3.5 pytest --version
WORKDIR /app
'''


async def prepare_shared_image(config, tasks, image_manifest: dict, output: Path) -> str:
    sources = {}
    for task_id in TASK_WORKDIRS:
        image = tasks[task_id].data.image
        if not image:
            raise ValueError(f"Shared task {task_id} has no source image")
        info = image_manifest[image]
        sources[task_id] = (info.get("RepoDigests") or [image])[0]
    recipe = combined_dockerfile(sources)
    source_info = {task_id: image_manifest[tasks[task_id].data.image] for task_id in TASK_WORKDIRS}
    fingerprint = hashlib.sha256((recipe + config.sandbox.platform + json.dumps(source_info, sort_keys=True)).encode()).hexdigest()[:20]
    tag = "harness-bench-shared:" + fingerprint
    directory = output / "shared-environments" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "Dockerfile").write_text(recipe)
    (directory / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    check = await docker("image", "inspect", tag)
    if check.exit_code:
        result = await docker("build", "--platform", config.sandbox.platform, "--tag", tag, str(directory))
        (directory / "build.log").write_text(result.stdout + result.stderr)
        if result.exit_code:
            raise RuntimeError(f"Combined image build failed; see {directory / 'build.log'}: " + result.stderr[-1500:])
    return tag
