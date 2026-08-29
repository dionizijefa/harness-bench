FROM docker:29-cli@sha256:000bb62ff495f986c9f5578eb67cc2cb98b91138eda81d7762d5371eb8a497fe AS docker_cli

FROM ghcr.io/astral-sh/uv:0.10.3-python3.12-trixie-slim

COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker_cli /usr/local/libexec/docker/cli-plugins /usr/local/libexec/docker/cli-plugins

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        openssh-client \
        procps \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/harness-bench
COPY pyproject.toml uv.lock README.md ./

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

ENV PATH="/opt/harness-bench/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1
