import json
import shlex

from verifiers.v1.runtimes import Runtime

INTERCEPT_PROVIDER = "verifiers"
INTERCEPT_MODEL = "model"
INTERCEPT_KEY_VAR = "VF_INTERCEPT_KEY"

NODE_VERSION = "22.23.2"


def release_version(version: str) -> str:
    """Normalize a release version so both ``1.2.3`` and ``v1.2.3`` work."""
    return version.removeprefix("v")


async def run_install(runtime: Runtime, name: str, version: str, script: str) -> None:
    result = await runtime.run(["sh", "-c", script], {})
    if result.exit_code != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:] or "<no output>"
        raise RuntimeError(f"{name} {version} install failed: {detail}")


def shell_assignment(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def node_install_script(node_dir: str) -> str:
    """Return a pinned Node.js installer snippet for Node-based harnesses."""
    return f"""\
{shell_assignment("NODE_VERSION", NODE_VERSION)}
NODE_DIR={shlex.quote(node_dir)}
NODE_BIN="$NODE_DIR/bin/node"

current="$($NODE_BIN --version 2>/dev/null || true)"
if [ "$current" != "v$NODE_VERSION" ]; then
    if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1 || ! command -v xz >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get -o Acquire::Retries=3 update -qq
            apt-get -o Acquire::Retries=3 install -y -qq curl ca-certificates tar xz-utils >/dev/null
        elif command -v apk >/dev/null 2>&1; then
            apk add --no-cache curl ca-certificates tar xz >/dev/null
        else
            echo "Node.js harnesses need curl, CA certificates, tar, and xz" >&2
            exit 1
        fi
    fi

    case "$(uname -m)" in
        aarch64|arm64) node_arch=arm64 ;;
        x86_64|amd64) node_arch=x64 ;;
        *) echo "unsupported Node.js architecture: $(uname -m)" >&2; exit 1 ;;
    esac

    node_archive="$NODE_DIR.tar.xz.tmp"
    node_stage="$NODE_DIR.stage"
    trap 'rm -f "$node_archive"; rm -rf "$node_stage"' EXIT
    rm -rf "$node_stage"
    mkdir -p "$node_stage"
    curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$node_arch.tar.xz" -o "$node_archive"
    tar -xJf "$node_archive" --strip-components=1 -C "$node_stage"
    rm -rf "$NODE_DIR"
    mv "$node_stage" "$NODE_DIR"
fi
"""


def openai_compat_model_config(
    *,
    endpoint: str,
    api_key: str,
) -> dict:
    """Return the minimum Pi/OMP custom-provider definition.

    Pi supplies its own defaults for model capabilities, context size, output
    size, compatibility flags, and cost metadata when they are omitted.
    """
    return {
        "providers": {
            INTERCEPT_PROVIDER: {
                "baseUrl": endpoint,
                "api": "openai-completions",
                "apiKey": api_key,
                "models": [{"id": INTERCEPT_MODEL}],
            }
        }
    }


def json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode()
