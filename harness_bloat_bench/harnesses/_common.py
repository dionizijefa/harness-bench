import json
import shlex

from verifiers.v1.runtimes import Runtime

INTERCEPT_PROVIDER = "verifiers"
INTERCEPT_MODEL = "model"
INTERCEPT_KEY_VAR = "VF_INTERCEPT_KEY"

# Custom providers do not have catalog metadata. These values match Pi's documented
# custom-model defaults, except for the more useful 32K coding output allowance.
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_TOKENS = 32_000


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


def openai_compat_model_config(
    *,
    endpoint: str,
    model_name: str,
    context_window: int,
    max_tokens: int,
    api_key: str,
) -> dict:
    """Return the shared Pi/OMP custom OpenAI-compatible model definition."""
    return {
        "providers": {
            INTERCEPT_PROVIDER: {
                "baseUrl": endpoint,
                "api": "openai-completions",
                "apiKey": api_key,
                "compat": {
                    # The interception trace recognizes ``system`` but not ``developer``.
                    "supportsDeveloperRole": False,
                    # Model capabilities are unknown behind the generic intercepted alias.
                    "supportsReasoningEffort": False,
                    "supportsStore": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [
                    {
                        "id": INTERCEPT_MODEL,
                        "name": model_name,
                        "reasoning": True,
                        "input": ["text"],
                        "contextWindow": context_window,
                        "maxTokens": max_tokens,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }


def json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode()
