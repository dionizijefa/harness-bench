"""Compatibility name for Verifiers' built-in OpenAI Codex CLI harness."""

from verifiers.v1.harnesses.codex import CodexHarness, CodexHarnessConfig


class CodexAgentHarness(CodexHarness):
    """Expose the stock Codex harness under the explicit ``codex_agent`` ID."""


__all__ = ["CodexAgentHarness", "CodexHarnessConfig"]
