#!/usr/bin/env python3
"""Run Claude Code version checks against Harbor's hello-world task."""

from harness_bloat_bench.smoke_harness_versions import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            description=__doc__ or "Smoke-test Claude Code versions",
            harness_id="claude_code_agent",
            temp_prefix="claude-code",
        )
    )
