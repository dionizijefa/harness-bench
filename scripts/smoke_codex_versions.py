#!/usr/bin/env python3
"""Run Codex version checks against Harbor's hello-world task."""

from harness_bloat_bench.smoke_harness_versions import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            description=__doc__ or "Smoke-test Codex versions",
            harness_id="codex_agent",
            temp_prefix="codex",
        )
    )
