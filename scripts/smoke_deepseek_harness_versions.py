#!/usr/bin/env python3
"""Run DeepSeek Harness checks against Harbor's hello-world task."""

from harness_bloat_bench.smoke_harness_versions import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            description=__doc__ or "Smoke-test DeepSeek Harness versions",
            harness_id="deepseek_harness",
            temp_prefix="deepseek-harness",
        )
    )
