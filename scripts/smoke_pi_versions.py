#!/usr/bin/env python3
"""Run Pi checks against Harbor's hello-world task."""

from harness_bloat_bench.smoke_harness_versions import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            description=__doc__ or "Smoke-test Pi versions",
            harness_id="pi_agent",
            temp_prefix="pi-agent",
        )
    )
