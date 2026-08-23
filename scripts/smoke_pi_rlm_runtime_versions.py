#!/usr/bin/env python3
"""Run pi-rlm-runtime checks against Harbor's hello-world task."""

from harness_bloat_bench.smoke_harness_versions import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            description=__doc__ or "Smoke-test pi-rlm-runtime versions",
            harness_id="pi_rlm_runtime",
            temp_prefix="pi-rlm-runtime",
        )
    )
