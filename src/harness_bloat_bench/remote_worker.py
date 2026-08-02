"""Execute one rollout on an SSH worker and return its row over stdout."""

import json
import os
import sys

from harness_bloat_bench.definitions import REMOTE_RESULT_PREFIX, _execute_rollout


def main() -> None:
    request = json.load(sys.stdin)
    spec = request["spec"]
    spec["remote"] = None
    if api_key := request.get("api_key"):
        os.environ[spec["api_key_var"]] = api_key
    row = _execute_rollout(spec, request["dagster_run_id"])
    print(f"{REMOTE_RESULT_PREFIX}{json.dumps(row, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
