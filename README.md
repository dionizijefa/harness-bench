# harness-bloat-bench

Terminal-Bench 2.1 rollouts using [Prime Intellect Verifiers v1](https://github.com/PrimeIntellect-ai/verifiers) with Dagster as the control plane.

Each dynamically mapped `run_rollout` step is exactly one Verifiers rollout. Verifiers supplies the Harbor taskset, selected coding harness, Docker/Prime runtime, scoring, retry policy, and trace format; Dagster supplies scheduling, concurrency, re-execution, logs, and observability.

## Run

```sh
uv sync
export OPENROUTER_API_KEY=sk-or-...
mkdir -p .dagster
export DAGSTER_HOME="$PWD/.dagster"
uv run dagster instance concurrency set rollouts 4
uv run dagster dev -m harness_bloat_bench.definitions
```

Open the Dagster UI, select `terminal_bench_rollouts`, and paste this into the Launchpad:

```yaml
ops:
  plan_rollouts:
    config:
      models: [qwen/qwen3.7-max]
      harnesses:
        - id: codex
        - id: opencode
        - id: pi
        - id: omp_agent
      task_ids: [adaptive-rejection-sampler]
      num_rollouts: 5

execution:
  config:
    max_concurrent: 4
```

The `rollouts` pool limit is global across Dagster runs; `max_concurrent` is the per-run process limit.

`harnesses` pairs a harness ID with an optional version. Omitted versions use reproducible defaults: Codex `0.137.0`, OpenCode `1.18.1`, Pi `0.80.7`, and OMP `16.5.2`. Pin a different release with, for example, `{id: pi, version: 0.80.6}`. The old `harness_versions` list remains accepted for Codex-only configs.

The adapters install the official Linux release artifacts inside each rollout sandbox and route their OpenAI-compatible calls through Verifiers interception. OpenCode and OMP retain their stock coding-agent surfaces and support task MCP servers. Pi uses its stock coding prompt with all seven documented built-ins (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`), medium thinking, project instructions, and no persisted session. See the upstream [OpenCode CLI](https://opencode.ai/docs/cli/), [Pi usage guide](https://pi.dev/docs/latest/usage), and [OMP repository](https://github.com/can1357/oh-my-pi) for the underlying behavior.

The default endpoint is OpenRouter. `base_url` and `api_key_var` are regular matrix config fields if another OpenAI-compatible endpoint is needed. Set `runtime: prime` to use Prime Sandboxes instead of local Docker.

To let Dagster expand every task in the dataset, set `task_ids: []`. To verify the graph without Docker or model calls, set `dry_run: true` and provide at least one task ID.

Headless execution uses the same run config:

```sh
uv run dagster job execute \
  -m harness_bloat_bench.definitions \
  -j terminal_bench_rollouts \
  -c run_config.yaml
```

A checked-in smoke config exercises the full Dagster control plane without model or Docker calls:

```sh
uv run dagster job execute \
  -m harness_bloat_bench.definitions \
  -j terminal_bench_rollouts \
  -c configs/dry-run.yaml
```

Verifiers writes `config.toml` and `traces.jsonl` beneath `outputs/<dagster-run-id>/<rollout>/`. The final Dagster step writes the flat benchmark table to `outputs/<dagster-run-id>/results.jsonl`.
