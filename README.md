# harness-bloat-bench

Terminal-Bench 2.1 rollouts using [Prime Intellect Verifiers v1](https://github.com/PrimeIntellect-ai/verifiers) with Dagster as the control plane.

Each dynamically mapped `run_rollout` step is exactly one Verifiers rollout. Verifiers supplies the Harbor taskset, selected coding harness, Docker/Prime runtime, scoring, retry policy, and trace format; Dagster supplies scheduling, concurrency, re-execution, logs, and observability.

## Run

```sh
uv sync
cp .env.example .env
# Add OPENROUTER_API_KEY to the private .env file.
./scripts/dagster-ui
```

The launcher creates the local `.dagster` directory, configures the global
`rollouts` concurrency limit, and starts the UI at http://localhost:3000.
Additional `dagster dev` options are passed through, for example
`./scripts/dagster-ui --port 3001`.

Open the Dagster UI, select `terminal_bench_rollouts`, and paste this into the Launchpad:

```yaml
ops:
  plan_rollouts:
    config:
      models: [deepseek/deepseek-v4-flash-latest]
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

## Remote SSH workers

Dagster remains on the local machine while individual rollouts run on a remote
Docker host over SSH. Define the host in `~/.ssh/config` so regular
`ssh terminal-bench` authentication works, then run the one-time setup:

```sh
./scripts/configure-remote terminal-bench
```

This checks the connection, syncs the project, runs `uv sync` remotely, and
writes the host and path to the ignored `.dagster/remote.json`. It uses
`~/harness-bloat-bench` on the worker by default; pass an absolute path as a
second argument to override it. Restart
`./scripts/dagster-ui`, select `terminal_bench_rollouts`, and click **Launch
Run**. The private remote config is automatically used as the job default.

An explicit Launchpad `remote` block overrides the private default when needed:

```yaml
ops:
  plan_rollouts:
    config:
      task_ids: [adaptive-rejection-sampler]
      remote:
        host: terminal-bench
        project_dir: /home/REMOTE_USER/harness-bloat-bench
```

A complete non-secret example is checked in at `configs/remote.example.yaml`.

Each mapped rollout starts a remote worker through the local OpenSSH client.
Its stdout is streamed into the local Dagster logs, and its output directory is
copied back to the matching local `outputs/<dagster-run-id>/<rollout>/` path.
The API key named by `api_key_var` is forwarded over the encrypted SSH stdin
stream when it exists locally; otherwise it must already exist in the remote
worker environment. Set `copy_artifacts: false` to keep large trace artifacts
only on the remote host.

Remote runs currently require explicit `task_ids`; `task_ids: []` performs
dataset discovery locally and is therefore rejected in SSH mode. The remote
setup command can be rerun to resync after local code or lockfile changes.

`.env`, `.env.*`, and the entire `.dagster/` directory are ignored. The
OpenRouter key, SSH host, username, key paths, and remote filesystem paths must
stay in those private files or in `~/.ssh/config`, never in committed config.

For local runs, let Dagster expand every task in the dataset with `task_ids: []`.
To verify the graph without Docker or model calls, set `dry_run: true` and
provide at least one task ID.

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

Verifiers writes `config.toml` and `traces.jsonl` beneath
`outputs/<dagster-run-id>/<rollout>/`. Each rollout is independently upserted
into the queryable cross-run database at `outputs/results.sqlite`, so completed
results and hard failures survive even when another mapped rollout fails. The
final Dagster step repeats the upsert for the complete batch. No separate flat
results file is created.

## Results database

The `rollout_results` table records the model, harness and exact harness
version, dataset, task, rollout number, status, pass/fail result, reward,
runtime, token usage, model-call count, provider-reported USD cost, trace ID,
and error details. `usage_source` distinguishes complete provider usage from
partial provider usage and trace-derived fallback counts. Cost and reasoning
tokens remain `NULL` when the provider does not return them.

Provider input-token counts include cached input. For uncached input, subtract
`cached_input_tokens` from `input_tokens`:

```sql
SELECT
  harness,
  harness_version,
  task_id,
  status,
  passed,
  input_tokens,
  COALESCE(cached_input_tokens, 0) AS cached_input_tokens,
  input_tokens - COALESCE(cached_input_tokens, 0) AS uncached_input_tokens,
  output_tokens,
  reasoning_tokens,
  total_tokens,
  model_call_count,
  cost_usd,
  usage_source
FROM rollout_results
ORDER BY timestamp, harness, task_id, rollout;
```
