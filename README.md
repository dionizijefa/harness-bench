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
`rollouts` concurrency limit, and starts the UI at http://localhost:3210.
Set `DAGSTER_UI_PORT` or pass an explicit `--port` to use another port.
Additional `dagster dev` options are passed through.

On macOS, install a **Dagster UI** launcher in `~/Applications` and pin it to
the Dock with:

```sh
./scripts/install-dagster-ui-app
```

Clicking the Dock icon runs `scripts/dagster-ui` in Terminal on port 3210 and
opens the UI in the default browser once it is ready.

Open the Dagster UI, select `terminal_bench_rollouts`, and paste this into the Launchpad:

```yaml
ops:
  plan_rollouts:
    config:
      batch_id: experiment-2026-08-15-a
      models: [~deepseek/deepseek-v4-flash-latest]
      harnesses:
        - id: codex_agent
        - id: claude_code_agent
        - id: hermes_agent
        - id: opencode
        - id: pi
        - id: omp_agent
        - id: prime_agent
      task_ids: [adaptive-rejection-sampler]
      num_rollouts: 5
      run_type: full
      container_cpus: 8
      container_memory_gb: 18
      rollout_timeout_seconds: 5400

execution:
  config:
    max_concurrent: 8
```

The `rollouts` pool limit is global across Dagster runs; `max_concurrent` is
the per-run process limit. The UI launcher currently provides eight global
rollout slots. On the 64-thread / 125 GiB worker, Docker rollouts default to a
hard limit of 8 CPUs and 18 GiB each, capping the aggregate CPU quota at 64
threads. Docker memory limits are ceilings rather than reservations, so eight
rollouts can nominally exceed host RAM if they all approach 18 GiB at once;
keep monitoring the recorded peak-memory and OOM fields when increasing
parallelism. Override `container_cpus` or `container_memory_gb` in the
Launchpad when measurements justify a different profile. The ignored `.env`
can override the global slot count with `HARNESS_BLOAT_ROLLOUT_SLOTS`; restart
the launcher after changing it.

Agent execution is capped at 5,400 seconds (90 minutes) per rollout by default.
This framework timeout covers the harness run, while setup and scoring remain
separate stages. Override `rollout_timeout_seconds` for a run, or set it to
`null` to disable the limit.

Every run is automatically tagged in Dagster from its actual configuration.
Set the optional `batch_id` to an experiment identifier shared by every
rollout in the matrix. It is stored on every result and exposed as the
`harness_bloat/batch_id` Dagster run tag, so the same identifier can group
related Dagster runs. Omit it (or set it to `null`) for an unassigned batch.
Set `run_type: test` for a real canary or smoke rollout and `run_type: full` for
a complete benchmark; full is the default, while `dry_run: true` always implies
test. The Runs page can filter on `harness_bloat/run_type`,
`harness_bloat/batch_id`, `harness_bloat/model`, `harness_bloat/harness`,
`harness_bloat/harness_version`, and `harness_bloat/harness_matrix`. The matrix
tag retains exact harness/version pairs, such as
`codex_agent@0.140.0,codex_agent@0.147.0`. The companion
`harness_bloat/dry_run=true|false` tag distinguishes a real test rollout from a
config-only dry run.

`harnesses` pairs a harness ID with an optional version. Omitted versions use reproducible defaults: Codex Agent `0.137.0`, Claude Code `2.1.226`, Hermes Agent `0.20.0`, OpenCode `1.18.1`, Pi `0.80.7`, OMP `17.2.10`, and PrimeAgent `0.7.1`. Pin a different release with, for example, `{id: pi, version: 0.80.6}`. `codex` remains an alias-compatible legacy ID, and the old `harness_versions` list remains accepted for Codex-only configs.

`configs/opencode-versions.yaml` is a ready-to-run matrix spanning the selected
OpenCode `0.1.x` through `1.18.x` releases. OpenCode `0.1.196` has no published
binary; its upstream tag points at the same commit as `0.1.195`, so the adapter
uses the official `0.1.195` artifact for that one matrix label.

`configs/hermes-agent-versions.yaml` is the corresponding Hermes Agent history
matrix for the selected `0.2.0` through `0.20.0` releases. Hermes uses calendar
Git tags for these product versions; the adapter resolves each matrix label to
its exact upstream tag and selects the compatible installer and CLI surface.

`configs/codex-agent-versions.yaml` and `configs/omp-agent-versions.yaml` add
ready-to-run history matrices for the selected Codex Agent `0.78.0` through
`0.147.0` releases and OMP `11.3.0` through `17.2.10` releases. Their adapters
select the CLI flags and configuration shapes supported by each pinned release.

`configs/prime-agent-versions.yaml` covers the modern PrimeAgent lineage from
`0.2.6` through `0.7.1`, and `configs/claude-code-versions.yaml` samples Claude
Code's `2.1.97` through `2.1.226` release history. Claude Code sends Anthropic
Messages requests to Verifiers interception while mapping its primary, alias,
and subagent model settings to the selected matrix model (DeepSeek by default).
Its local plugin ID is `claude_code_agent` because Verifiers reserves
`claude_code` for its built-in adapter, mirroring this project's existing
`codex_agent` naming.

Before launching a full version matrix, run the matching Harbor `hello-world`
smoke check. Both commands use the same Docker limits, timeouts, pass criteria,
OpenRouter endpoint, and default `~deepseek/deepseek-v4-flash-latest` model:

```sh
uv run python scripts/smoke_codex_versions.py 0.147.0
uv run python scripts/smoke_claude_code_versions.py 2.1.226
uv run python scripts/smoke_omp_versions.py 17.2.10
```

Pinned harness artifacts are cached persistently under
`.harness-cache/<harness>/linux-<arch>/<version>/`. Docker rollouts mount that
directory read-only and symlink the selected release into the task container;
other runtimes receive a copy. A missing version is downloaded or built once
under a cross-process lock, and other rollouts reuse it instead of contacting
release hosts or package registries. Hermes Agent and PrimeAgent cache their
complete installed trees, including a portable Python runtime, so their
dependency resolvers also run only during prefetch. Every real rollout performs
this host-side cache preflight before its benchmark timer starts, so a cold
cache cannot inflate one task's recorded runtime. Preload every harness version
in the checked-in matrices ahead of a run with:

```sh
uv run python scripts/cache_harnesses.py
```

Use `--harness omp_agent` (repeatable) to preload a subset. The older
`scripts/cache_omp_versions.py` command remains available for explicit OMP-only
prefetching. `scripts/configure-remote` preloads the full cache on the SSH
worker before it writes the worker configuration.

Pass `--model` to override the model or list multiple versions to check them
concurrently. The commands read `OPENROUTER_API_KEY` from the environment; use
`--api-key-stdin` when invoking them through SSH without persisting the key.

The adapters install the official Linux releases inside each rollout sandbox and route model calls through Verifiers interception. Codex Agent layers historical-version compatibility over Verifiers' stock Codex CLI adapter. Claude Code uses its stock headless coding surface and Anthropic Messages protocol, including remote task MCP servers. Hermes Agent runs its headless chat surface with isolated state, coding tools, task system prompts, and remote task MCP servers. OpenCode and OMP retain their stock coding-agent surfaces and also support task MCP servers. Pi uses its stock coding prompt with all seven documented built-ins (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`), medium thinking, project instructions, and no persisted session. PrimeAgent retains its stock IPython/RLM tool surface with an isolated custom model registry and shared prepared kernel runtime. See the upstream [Codex CLI](https://github.com/openai/codex), [Claude Code](https://github.com/anthropics/claude-code), [Hermes Agent](https://github.com/NousResearch/hermes-agent), [OpenCode CLI](https://opencode.ai/docs/cli/), [Pi usage guide](https://pi.dev/docs/latest/usage), [OMP repository](https://github.com/can1357/oh-my-pi), and [PrimeAgent repository](https://github.com/PrimeIntellect-ai/prime-agent) for the underlying behavior.

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
        timeout_grace_seconds: 1800
        artifact_copy_timeout_seconds: 900
```

A complete non-secret example is checked in at `configs/remote.example.yaml`.

Each mapped rollout starts a remote worker through the local OpenSSH client.
Its stdout is streamed into the local Dagster logs, and its output directory is
copied back to the matching local `outputs/<dagster-run-id>/<rollout>/` path.
The API key named by `api_key_var` is forwarded over the encrypted SSH stdin
stream when it exists locally; otherwise it must already exist in the remote
worker environment. Set `copy_artifacts: false` to keep large trace artifacts
only on the remote host.

Remote SSH commands use batch mode, a 30-second connection timeout, and
30-second keepalives with three missed replies allowed. This closes a half-open
SSH transport in roughly 90 seconds instead of leaving its Dagster rollout slot
occupied indefinitely. The remote worker is also wrapped in a host-side outer
deadline. By default that deadline is `rollout_timeout_seconds + 1800` seconds,
allowing 30 minutes for image setup, scoring, and cleanup around the Verifiers
harness timeout. Set `remote.wall_timeout_seconds` for an explicit wall-clock
limit, or adjust `remote.timeout_grace_seconds`. Artifact copying has its own
900-second limit, configurable with
`remote.artifact_copy_timeout_seconds`.

### Recovering a completed rollout after SSH loss

A worker can finish and write `traces.jsonl` just as its SSH route disappears.
In that case Dagster may report a running step even though the remote worker and
container no longer exist. First verify that the remote trace has
`is_completed: true`; never recover a partial trace. Then copy the trace and
restore its outcome row with:

```sh
./scripts/recover-remote-rollout <dagster-run-id> <rollout-number> [...]
```

The command reads the original mapped spec from local Dagster storage, copies
the finalized remote artifacts, and upserts the recovered score, timing, token
usage, model-call count, and cost into `outputs/results.sqlite`. Resource usage
may be `NULL` because its cgroup summary normally travels in the lost final SSH
message. After recovery, terminate the stale SSH client or cancel the stale
Dagster run to release its concurrency slot. Dagster will still record the
transport failure, while the benchmark outcome remains preserved in the results
database.

On 2026-08-13, run `ac064dd5-f953-434c-a454-ebb321df373f` exposed this failure:
rollouts 363 and 377 had finalized remotely, but a Tailscale route loss left
their local SSH clients half-open for 18 and 17 hours. Rollout 363 had reached
the 90-minute harness timeout with reward 0; rollout 377 had completed with
reward 1. Their traces and result rows were recovered before releasing the two
stale slots. The keepalive and outer-deadline controls above were added in
response.

Remote runs currently require explicit `task_ids`; `task_ids: []` performs
dataset discovery locally and is therefore rejected in SSH mode. The remote
setup command can be rerun to resync after local code or lockfile changes.

`.env`, `.env.*`, and the entire `.dagster/` directory are ignored. The
OpenRouter key, SSH host, username, key paths, and remote filesystem paths must
stay in those private files or in `~/.ssh/config`, never in committed config.

For local runs, let Dagster expand every runnable task in the dataset with
`task_ids: []`. Tasks that require image input are always excluded from both
discovered and explicit task lists. To verify the graph without Docker or model
calls, set `dry_run: true` and provide at least one runnable task ID.

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

The `rollout_results` table records the optional batch ID, model, harness and
exact harness version, dataset, task, rollout number, status, pass/fail result,
reward, runtime, token usage, model-call count, provider-reported USD cost,
trace ID, error details, configured container limits, cumulative container CPU
time, peak container memory, disk I/O, peak process count, and OOM kills.
`resource_usage_source=cgroup_v2` identifies measurements collected from the
remote Docker container's kernel counters. These figures are also shown in
each mapped `run_rollout` step's output metadata in Dagster; `write_results`
shows run-level CPU, peak-memory, and OOM summaries. `usage_source` distinguishes complete provider usage from
partial provider usage and trace-derived fallback counts. Cost and reasoning
tokens remain `NULL` when the provider does not return them.

Provider input-token counts include cached input. For uncached input, subtract
`cached_input_tokens` from `input_tokens`:

```sql
SELECT
  batch_id,
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
  usage_source,
  container_cpu_limit,
  container_memory_limit_gb,
  cpu_seconds,
  ROUND(peak_memory_bytes / 1073741824.0, 2) AS peak_memory_gib,
  io_read_bytes,
  io_write_bytes,
  peak_pids,
  oom_kill_count,
  resource_usage_source
FROM rollout_results
ORDER BY timestamp, harness, task_id, rollout;
```

### Plot OpenCode history

Generate the task outcome matrix, per-task token trends, version-level token
variance, mean token/cost history, and successful-task cost/budget plots from
the cross-run database:

```sh
uv run python scripts/plot_opencode_results.py
```

The command writes PNG plots plus `all_version_successful_spend.csv`,
`version_variance.csv`, `mean_tokens_and_cost_by_version.csv`, and `summary.json`
beneath `outputs/plots/opencode/`.
A task counts as successful in all versions only when it is present in every
selected version and every recorded rollout passed. Repeated rollouts are
represented by their median usage. The budget plot compares output tokens with
the cumulative configured generation allowance (`model_call_count ×` the
per-call harness `max_tokens`); total tokens also include input and cached input.
Version-variance comparisons use only tasks present in every selected version,
so missing task/version pairs cannot skew the historical comparison.

Use `--model` or `--dataset` when the database contains more than one value for
the selected harness, and `--output-dir` to place the generated artifacts
elsewhere.
